//! VM executor: interprets compiled subroutines on a simulation state.
//!
//! The VM maintains a bit vector + phase exponent, and uses call frames with
//! slot-indexed ranges into a shared `BitArena` to manage data flow.
//! All dispatch is via direct array indexing (zero hashing), all register
//! lookups use pre-resolved slot indices embedded in ExternGate/CastOp
//! variants, and SlotIdx resolution requires no SlotInfo lookup at runtime.
//!
//! The `BitArena` is a bump allocator for bit-index arrays. It is cleared
//! once per simulation run and grows to its high-water mark, eliminating
//! all per-call heap allocation from the hot path.

use crate::fastsim::compiler::*;
use crate::fastsim::gates;

#[cfg(feature = "py")]
use pyo3::prelude::*;

/// The simulation state: a flat bit vector plus a phase exponent.
///
/// The phase exponent tracks the value `x` in `exp(iπx)`. To recover the
/// complex phase, compute `exp(iπ * phase_exponent)`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "py", pyo3::pyclass(get_all, set_all, from_py_object))]
pub struct SimState {
    pub bits: Vec<bool>,
    pub phase_exponent: f64,
}

impl SimState {
    pub fn new(capacity_bits: usize) -> Self {
        Self {
            bits: Vec::with_capacity(capacity_bits),
            phase_exponent: 0.0,
        }
    }

    pub fn from_bits(bits: impl IntoIterator<Item = bool>) -> Self {
        let mut state = Self::new(0);
        state.extend_bits(bits);
        state
    }

    pub fn get_bit(&self, index: usize) -> bool {
        assert!(
            index < self.bits.len(),
            "get_bit: index {} out of bounds {}",
            index,
            self.bits.len()
        );
        self.bits[index]
    }

    pub fn set_bit(&mut self, index: usize, val: bool) {
        assert!(
            index < self.bits.len(),
            "set_bit: index {} out of bounds {}",
            index,
            self.bits.len()
        );
        self.bits[index] = val;
    }

    pub fn flip_bit(&mut self, index: usize) {
        assert!(
            index < self.bits.len(),
            "flip_bit: index {} out of bounds {}",
            index,
            self.bits.len()
        );
        self.bits[index] = !self.bits[index];
    }

    pub fn push_bit(&mut self, val: bool) -> usize {
        let idx = self.bits.len();
        self.bits.push(val);
        idx
    }

    pub fn extend_bits(&mut self, vals: impl IntoIterator<Item = bool>) {
        self.bits.extend(vals);
    }

    pub fn extend_false(&mut self, n: usize) {
        self.bits.extend(std::iter::repeat_n(false, n));
    }

    pub fn clear(&mut self) {
        self.bits.clear();
        self.phase_exponent = 0.0;
    }

    pub fn len(&self) -> usize {
        self.bits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }
}

#[cfg(feature = "py")]
#[pymethods]
impl SimState {
    #[new]
    pub fn py_new(capacity_bits: usize) -> Self {
        SimState::new(capacity_bits)
    }

    #[staticmethod]
    #[pyo3(name = "from_bits")]
    pub fn py_from_bits(bits: Vec<bool>) -> Self {
        SimState::from_bits(bits)
    }

    #[pyo3(name = "get_bit")]
    pub fn py_get_bit(&self, index: usize) -> bool {
        self.get_bit(index)
    }

    #[pyo3(name = "set_bit")]
    pub fn py_set_bit(&mut self, index: usize, val: bool) {
        self.set_bit(index, val);
    }

    #[pyo3(name = "flip_bit")]
    pub fn py_flip_bit(&mut self, index: usize) {
        self.flip_bit(index);
    }

    #[pyo3(name = "push_bit")]
    pub fn py_push_bit(&mut self, val: bool) -> usize {
        self.push_bit(val)
    }

    #[pyo3(name = "extend_bits")]
    pub fn py_extend_bits(&mut self, vals: Vec<bool>) {
        self.extend_bits(vals);
    }

    #[pyo3(name = "extend_false")]
    pub fn py_extend_false(&mut self, n: usize) {
        self.extend_false(n);
    }

    #[pyo3(name = "clear")]
    pub fn py_clear(&mut self) {
        self.clear();
    }

    #[pyo3(name = "len")]
    pub fn py_len(&self) -> usize {
        self.len()
    }

    #[pyo3(name = "is_empty")]
    pub fn py_is_empty(&self) -> bool {
        self.is_empty()
    }
}

/// A bump allocator for bit-index arrays, reused across calls within a run.
///
/// All `CallFrame` slots store `(u32, u32)` ranges into this arena instead
/// of individually heap-allocated `Vec<usize>`. The arena is cleared once
/// per simulation run and grows to its high-water mark.
#[derive(Debug, Clone)]
pub struct BitArena {
    data: Vec<usize>,
}

impl BitArena {
    /// Create a new arena with the given initial capacity.
    pub fn new(capacity: usize) -> Self {
        BitArena {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Push a slice of bit indices into the arena. Returns `(start, len)`.
    fn push_slice(&mut self, bits: &[usize]) -> (u32, u32) {
        let start = self.data.len() as u32;
        self.data.extend_from_slice(bits);
        (start, bits.len() as u32)
    }

    /// Push a single bit index. Returns `(start, 1)`.
    fn push_single(&mut self, bit: usize) -> (u32, u32) {
        let start = self.data.len() as u32;
        self.data.push(bit);
        (start, 1)
    }

    /// Push a range of consecutive indices `[start_idx..start_idx+count)`.
    fn push_range(&mut self, start_idx: usize, count: usize) -> (u32, u32) {
        let arena_start = self.data.len() as u32;
        self.data.extend(start_idx..start_idx + count);
        (arena_start, count as u32)
    }

    /// Get a slice of bit indices from a `(start, len)` range.
    fn get(&self, range: (u32, u32)) -> &[usize] {
        let (start, len) = range;
        &self.data[start as usize..(start + len) as usize]
    }

    /// Get a single bit index from a range, asserting length == 1.
    fn get_single(&self, range: (u32, u32)) -> usize {
        assert_eq!(range.1, 1, "Expected exactly 1 bit in arena range");
        self.data[range.0 as usize]
    }

    /// Copy a range within the arena to the end. Used when `resolve_bits`
    /// needs to append slot data that is already in the arena.
    fn copy_range_to_end(&mut self, range: (u32, u32)) {
        let (start, len) = (range.0 as usize, range.1 as usize);
        for i in start..start + len {
            let val = self.data[i];
            self.data.push(val);
        }
    }

    /// Clear the arena, retaining allocated capacity for reuse.
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

/// A call frame tracking variable-to-bit-index mappings via slot-indexed
/// ranges into a shared `BitArena`.
///
/// Each slot corresponds to a variable in the subroutine, indexed by the
/// slot index from `SlotInfo`. Slots store `(u32, u32)` arena ranges instead
/// of heap-allocated `Vec<usize>`, eliminating per-call allocation.
#[derive(Debug)]
pub struct CallFrame {
    /// Slot index → `(start, len)` range into the `BitArena`.
    /// `None` means the slot is not yet populated (or has been deallocated).
    slots: Vec<Option<(u32, u32)>>,
}

impl CallFrame {
    /// Create a new call frame with the given number of slots, all initially empty.
    fn new(n_slots: usize) -> Self {
        CallFrame {
            slots: vec![None; n_slots],
        }
    }

    /// Clear all slots in the call frame without deallocating the underlying vector.
    fn clear(&mut self) {
        self.slots.fill(None);
    }

    /// Resize to `n_slots` and clear all slots. Reuses existing capacity when possible.
    fn resize_and_clear(&mut self, n_slots: usize) {
        self.slots.clear();
        self.slots.resize(n_slots, None);
    }

    /// Set the arena range for a slot.
    fn set_slot(&mut self, slot: usize, range: (u32, u32)) {
        self.slots[slot] = Some(range);
    }

    /// Get the arena range for a slot, panicking if not populated.
    fn get_slot(&self, slot: usize) -> (u32, u32) {
        self.slots[slot].unwrap_or_else(|| {
            panic!(
                "Slot {} not populated in call frame (n_slots={})",
                slot,
                self.slots.len()
            );
        })
    }

    /// Remove (deallocate) the variable at a slot. Returns the arena range.
    fn remove_slot(&mut self, slot: usize) -> Option<(u32, u32)> {
        self.slots[slot].take()
    }

    /// Resolve an ArgMapping to a contiguous arena range.
    fn resolve_arg_mapping(&self, mapping: &ArgMapping, arena: &mut BitArena) -> (u32, u32) {
        self.resolve_bits(&mapping.bits, arena)
    }

    /// Resolve a single SlotIdx to a concrete bit index without arena allocation.
    /// The slot must contain exactly 1 bit.
    fn resolve_single_bit(&self, slot: SlotIdx, arena: &BitArena) -> usize {
        let range = self.get_slot(slot);
        assert_eq!(
            range.1, 1,
            "Expected exactly 1 bit in slot {}, found {}",
            slot, range.1
        );
        arena.data[range.0 as usize]
    }

    /// Resolve a slice of SlotIdx to a contiguous arena range.
    fn resolve_bits(&self, slots: &[SlotIdx], arena: &mut BitArena) -> (u32, u32) {
        let start = arena.data.len() as u32;
        for &slot in slots {
            let range = self.get_slot(slot);
            arena.copy_range_to_end(range);
        }
        let len = arena.data.len() as u32 - start;
        (start, len)
    }

    /// Distribute a parent slot's bits into its element slots.
    /// Each element slot gets exactly 1 bit from the parent.
    /// Element at flat index i maps to offset i from the parent range start (MSB-first).
    fn distribute_elements(
        &mut self,
        parent_slot: SlotIdx,
        element_slots: &[SlotIdx],
        arena: &mut BitArena,
    ) {
        let range = self.get_slot(parent_slot);
        let n = element_slots.len();
        assert_eq!(
            range.1 as usize, n,
            "distribute_elements: parent slot {} has {} bits but {} element slots",
            parent_slot, range.1, n
        );
        for (offset, &elem_slot) in element_slots.iter().enumerate() {
            let bit_val = arena.data[range.0 as usize + offset];
            self.set_slot(elem_slot, arena.push_single(bit_val));
        }
    }
}

/// A reusable virtual machine simulator that preserves allocations across
/// multiple simulation runs. The `BitArena` and frame pool eliminate per-call
/// heap allocation from the hot path.
pub struct VmSimulator<'a> {
    compiled: &'a CompiledModule,
    entrypoint_sub: &'a CompiledSubroutine,
    state: SimState,
    frame: CallFrame,
    arena: BitArena,
    frame_pool: Vec<CallFrame>,
}

impl<'a> VmSimulator<'a> {
    /// Create a new VmSimulator for the given compiled module and entrypoint.
    /// Pre-allocates `SimState`, `CallFrame`, and `BitArena` to be reused across runs.
    pub fn new(compiled: &'a CompiledModule, entrypoint: &str) -> Result<Self, String> {
        let entrypoint_sub = find_subroutine(compiled, entrypoint)?;
        let state = SimState::new(1024);
        let frame = CallFrame::new(entrypoint_sub.slots.n_slots);
        let arena = BitArena::new(4096);
        Ok(Self {
            compiled,
            entrypoint_sub,
            state,
            frame,
            arena,
            frame_pool: Vec::new(),
        })
    }

    /// Execute a single simulation run using the pre-allocated state and frame buffers.
    /// `input_values` maps register names to their bit vectors.
    pub fn execute_run(&mut self, input_values: &[(String, Vec<bool>)]) -> Result<(), String> {
        self.state.clear();
        self.frame.clear();
        self.arena.clear();

        for (name, bits) in input_values {
            let start = self.state.len();
            let n_bits = bits.len();
            self.state.extend_bits(bits.iter().copied());
            let range = self.arena.push_range(start, n_bits);

            let intern_id = self.compiled.intern_table.get_id(name).unwrap_or_else(|| {
                panic!(
                    "Input register '{}' was never interned during compilation",
                    name,
                );
            });
            let slot = *self
                .entrypoint_sub
                .slots
                .intern_to_slot
                .get(&intern_id)
                .unwrap_or_else(|| {
                    panic!(
                        "Input register '{}' (intern_id={}) has no slot in entrypoint subroutine",
                        name, intern_id,
                    );
                });
            self.frame.set_slot(slot, range);

            // Distribute bits to element slots if this is a shaped register.
            // At the entrypoint, we still need element_slots from SlotInfo since
            // there is no compiled instruction to embed this data in.
            if let Some(elem_slots) = self.entrypoint_sub.slots.element_slots.get(&slot) {
                let elem_slots_cloned = elem_slots.clone();
                self.frame
                    .distribute_elements(slot, &elem_slots_cloned, &mut self.arena);
            }
        }

        execute_subroutine(
            self.compiled,
            self.entrypoint_sub,
            &mut self.state,
            &mut self.frame,
            &mut self.arena,
            &mut self.frame_pool,
        )?;
        Ok(())
    }

    /// Extract the output values from the current simulation state and frame.
    pub fn extract_outputs(&self) -> Vec<(String, String, String)> {
        extract_outputs(
            &self.state,
            self.compiled,
            self.entrypoint_sub,
            &self.frame,
            &self.arena,
        )
    }

    /// Get the accumulated phase exponent.
    ///
    /// Returns the value `x` such that the global phase is `exp(iπx)`.
    pub fn phase_exponent(&self) -> f64 {
        self.state.phase_exponent
    }
}

#[cfg(feature = "py")]
#[pyclass(name = "VmSimulator")]
pub struct PyVmSimulator {
    compiled: Py<CompiledModule>,
    entrypoint: String,
    state: SimState,
    frame: CallFrame,
    arena: BitArena,
    frame_pool: Vec<CallFrame>,
}

#[cfg(feature = "py")]
#[pymethods]
impl PyVmSimulator {
    #[new]
    pub fn new(py: Python<'_>, compiled: Py<CompiledModule>, entrypoint: &str) -> PyResult<Self> {
        let compiled_ref = compiled.borrow(py);
        let sim = VmSimulator::new(&compiled_ref, entrypoint)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self {
            compiled: compiled.clone_ref(py),
            entrypoint: entrypoint.to_string(),
            state: sim.state,
            frame: sim.frame,
            arena: sim.arena,
            frame_pool: sim.frame_pool,
        })
    }

    pub fn execute_run(
        &mut self,
        py: Python<'_>,
        input_values: Vec<(String, Vec<bool>)>,
    ) -> PyResult<(Vec<(String, String, String)>, f64)> {
        let compiled_ref = self.compiled.borrow(py);
        let entrypoint_sub = find_subroutine(&compiled_ref, &self.entrypoint)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let mut sim = VmSimulator {
            compiled: &compiled_ref,
            entrypoint_sub,
            state: std::mem::replace(&mut self.state, SimState::new(0)),
            frame: std::mem::replace(&mut self.frame, CallFrame::new(0)),
            arena: std::mem::replace(&mut self.arena, BitArena::new(0)),
            frame_pool: std::mem::replace(&mut self.frame_pool, Vec::new()),
        };
        let res = sim.execute_run(&input_values);
        // Extract outputs and phase exponent before moving state back, while we still
        // hold the borrow on `compiled_ref` and `entrypoint_sub`.
        let outputs = sim.extract_outputs();
        let phase_exponent = sim.phase_exponent();
        self.state = sim.state;
        self.frame = sim.frame;
        self.arena = sim.arena;
        self.frame_pool = sim.frame_pool;
        res.map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok((outputs, phase_exponent))
    }
}

/// Execute a compiled module starting from the given entrypoint.
///
/// This is a one-shot convenience wrapper around `VmSimulator`. For batched or
/// repeated execution, use `VmSimulator` directly.
fn execute_once<'a>(
    compiled: &'a CompiledModule,
    entrypoint: &str,
    input_values: &[(String, Vec<bool>)],
) -> Result<VmSimulator<'a>, String> {
    let mut simulator = VmSimulator::new(compiled, entrypoint)?;
    simulator.execute_run(input_values)?;
    Ok(simulator)
}

/// Find a subroutine by entrypoint name.
///
/// The entrypoint is canonicalized through the parser (e.g. `"Negate()"` →
/// `"Negate"`) before lookup. Aliases are already resolved at compile time.
fn find_subroutine<'a>(
    compiled: &'a CompiledModule,
    entrypoint: &str,
) -> Result<&'a CompiledSubroutine, String> {
    let canonical = crate::parser::canonicalize_bloq_key(entrypoint)
        .map_err(|e| format!("Invalid entrypoint '{}': {}", entrypoint, e))?;
    if let Some(sub) = compiled.get_subroutine(&canonical) {
        return Ok(sub);
    }
    Err(format!(
        "Entrypoint '{}' not found. Available: {:?}",
        canonical,
        compiled
            .subroutines
            .iter()
            .map(|s| compiled.intern_table.resolve(s.bloq_key))
            .collect::<Vec<_>>()
    ))
}

/// Execute a single subroutine in the context of a call frame.
fn execute_subroutine(
    compiled: &CompiledModule,
    sub: &CompiledSubroutine,
    state: &mut SimState,
    frame: &mut CallFrame,
    arena: &mut BitArena,
    frame_pool: &mut Vec<CallFrame>,
) -> Result<(), String> {
    let result = match &sub.body {
        SubroutineBody::Impl(instructions) => {
            execute_impl(compiled, instructions, state, frame, arena, frame_pool)
        }
        SubroutineBody::Extern(gate) => execute_extern_gate(gate, state, frame, arena),
        SubroutineBody::Cast(cast_op) => execute_cast(cast_op, state, frame, arena),
    };
    result.map_err(|e| {
        format!(
            "in '{}': {}",
            compiled.intern_table.resolve(sub.bloq_key),
            e
        )
    })
}

/// Execute an Impl subroutine's instructions.
fn execute_impl(
    compiled: &CompiledModule,
    instructions: &[Instruction],
    state: &mut SimState,
    frame: &mut CallFrame,
    arena: &mut BitArena,
    frame_pool: &mut Vec<CallFrame>,
) -> Result<(), String> {
    for instr in instructions {
        match instr {
            Instruction::Call {
                callee,
                arg_mappings,
                output_mapping,
            } => {
                execute_call(
                    compiled,
                    *callee,
                    arg_mappings,
                    output_mapping,
                    state,
                    frame,
                    arena,
                    frame_pool,
                )?;
            }
            Instruction::Return { ret_mappings } => {
                execute_return(ret_mappings, frame, arena)?;
            }
            Instruction::InlineX { q, lvalue_slot } => {
                let bit = frame.resolve_single_bit(*q, arena);
                state.flip_bit(bit);
                frame.set_slot(*lvalue_slot, arena.push_single(bit));
            }
            Instruction::InlineCNOT {
                ctrl,
                target,
                ctrl_lvalue,
                target_lvalue,
            } => {
                let ctrl_bit = frame.resolve_single_bit(*ctrl, arena);
                let target_bit = frame.resolve_single_bit(*target, arena);
                if state.get_bit(ctrl_bit) {
                    state.flip_bit(target_bit);
                }
                frame.set_slot(*ctrl_lvalue, arena.push_single(ctrl_bit));
                frame.set_slot(*target_lvalue, arena.push_single(target_bit));
            }
            Instruction::InlineToffoli {
                ctrl0,
                ctrl1,
                target,
                ctrl_lvalue,
                ctrl0_lvalue,
                ctrl1_lvalue,
                target_lvalue,
            } => {
                let ctrl0_bit = frame.resolve_single_bit(*ctrl0, arena);
                let ctrl1_bit = frame.resolve_single_bit(*ctrl1, arena);
                let target_bit = frame.resolve_single_bit(*target, arena);
                gates::apply_toffoli(state, &[ctrl0_bit, ctrl1_bit], &[target_bit]);
                let ctrl_range = arena.push_slice(&[ctrl0_bit, ctrl1_bit]);
                frame.set_slot(*ctrl_lvalue, ctrl_range);
                if let Some(c0_lv) = ctrl0_lvalue {
                    frame.set_slot(*c0_lv, arena.push_single(ctrl0_bit));
                }
                if let Some(c1_lv) = ctrl1_lvalue {
                    frame.set_slot(*c1_lv, arena.push_single(ctrl1_bit));
                }
                frame.set_slot(*target_lvalue, arena.push_single(target_bit));
            }
            Instruction::InlineAnd {
                ctrl0,
                ctrl1,
                target_lvalue,
                ctrl_lvalue,
                ctrl0_lvalue,
                ctrl1_lvalue,
                cv,
            } => {
                let ctrl0_bit = frame.resolve_single_bit(*ctrl0, arena);
                let ctrl1_bit = frame.resolve_single_bit(*ctrl1, arena);
                let new_idx = gates::apply_and(state, &[ctrl0_bit, ctrl1_bit], cv);
                let ctrl_range = arena.push_slice(&[ctrl0_bit, ctrl1_bit]);
                frame.set_slot(*ctrl_lvalue, ctrl_range);
                if let Some(c0_lv) = ctrl0_lvalue {
                    frame.set_slot(*c0_lv, arena.push_single(ctrl0_bit));
                }
                if let Some(c1_lv) = ctrl1_lvalue {
                    frame.set_slot(*c1_lv, arena.push_single(ctrl1_bit));
                }
                frame.set_slot(*target_lvalue, arena.push_single(new_idx));
            }
            Instruction::InlineAndDag {
                ctrl0,
                ctrl1,
                target,
                ctrl_lvalue,
                ctrl0_lvalue,
                ctrl1_lvalue,
                cv,
            } => {
                let ctrl0_bit = frame.resolve_single_bit(*ctrl0, arena);
                let ctrl1_bit = frame.resolve_single_bit(*ctrl1, arena);
                let target_bit = frame.resolve_single_bit(*target, arena);
                gates::apply_and_dag(state, &[ctrl0_bit, ctrl1_bit], target_bit, cv);
                let ctrl_range = arena.push_slice(&[ctrl0_bit, ctrl1_bit]);
                frame.set_slot(*ctrl_lvalue, ctrl_range);
                if let Some(c0_lv) = ctrl0_lvalue {
                    frame.set_slot(*c0_lv, arena.push_single(ctrl0_bit));
                }
                if let Some(c1_lv) = ctrl1_lvalue {
                    frame.set_slot(*c1_lv, arena.push_single(ctrl1_bit));
                }
            }
            Instruction::InlineTwoBitCSwap {
                ctrl,
                x,
                y,
                ctrl_lvalue,
                x_lvalue,
                y_lvalue,
            } => {
                let ctrl_bit = frame.resolve_single_bit(*ctrl, arena);
                let x_bit = frame.resolve_single_bit(*x, arena);
                let y_bit = frame.resolve_single_bit(*y, arena);
                gates::apply_twobitcswap(state, &[ctrl_bit], &[x_bit], &[y_bit]);
                frame.set_slot(*ctrl_lvalue, arena.push_single(ctrl_bit));
                frame.set_slot(*x_lvalue, arena.push_single(x_bit));
                frame.set_slot(*y_lvalue, arena.push_single(y_bit));
            }
            Instruction::InlineZ { q, lvalue_slot } => {
                let bit = frame.resolve_single_bit(*q, arena);
                if state.get_bit(bit) {
                    state.phase_exponent += 1.0;
                }
                frame.set_slot(*lvalue_slot, arena.push_single(bit));
            }
            Instruction::InlineS { q, lvalue_slot } => {
                let bit = frame.resolve_single_bit(*q, arena);
                if state.get_bit(bit) {
                    state.phase_exponent += 0.5;
                }
                frame.set_slot(*lvalue_slot, arena.push_single(bit));
            }
            Instruction::InlineT { q, lvalue_slot } => {
                let bit = frame.resolve_single_bit(*q, arena);
                if state.get_bit(bit) {
                    state.phase_exponent += 0.25;
                }
                frame.set_slot(*lvalue_slot, arena.push_single(bit));
            }
            Instruction::InlineCZ {
                q0,
                q1,
                lvalue_slot,
                q0_lvalue,
                q1_lvalue,
            } => {
                let q0_bit = frame.resolve_single_bit(*q0, arena);
                let q1_bit = frame.resolve_single_bit(*q1, arena);
                gates::apply_cz(state, &[q0_bit], &[q1_bit]);
                let q_range = arena.push_slice(&[q0_bit, q1_bit]);
                frame.set_slot(*lvalue_slot, q_range);
                if let Some(q0_lv) = q0_lvalue {
                    frame.set_slot(*q0_lv, arena.push_single(q0_bit));
                }
                if let Some(q1_lv) = q1_lvalue {
                    frame.set_slot(*q1_lv, arena.push_single(q1_bit));
                }
            }
            Instruction::InlineCCZ {
                q0,
                q1,
                q2,
                lvalue_slot,
                q0_lvalue,
                q1_lvalue,
                q2_lvalue,
            } => {
                let q0_bit = frame.resolve_single_bit(*q0, arena);
                let q1_bit = frame.resolve_single_bit(*q1, arena);
                let q2_bit = frame.resolve_single_bit(*q2, arena);
                gates::apply_ccz(state, &[q0_bit], &[q1_bit], &[q2_bit]);
                let q_range = arena.push_slice(&[q0_bit, q1_bit, q2_bit]);
                frame.set_slot(*lvalue_slot, q_range);
                if let Some(q0_lv) = q0_lvalue {
                    frame.set_slot(*q0_lv, arena.push_single(q0_bit));
                }
                if let Some(q1_lv) = q1_lvalue {
                    frame.set_slot(*q1_lv, arena.push_single(q1_bit));
                }
                if let Some(q2_lv) = q2_lvalue {
                    frame.set_slot(*q2_lv, arena.push_single(q2_bit));
                }
            }
            Instruction::InlineAllocate {
                n_bits,
                lvalue_slot,
                element_lvalue_slots,
            } => {
                let start = state.len();
                state.extend_false(*n_bits);
                let range = arena.push_range(start, *n_bits);
                frame.set_slot(*lvalue_slot, range);
                if !element_lvalue_slots.is_empty() {
                    frame.distribute_elements(*lvalue_slot, element_lvalue_slots, arena);
                }
            }
            Instruction::InlineZeroState { lvalue_slot } => {
                let idx = state.push_bit(false);
                frame.set_slot(*lvalue_slot, arena.push_single(idx));
            }
            Instruction::InlineOneState { lvalue_slot } => {
                let idx = state.push_bit(true);
                frame.set_slot(*lvalue_slot, arena.push_single(idx));
            }
            Instruction::InlineFree { n_bits, in_bits } => {
                let range = frame.resolve_bits(in_bits, arena);
                let bits = arena.get(range);
                assert_eq!(bits.len(), *n_bits, "Free expects expected number of bits");
                for &idx in bits {
                    if state.get_bit(idx) {
                        return Err(format!("Free: bit {} is 1 but should be 0", idx));
                    }
                }
            }
            Instruction::InlineZeroEffect { q } => {
                let bit = frame.resolve_single_bit(*q, arena);
                if state.get_bit(bit) {
                    return Err(format!("ZeroEffect: bit {} is 1 but should be 0", bit));
                }
            }
            Instruction::InlineOneEffect { q } => {
                let bit = frame.resolve_single_bit(*q, arena);
                if !state.get_bit(bit) {
                    return Err(format!("OneEffect: bit {} is 0 but should be 1", bit));
                }
            }
        }
    }
    Ok(())
}

/// Execute a call instruction: create a child frame, execute the subroutine, map outputs.
///
/// Subroutine dispatch is a direct array index — zero hashing, zero alias resolution.
/// Child frames are drawn from a reusable pool to avoid per-call heap allocation.
fn execute_call(
    compiled: &CompiledModule,
    target: usize,
    arg_mappings: &[ArgMapping],
    output_mapping: &OutputMapping,
    state: &mut SimState,
    caller_frame: &mut CallFrame,
    arena: &mut BitArena,
    frame_pool: &mut Vec<CallFrame>,
) -> Result<(), String> {
    // Direct array index — the target sub_id was resolved at compile time.
    let sub = &compiled.subroutines[target];

    // VM-level short-circuit for single-register Cast operations (Split, Join,
    // Identity Cast). Bypasses child frame creation entirely. Multi-register
    // Partition casts do NOT qualify — they have multiple arg/output mappings
    // and fall through to the general child-frame path below.
    if let SubroutineBody::Cast(cast_op) = &sub.body {
        let single_reg = match cast_op {
            CastOp::Split { total_bits, .. } => Some((*total_bits, "Split")),
            CastOp::Join { total_bits, .. } => Some((*total_bits, "Join")),
            CastOp::Partition { .. } => None,
        };

        if let Some((total_bits, op_name)) = single_reg {
            assert_eq!(
                arg_mappings.len(),
                1,
                "Cast subroutine must have exactly 1 arg mapping"
            );
            let bits_range = caller_frame.resolve_arg_mapping(&arg_mappings[0], arena);

            if bits_range.1 as usize != total_bits {
                return Err(format!(
                    "in '{}': {}: expected {} bits but found {}",
                    compiled.intern_table.resolve(sub.bloq_key),
                    op_name,
                    total_bits,
                    bits_range.1
                ));
            }

            match output_mapping {
                OutputMapping::Direct(pairs) => {
                    assert_eq!(
                        pairs.len(),
                        1,
                        "Cast subroutine must have exactly 1 output mapping pair"
                    );
                    let caller_slot = pairs[0].caller;
                    caller_frame.set_slot(caller_slot, bits_range);
                    // Distribute element slots for Cast outputs
                    if let Some(elem_slots) = &pairs[0].caller_elements {
                        caller_frame.distribute_elements(caller_slot, elem_slots, arena);
                    }
                }
                OutputMapping::Concat { .. } => {
                    panic!("Cast subroutine output cannot be Concat");
                }
            }
            return Ok(());
        }
    }

    // Build the child frame from the pool (reuses allocation when available).
    let mut child_frame = frame_pool.pop().unwrap_or_else(|| CallFrame::new(0));
    child_frame.resize_and_clear(sub.slots.n_slots);

    for mapping in arg_mappings {
        let range = caller_frame.resolve_arg_mapping(mapping, arena);
        child_frame.set_slot(mapping.callee_slot, range);
        // Distribute element slots for callee signature registers
        if let Some(elem_slots) = &mapping.callee_elements {
            child_frame.distribute_elements(mapping.callee_slot, elem_slots, arena);
        }
    }

    // Execute the subroutine
    execute_subroutine(compiled, sub, state, &mut child_frame, arena, frame_pool)?;

    // Map outputs back to caller's lvalues.
    match output_mapping {
        OutputMapping::Direct(pairs) => {
            for pair in pairs {
                let range = child_frame.remove_slot(pair.callee).unwrap_or_else(|| {
                    panic!(
                        "Call to '{}': output slot {} is empty",
                        compiled.intern_table.resolve(sub.bloq_key),
                        pair.callee,
                    );
                });
                caller_frame.set_slot(pair.caller, range);
                // Distribute element slots for this output
                if let Some(elem_slots) = &pair.caller_elements {
                    caller_frame.distribute_elements(pair.caller, elem_slots, arena);
                }
            }
        }
        OutputMapping::Concat {
            callee_slots,
            caller_slot,
            caller_elements,
        } => {
            let start = arena.data.len() as u32;
            for &callee_slot in callee_slots {
                let range = child_frame.remove_slot(callee_slot).unwrap_or_else(|| {
                    panic!(
                        "Call to '{}': output slot {} is empty (concat case)",
                        compiled.intern_table.resolve(sub.bloq_key),
                        callee_slot,
                    );
                });
                arena.copy_range_to_end(range);
            }
            let len = arena.data.len() as u32 - start;
            caller_frame.set_slot(*caller_slot, (start, len));
            // Distribute element slots for Concat output
            if let Some(elem_slots) = caller_elements {
                caller_frame.distribute_elements(*caller_slot, elem_slots, arena);
            }
        }
    }

    // Return child frame to pool for reuse.
    frame_pool.push(child_frame);

    Ok(())
}

/// Execute a return instruction: remap variables to match signature names.
fn execute_return(
    ret_mappings: &[ArgMapping],
    frame: &mut CallFrame,
    arena: &mut BitArena,
) -> Result<(), String> {
    let mut new_mappings: Vec<(usize, (u32, u32))> = Vec::with_capacity(ret_mappings.len());

    for mapping in ret_mappings {
        let range = frame.resolve_arg_mapping(mapping, arena);
        new_mappings.push((mapping.callee_slot, range));
    }

    for (slot, range) in new_mappings {
        frame.set_slot(slot, range);
    }

    Ok(())
}

/// Execute an extern gate operation using pre-resolved slot indices.
///
/// No `CompiledModule` or `CompiledSubroutine` needed — all slot indices
/// are embedded directly in the `ExternGate` variant.
fn execute_extern_gate(
    gate: &ExternGate,
    state: &mut SimState,
    frame: &mut CallFrame,
    arena: &mut BitArena,
) -> Result<(), String> {
    match gate {
        ExternGate::XGate { q_slot } => {
            let q_range = frame.get_slot(*q_slot);
            gates::apply_x(state, arena.get(q_range));
        }
        ExternGate::CNOT {
            ctrl_slot,
            target_slot,
        } => {
            let ctrl_range = frame.get_slot(*ctrl_slot);
            let target_range = frame.get_slot(*target_slot);
            gates::apply_cnot(state, arena.get(ctrl_range), arena.get(target_range));
        }
        ExternGate::Toffoli {
            ctrl_slot,
            target_slot,
        } => {
            let ctrl_range = frame.get_slot(*ctrl_slot);
            let target_range = frame.get_slot(*target_slot);
            gates::apply_toffoli(state, arena.get(ctrl_range), arena.get(target_range));
        }
        ExternGate::And {
            ctrl_slot,
            target_slot,
            cv,
        } => {
            // Arena-based: no clone needed — data lives in the arena, not the frame.
            let ctrl_range = frame.get_slot(*ctrl_slot);
            let new_idx = gates::apply_and(state, arena.get(ctrl_range), cv);
            frame.set_slot(*target_slot, arena.push_single(new_idx));
        }
        ExternGate::AndDag {
            ctrl_slot,
            target_slot,
            cv,
        } => {
            let ctrl_range = frame.get_slot(*ctrl_slot);
            let target_range = frame.get_slot(*target_slot);
            assert_eq!(target_range.1, 1, "And_dag target should be 1 bit");
            let target_bit = arena.get_single(target_range);
            gates::apply_and_dag(state, arena.get(ctrl_range), target_bit, cv);
            // Target is deallocated — remove from frame.
            frame.remove_slot(*target_slot);
        }
        ExternGate::TwoBitCSwap {
            ctrl_slot,
            x_slot,
            y_slot,
        } => {
            let ctrl_range = frame.get_slot(*ctrl_slot);
            let x_range = frame.get_slot(*x_slot);
            let y_range = frame.get_slot(*y_slot);
            gates::apply_twobitcswap(
                state,
                arena.get(ctrl_range),
                arena.get(x_range),
                arena.get(y_range),
            );
        }
        ExternGate::ZGate { q_slot } => {
            let q_range = frame.get_slot(*q_slot);
            gates::apply_z(state, arena.get(q_range));
        }
        ExternGate::SGate { q_slot } => {
            let q_range = frame.get_slot(*q_slot);
            gates::apply_s(state, arena.get(q_range));
        }
        ExternGate::TGate { q_slot } => {
            let q_range = frame.get_slot(*q_slot);
            gates::apply_t(state, arena.get(q_range));
        }
        ExternGate::CZGate { q_slot } => {
            let q_range = frame.get_slot(*q_slot);
            let q_bits = arena.get(q_range);
            assert_eq!(q_bits.len(), 2, "CZGate expects 2 qubits");
            gates::apply_cz(state, &q_bits[0..1], &q_bits[1..2]);
        }
        ExternGate::CCZGate { q_slot } => {
            let q_range = frame.get_slot(*q_slot);
            let q_bits = arena.get(q_range);
            assert_eq!(q_bits.len(), 3, "CCZGate expects 3 qubits");
            gates::apply_ccz(state, &q_bits[0..1], &q_bits[1..2], &q_bits[2..3]);
        }
        ExternGate::Allocate { n_bits, out_slot } => {
            let start = state.len();
            state.extend_false(*n_bits);
            let range = arena.push_range(start, *n_bits);
            frame.set_slot(*out_slot, range);
        }
        ExternGate::Free { n_bits: _, in_slot } => {
            {
                let range = frame.get_slot(*in_slot);
                let bits = arena.get(range);
                for &idx in bits {
                    if state.get_bit(idx) {
                        return Err(format!("Free: bit {} is 1 but should be 0", idx));
                    }
                }
            }
            frame.remove_slot(*in_slot);
        }
        ExternGate::ZeroState { q_slot } => {
            let idx = state.push_bit(false);
            frame.set_slot(*q_slot, arena.push_single(idx));
        }
        ExternGate::ZeroEffect { q_slot } => {
            {
                let range = frame.get_slot(*q_slot);
                assert_eq!(range.1, 1, "ZeroEffect expects exactly 1 bit");
                let bit = arena.get_single(range);
                if state.get_bit(bit) {
                    return Err(format!("ZeroEffect: bit {} is 1 but should be 0", bit));
                }
            }
            frame.remove_slot(*q_slot);
        }
        ExternGate::OneState { q_slot } => {
            let idx = state.push_bit(true);
            frame.set_slot(*q_slot, arena.push_single(idx));
        }
        ExternGate::OneEffect { q_slot } => {
            {
                let range = frame.get_slot(*q_slot);
                assert_eq!(range.1, 1, "OneEffect expects exactly 1 bit");
                let bit = arena.get_single(range);
                if !state.get_bit(bit) {
                    return Err(format!("OneEffect: bit {} is 0 but should be 1", bit));
                }
            }
            frame.remove_slot(*q_slot);
        }
    }
    Ok(())
}

/// Execute a cast operation using pre-resolved slot indices.
///
/// No `CompiledSubroutine` needed — slot indices are embedded in `CastOp`.
fn execute_cast(
    cast_op: &CastOp,
    _state: &mut SimState,
    frame: &mut CallFrame,
    arena: &mut BitArena,
) -> Result<(), String> {
    match cast_op {
        CastOp::Split {
            total_bits,
            reg_slot,
        } => {
            let range = frame.get_slot(*reg_slot);
            if range.1 as usize != *total_bits {
                return Err(format!(
                    "Split: expected {} bits but found {}",
                    total_bits, range.1
                ));
            }
        }
        CastOp::Join {
            total_bits,
            reg_slot,
        } => {
            let range = frame.get_slot(*reg_slot);
            if range.1 as usize != *total_bits {
                return Err(format!(
                    "Join: expected {} bits but found {}",
                    total_bits, range.1
                ));
            }
        }
        CastOp::Partition {
            input_slots,
            output_slots,
            total_bits,
        } => {
            // Gather every input register's bits into a single contiguous arena
            // range, preserving signature order. This is the flattened bit
            // stream shared by both sides of the partition.
            let gather_start = arena.data.len() as u32;
            for &(slot, n_bits) in input_slots {
                let range = frame.get_slot(slot);
                if range.1 as usize != n_bits {
                    return Err(format!(
                        "Partition: input slot {} expected {} bits but found {}",
                        slot, n_bits, range.1
                    ));
                }
                arena.copy_range_to_end(range);
            }
            let gathered_len = arena.data.len() as u32 - gather_start;
            if gathered_len as usize != *total_bits {
                return Err(format!(
                    "Partition: gathered {} bits but expected {}",
                    gathered_len, total_bits
                ));
            }

            // Scatter the flattened stream out to the output registers in
            // signature order. Each output slot receives its own arena range
            // so subsequent slot mutations stay independent.
            let mut offset = gather_start as usize;
            for &(slot, n_bits) in output_slots {
                let bits: Vec<usize> = arena.data[offset..offset + n_bits].to_vec();
                let out_range = arena.push_slice(&bits);
                frame.set_slot(slot, out_range);
                offset += n_bits;
            }
        }
    }
    Ok(())
}

/// Convert an integer to a big-endian bit vector.
/// Index 0 = MSB, index n-1 = LSB.
///
/// Panics if n_bits > 63. For larger bit widths, use `decimal_str_to_bits`.
pub fn int_to_bits(value: i64, n_bits: usize) -> Vec<bool> {
    assert!(
        n_bits <= 63,
        "int_to_bits: n_bits={} exceeds i64 capacity (max 63). Use decimal_str_to_bits for larger values.",
        n_bits
    );
    if value < 0 {
        let min_neg = if n_bits == 0 { 0 } else { -(1i64 << (n_bits - 1)) };
        assert!(
            value >= min_neg,
            "int_to_bits: value {} exceeds minimum representable negative value for {} bits",
            value,
            n_bits
        );
    } else if n_bits < 63 {
        let max_pos = (1i64 << n_bits) - 1;
        assert!(
            value <= max_pos,
            "int_to_bits: value {} exceeds maximum representable value for {} bits",
            value,
            n_bits
        );
    }
    // Treat value as unsigned for bit representation
    let unsigned_val = if value < 0 {
        // Two's complement: value + 2^n_bits
        ((1i64 << n_bits) + value) as u64
    } else {
        value as u64
    };

    let mut bits = vec![false; n_bits];
    for (i, bit) in bits.iter_mut().enumerate().take(n_bits) {
        // bit index i corresponds to significance 2^(n_bits-1-i)
        let bit_pos = n_bits - 1 - i;
        *bit = (unsigned_val >> bit_pos) & 1 == 1;
    }
    bits
}

/// Convert a big-endian bit vector to an unsigned integer.
///
/// Panics if bits.len() > 64. For larger bit widths, use `bits_to_uint_str`.
pub fn bits_to_uint(bits: &[bool]) -> u64 {
    assert!(
        bits.len() <= 64,
        "bits_to_uint: {} bits exceeds u64 capacity (max 64). Use bits_to_uint_str for larger values.",
        bits.len()
    );
    let mut val: u64 = 0;
    for (i, &bit) in bits.iter().enumerate() {
        if bit {
            val |= 1 << (bits.len() - 1 - i);
        }
    }
    val
}

/// Convert a big-endian bit vector to a signed integer (two's complement).
///
/// Panics if bits.len() > 63. For larger bit widths, use `bits_to_int_str`.
pub fn bits_to_int(bits: &[bool]) -> i64 {
    assert!(
        bits.len() <= 63,
        "bits_to_int: {} bits exceeds i64 capacity (max 63). Use bits_to_int_str for larger values.",
        bits.len()
    );
    let unsigned = bits_to_uint(bits);
    let n = bits.len();
    if n == 0 {
        return 0;
    }
    let max_positive = 1u64 << (n - 1);
    if unsigned >= max_positive {
        unsigned as i64 - (1i64 << n)
    } else {
        unsigned as i64
    }
}

/// Extract the output values from a SimState using the subroutine's signature
/// and the call frame's slot-indexed bit mappings.
fn extract_outputs(
    state: &SimState,
    compiled: &CompiledModule,
    sub: &CompiledSubroutine,
    frame: &CallFrame,
    arena: &BitArena,
) -> Vec<(String, String, String)> {
    let mut results = Vec::new();

    for reg in &sub.signature {
        if reg.direction == RegisterDirection::LeftOnly {
            // This was freed during execution, skip
            continue;
        }

        // Look up the register's bit indices from the call frame via slot
        let slot = *sub.slots.intern_to_slot.get(&reg.name).unwrap_or_else(|| {
            panic!(
                "extract_outputs: register '{}' has no slot assignment",
                compiled.intern_table.resolve(reg.name),
            );
        });
        let range = frame.get_slot(slot);
        let indices = arena.get(range);

        // Read bits from state using the frame's index mapping
        let bit_values: Vec<bool> = indices.iter().map(|&i| state.get_bit(i)).collect();

        let dtype_str = compiled.intern_table.resolve(reg.dtype_name);
        let value_str = if reg.is_signed(&compiled.intern_table) {
            bits_to_int_str(&bit_values)
        } else {
            bits_to_uint_str(&bit_values)
        };

        let name_str = compiled.intern_table.resolve(reg.name).to_string();
        results.push((name_str, value_str, dtype_str.to_string()));
    }

    results
}

/// Convert a non-negative decimal string to a big-endian bit vector.
///
/// Algorithm: Repeated divide-by-2 on decimal digit array.
pub fn decimal_str_to_bits(s: &str, n_bits: usize) -> Vec<bool> {
    // Parse decimal string into Vec<u8> of digits
    let mut digits: Vec<u8> = s
        .bytes()
        .map(|b| {
            assert!(b >= b'0' && b <= b'9', "Invalid digit in decimal string");
            b - b'0'
        })
        .collect();

    // Repeatedly divide by 2, collecting remainders as LSBs
    let mut lsb_first: Vec<bool> = Vec::new();
    while !digits.is_empty() {
        // Check if all digits are zero
        let all_zero = digits.iter().all(|&d| d == 0);
        if all_zero {
            break;
        }

        // Divide digits by 2, remainder is the next LSB
        let mut remainder: u8 = 0;
        for d in digits.iter_mut() {
            let cur = remainder * 10 + *d;
            *d = cur / 2;
            remainder = cur % 2;
        }
        lsb_first.push(remainder == 1);

        // Strip leading zeros
        while digits.first() == Some(&0) {
            digits.remove(0);
        }
    }

    // Pad to n_bits (LSB-first, then reverse for big-endian)
    while lsb_first.len() < n_bits {
        lsb_first.push(false);
    }
    if lsb_first.len() > n_bits {
        assert!(
            !lsb_first[n_bits..].iter().any(|&b| b),
            "exceeds maximum value for bit width"
        );
        lsb_first.truncate(n_bits);
    }
    // Reverse to get big-endian (MSB first)
    lsb_first.reverse();
    lsb_first
}

/// Convert a big-endian bit vector to an unsigned decimal string.
///
/// Algorithm: MSB-to-LSB traversal, maintaining result as Vec<u8> decimal digits.
pub fn bits_to_uint_str(bits: &[bool]) -> String {
    if bits.is_empty() {
        return "0".to_string();
    }
    // result stores decimal digits, least-significant first
    let mut result: Vec<u8> = vec![0];

    for &bit in bits.iter() {
        // Double all digits (multiply by 2)
        let mut carry: u8 = 0;
        for d in result.iter_mut() {
            let val = *d * 2 + carry;
            *d = val % 10;
            carry = val / 10;
        }
        if carry > 0 {
            result.push(carry);
        }

        // Add bit value to least-significant digit
        if bit {
            let mut carry: u8 = 1;
            for d in result.iter_mut() {
                let val = *d + carry;
                *d = val % 10;
                carry = val / 10;
                if carry == 0 {
                    break;
                }
            }
            if carry > 0 {
                result.push(carry);
            }
        }
    }

    // Convert digits to string (result is LSB-first, so reverse)
    result.iter().rev().map(|d| (b'0' + d) as char).collect()
}

/// Helper: add one to a big-endian bit vector in place.
fn add_one_to_bits(bits: &mut [bool]) {
    // Carry propagation from LSB (index n-1) to MSB
    for i in (0..bits.len()).rev() {
        if bits[i] {
            bits[i] = false; // 1+1 = 10, carry continues
        } else {
            bits[i] = true;
            return;
        }
    }
    // If carry overflows, we've wrapped around (all zeros). That's fine for 2's complement.
}

/// Convert a big-endian bit vector to a signed decimal string (two's complement).
pub fn bits_to_int_str(bits: &[bool]) -> String {
    if bits.is_empty() {
        return "0".to_string();
    }
    // If MSB=0: positive, return unsigned string
    if !bits[0] {
        return bits_to_uint_str(bits);
    }
    // MSB=1: negative. Invert all bits, add 1, convert to unsigned, prepend "-"
    let mut inverted: Vec<bool> = bits.iter().map(|&b| !b).collect();
    add_one_to_bits(&mut inverted);
    let magnitude = bits_to_uint_str(&inverted);
    if magnitude == "0" {
        return "0".to_string();
    }
    format!("-{}", magnitude)
}

/// Convert a possibly-negative decimal string to bits using two's complement.
pub fn signed_decimal_str_to_bits(s: &str, n_bits: usize) -> Vec<bool> {
    if let Some(positive_part) = s.strip_prefix('-') {
        if positive_part == "0" {
            return vec![false; n_bits];
        }
        // Parse positive part, convert to bits, then two's complement (invert + add 1)
        let mut bits = decimal_str_to_bits(positive_part, n_bits);
        // Check for signed overflow: if MSB is 1, no other bit can be 1
        assert!(
            bits.is_empty() || !bits[0] || !bits[1..].iter().any(|&b| b),
            "Value {} exceeds maximum negative value for {} bits",
            s,
            n_bits
        );
        // Invert all bits
        for b in bits.iter_mut() {
            *b = !*b;
        }
        // Add 1
        add_one_to_bits(&mut bits);
        bits
    } else {
        let bits = decimal_str_to_bits(s, n_bits);
        assert!(
            bits.is_empty() || !bits[0],
            "Value {} exceeds maximum positive value for {} signed bits",
            s,
            n_bits
        );
        bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fastsim::compiler;
    use crate::parser;

    fn run_negate(input: i64) -> (f64, Vec<(String, String, String)>) {
        let source = std::fs::read_to_string("example_qlts/negate.qlt").unwrap();
        let (module, errors) = parser::parse_l1_module(&source);
        assert!(errors.is_empty());

        let compiled = compiler::compile(&module).unwrap();

        let bits = int_to_bits(input, 8);
        let input_values = vec![("x".to_string(), bits)];
        let sim = execute_once(&compiled, "Negate", &input_values).unwrap();
        let outputs = sim.extract_outputs();
        (sim.phase_exponent(), outputs)
    }

    fn run_cswap(ctrl: i64, x: i64, y: i64) -> Vec<(String, String, String)> {
        let source = std::fs::read_to_string("example_qlts/cswap.qlt").unwrap();
        let (module, errors) = parser::parse_l1_module(&source);
        assert!(errors.is_empty());

        let compiled = compiler::compile(&module).unwrap();

        let input_values = vec![
            ("ctrl".to_string(), int_to_bits(ctrl, 1)),
            ("x".to_string(), int_to_bits(x, 5)),
            ("y".to_string(), int_to_bits(y, 5)),
        ];
        let sim = execute_once(&compiled, "CSwap", &input_values).unwrap();
        sim.extract_outputs()
    }

    #[test]
    fn test_negate_5() {
        let (phase_exponent, outputs) = run_negate(5);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].0, "x");
        assert_eq!(outputs[0].1, "-5");
        assert!((phase_exponent - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_negate_0() {
        let (_phase_exponent, outputs) = run_negate(0);
        assert_eq!(outputs[0].1, "0");
    }

    #[test]
    fn test_negate_1() {
        let (_phase_exponent, outputs) = run_negate(1);
        assert_eq!(outputs[0].1, "-1");
    }

    #[test]
    fn test_negate_127() {
        let (_phase_exponent, outputs) = run_negate(127);
        assert_eq!(outputs[0].1, "-127");
    }

    #[test]
    fn test_negate_128() {
        let (_phase_exponent, outputs) = run_negate(128);
        assert_eq!(outputs[0].1, "-128");
    }

    #[test]
    fn test_negate_all_256() {
        for i in 0u16..256 {
            let input = i as i64;
            let (phase_exponent, outputs) = run_negate(input);

            let expected_unsigned = ((256u16 - i) % 256) as u64;

            let expected_signed = if expected_unsigned >= 128 {
                expected_unsigned as i64 - 256
            } else {
                expected_unsigned as i64
            };

            assert_eq!(
                outputs[0].1,
                expected_signed.to_string(),
                "Negate({}) expected {} but got {}",
                input,
                expected_signed,
                outputs[0].1
            );
            assert!(
                (phase_exponent - 0.0).abs() < 1e-10,
                "Negate({}) phase_exponent should be 0 but got {}",
                input,
                phase_exponent
            );
        }
    }

    #[test]
    fn test_cswap_no_swap() {
        let outputs = run_cswap(0, 3, 7);
        let ctrl_val = &outputs.iter().find(|o| o.0 == "ctrl").unwrap().1;
        let x_val = &outputs.iter().find(|o| o.0 == "x").unwrap().1;
        let y_val = &outputs.iter().find(|o| o.0 == "y").unwrap().1;
        assert_eq!(ctrl_val, "0");
        assert_eq!(x_val, "3");
        assert_eq!(y_val, "7");
    }

    #[test]
    fn test_cswap_swap() {
        let outputs = run_cswap(1, 3, 7);
        let ctrl_val = &outputs.iter().find(|o| o.0 == "ctrl").unwrap().1;
        let x_val = &outputs.iter().find(|o| o.0 == "x").unwrap().1;
        let y_val = &outputs.iter().find(|o| o.0 == "y").unwrap().1;
        assert_eq!(ctrl_val, "1");
        assert_eq!(x_val, "7");
        assert_eq!(y_val, "3");
    }

    #[test]
    fn test_cswap_various() {
        for (c, a, b) in [
            (0, 0, 0),
            (1, 0, 0),
            (0, 15, 31),
            (1, 15, 31),
            (1, 1, 2),
            (0, 1, 2),
        ] {
            let outputs = run_cswap(c, a, b);
            let x_val: i64 = outputs
                .iter()
                .find(|o| o.0 == "x")
                .unwrap()
                .1
                .parse()
                .unwrap();
            let y_val: i64 = outputs
                .iter()
                .find(|o| o.0 == "y")
                .unwrap()
                .1
                .parse()
                .unwrap();
            if c == 0 {
                assert_eq!(x_val, a, "ctrl=0: x should be {} but got {}", a, x_val);
                assert_eq!(y_val, b, "ctrl=0: y should be {} but got {}", b, y_val);
            } else {
                assert_eq!(x_val, b, "ctrl=1: x should be {} but got {}", b, x_val);
                assert_eq!(y_val, a, "ctrl=1: y should be {} but got {}", a, y_val);
            }
        }
    }

    #[test]
    fn test_int_to_bits() {
        let bits = int_to_bits(5, 8);
        assert_eq!(
            bits,
            vec![false, false, false, false, false, true, false, true]
        );
    }

    #[test]
    fn test_bits_to_uint() {
        let bits = vec![false, false, false, false, false, true, false, true];
        assert_eq!(bits_to_uint(&bits), 5);
    }

    #[test]
    fn test_bits_to_int_positive() {
        let bits = vec![false, false, false, false, false, true, false, true];
        assert_eq!(bits_to_int(&bits), 5);
    }

    #[test]
    fn test_bits_to_int_negative() {
        let bits = int_to_bits(-5i64, 8);
        assert_eq!(bits_to_int(&bits), -5);
    }

    #[test]
    fn test_decimal_str_to_bits_small() {
        let bits = decimal_str_to_bits("5", 8);
        assert_eq!(bits, int_to_bits(5, 8));
    }

    #[test]
    fn test_decimal_str_to_bits_zero() {
        let bits = decimal_str_to_bits("0", 8);
        assert_eq!(bits, vec![false; 8]);
    }

    #[test]
    fn test_decimal_str_to_bits_255() {
        let bits = decimal_str_to_bits("255", 8);
        assert_eq!(bits, vec![true; 8]);
    }

    #[test]
    fn test_bits_to_uint_str_small() {
        let bits = int_to_bits(42, 8);
        assert_eq!(bits_to_uint_str(&bits), "42");
    }

    #[test]
    fn test_bits_to_uint_str_zero() {
        let bits = vec![false; 8];
        assert_eq!(bits_to_uint_str(&bits), "0");
    }

    #[test]
    fn test_bits_to_int_str_negative() {
        let bits = int_to_bits(-5i64, 8);
        assert_eq!(bits_to_int_str(&bits), "-5");
    }

    #[test]
    fn test_bits_to_int_str_positive() {
        let bits = int_to_bits(42, 8);
        assert_eq!(bits_to_int_str(&bits), "42");
    }

    #[test]
    fn test_signed_decimal_str_to_bits_negative() {
        let bits = signed_decimal_str_to_bits("-5", 8);
        let expected = int_to_bits(-5, 8);
        assert_eq!(bits, expected);
    }

    #[test]
    fn test_signed_decimal_str_to_bits_positive() {
        let bits = signed_decimal_str_to_bits("5", 8);
        let expected = int_to_bits(5, 8);
        assert_eq!(bits, expected);
    }

    #[test]
    fn test_roundtrip_uint_str_8bit() {
        for val in 0u64..256 {
            let bits = decimal_str_to_bits(&val.to_string(), 8);
            let back = bits_to_uint_str(&bits);
            assert_eq!(
                back,
                val.to_string(),
                "Unsigned str roundtrip failed for {}",
                val
            );
        }
    }

    #[test]
    fn test_roundtrip_int_str_8bit() {
        for val in -128i64..128 {
            let bits = signed_decimal_str_to_bits(&val.to_string(), 8);
            let back = bits_to_int_str(&bits);
            assert_eq!(
                back,
                val.to_string(),
                "Signed str roundtrip failed for {}",
                val
            );
        }
    }

    fn compile_and_run(
        source: &str,
        entrypoint: &str,
        input_values: &[(String, Vec<bool>)],
    ) -> Result<Vec<(String, String, String)>, String> {
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        let compiled = compiler::compile(&module)?;
        let sim = execute_once(&compiled, entrypoint, input_values)?;
        Ok(sim.extract_outputs())
    }

    #[test]
    fn test_zero_state_allocates_zero() {
        let source = r#"
# Qualtran-L1
# 1.0.0

qdef PrepZero
[
    q: | -> QBit,
] {
    q                    = ZeroState        []
                           return           [q=q]
}

extern qdef ZeroState
from qualtran.bloqs.basic_gates.ZeroState
[q: | -> QBit]
"#;
        let outputs = compile_and_run(source, "PrepZero", &[]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].0, "q");
        assert_eq!(outputs[0].1, "0");
    }

    #[test]
    fn test_one_state_allocates_one() {
        let source = r#"
# Qualtran-L1
# 1.0.0

qdef PrepOne
[
    q: | -> QBit,
] {
    q                    = OneState         []
                           return           [q=q]
}

extern qdef OneState
from qualtran.bloqs.basic_gates.OneState
[q: | -> QBit]
"#;
        let outputs = compile_and_run(source, "PrepOne", &[]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].0, "q");
        assert_eq!(outputs[0].1, "1");
    }

    #[test]
    fn test_zero_effect_accepts_zero() {
        let source = r#"
# Qualtran-L1
# 1.0.0

qdef TestZeroEffect
[
    q: QBit -> |,
] {
    |                    = ZeroEffect       [q=q]
}

extern qdef ZeroEffect
from qualtran.bloqs.basic_gates.ZeroEffect
[q: QBit -> |]
"#;
        let result = compile_and_run(source, "TestZeroEffect", &[("q".to_string(), vec![false])]);
        assert!(result.is_ok(), "ZeroEffect on |0> should succeed");
    }

    #[test]
    fn test_zero_effect_rejects_one() {
        let source = r#"
# Qualtran-L1
# 1.0.0

qdef TestZeroEffect
[
    q: QBit -> |,
] {
    |                    = ZeroEffect       [q=q]
}

extern qdef ZeroEffect
from qualtran.bloqs.basic_gates.ZeroEffect
[q: QBit -> |]
"#;
        let result = compile_and_run(source, "TestZeroEffect", &[("q".to_string(), vec![true])]);
        assert!(result.is_err(), "ZeroEffect on |1> should fail");
        let err = result.unwrap_err();
        assert!(err.contains("ZeroEffect"), "Error: {}", err);
        assert!(err.contains("should be 0"), "Error: {}", err);
    }

    #[test]
    fn test_one_effect_accepts_one() {
        let source = r#"
# Qualtran-L1
# 1.0.0

qdef TestOneEffect
[
    q: QBit -> |,
] {
    |                    = OneEffect        [q=q]
}

extern qdef OneEffect
from qualtran.bloqs.basic_gates.OneEffect
[q: QBit -> |]
"#;
        let result = compile_and_run(source, "TestOneEffect", &[("q".to_string(), vec![true])]);
        assert!(result.is_ok(), "OneEffect on |1> should succeed");
    }

    #[test]
    fn test_one_effect_rejects_zero() {
        let source = r#"
# Qualtran-L1
# 1.0.0

qdef TestOneEffect
[
    q: QBit -> |,
] {
    |                    = OneEffect        [q=q]
}

extern qdef OneEffect
from qualtran.bloqs.basic_gates.OneEffect
[q: QBit -> |]
"#;
        let result = compile_and_run(source, "TestOneEffect", &[("q".to_string(), vec![false])]);
        assert!(result.is_err(), "OneEffect on |0> should fail");
        let err = result.unwrap_err();
        assert!(err.contains("OneEffect"), "Error: {}", err);
        assert!(err.contains("should be 1"), "Error: {}", err);
    }

    #[test]
    fn test_allocate_multi_bit() {
        let source = r#"
# Qualtran-L1
# 1.0.0

qdef TestAlloc
[
    reg: | -> QUInt(4),
] {
    reg                  = Allocate(QUInt(4))[]
                           return           [reg=reg]
}

extern qdef Allocate(QUInt(4))
from qualtran.bloqs.bookkeeping.Allocate
[reg: | -> QUInt(4)]
"#;
        let outputs = compile_and_run(source, "TestAlloc", &[]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].0, "reg");
        assert_eq!(outputs[0].1, "0");
    }

    #[test]
    fn test_free_rejects_nonzero() {
        let source = r#"
# Qualtran-L1
# 1.0.0

qdef TestFree
[
    reg: QUInt(4) -> |,
] {
    |                    = Free(QUInt(4))   [reg=reg]
}

extern qdef Free(QUInt(4))
from qualtran.bloqs.bookkeeping.Free
[reg: QUInt(4) -> |]
"#;
        let result = compile_and_run(
            source,
            "TestFree",
            &[("reg".to_string(), vec![false, true, false, true])],
        );
        assert!(result.is_err(), "Free on non-zero should fail");
        let err = result.unwrap_err();
        assert!(err.contains("Free"), "Error: {}", err);
        assert!(err.contains("should be 0"), "Error: {}", err);
    }

    #[test]
    fn test_free_accepts_zero() {
        let source = r#"
# Qualtran-L1
# 1.0.0

qdef TestFree
[
    reg: QUInt(4) -> |,
] {
    |                    = Free(QUInt(4))   [reg=reg]
}

extern qdef Free(QUInt(4))
from qualtran.bloqs.bookkeeping.Free
[reg: QUInt(4) -> |]
"#;
        let result = compile_and_run(
            source,
            "TestFree",
            &[("reg".to_string(), vec![false, false, false, false])],
        );
        assert!(result.is_ok(), "Free on all-zero should succeed");
    }

    #[test]
    fn test_zero_state_then_zero_effect_roundtrip() {
        let source = r#"
# Qualtran-L1
# 1.0.0

qdef TestRoundtrip
[
] {
    q                    = ZeroState        []
    |                    = ZeroEffect       [q=q]
}

extern qdef ZeroState
from qualtran.bloqs.basic_gates.ZeroState
[q: | -> QBit]

extern qdef ZeroEffect
from qualtran.bloqs.basic_gates.ZeroEffect
[q: QBit -> |]
"#;
        let result = compile_and_run(source, "TestRoundtrip", &[]);
        assert!(result.is_ok(), "ZeroState -> ZeroEffect should succeed");
    }

    #[test]
    fn test_one_state_then_one_effect_roundtrip() {
        let source = r#"
# Qualtran-L1
# 1.0.0

qdef TestRoundtrip
[
] {
    q                    = OneState         []
    |                    = OneEffect        [q=q]
}

extern qdef OneState
from qualtran.bloqs.basic_gates.OneState
[q: | -> QBit]

extern qdef OneEffect
from qualtran.bloqs.basic_gates.OneEffect
[q: QBit -> |]
"#;
        let result = compile_and_run(source, "TestRoundtrip", &[]);
        assert!(result.is_ok(), "OneState -> OneEffect should succeed");
    }

    #[test]
    fn test_one_state_then_zero_effect_fails() {
        let source = r#"
# Qualtran-L1
# 1.0.0

qdef TestMismatch
[
] {
    q                    = OneState         []
    |                    = ZeroEffect       [q=q]
}

extern qdef OneState
from qualtran.bloqs.basic_gates.OneState
[q: | -> QBit]

extern qdef ZeroEffect
from qualtran.bloqs.basic_gates.ZeroEffect
[q: QBit -> |]
"#;
        let result = compile_and_run(source, "TestMismatch", &[]);
        assert!(result.is_err(), "OneState -> ZeroEffect should fail");
    }

    #[test]
    fn test_alias_scoping_execution() {
        // End-to-end test: two subroutines use the same alias name "op"
        // but targeting different subroutines. CallerA's "op" flips the bit,
        // CallerB's "op" is a no-op. Verify the outputs differ.
        let source = r#"
# Qualtran-L1
# 1.0.0

qdef CallerA
[
    q: QBit,
] {
    op                   = FlipOnce
    q                    = op               [q=q]
                           return           [q=q]
}

qdef CallerB
[
    q: QBit,
] {
    op                   = Identity
    q                    = op               [q=q]
                           return           [q=q]
}

qdef FlipOnce
[
    q: QBit,
] {
    q                    = X                [q=q]
                           return           [q=q]
}

qdef Identity
[
    q: QBit,
] {
                           return           [q=q]
}

extern qdef X
from qualtran.bloqs.basic_gates.XGate
[q: QBit]
"#;
        // CallerA should flip: |0⟩ → |1⟩
        let outputs_a =
            compile_and_run(source, "CallerA", &[("q".to_string(), vec![false])]).unwrap();
        assert_eq!(
            outputs_a[0].1, "1",
            "CallerA should flip the bit via FlipOnce"
        );

        // CallerB should not flip: |0⟩ → |0⟩
        let outputs_b =
            compile_and_run(source, "CallerB", &[("q".to_string(), vec![false])]).unwrap();
        assert_eq!(
            outputs_b[0].1, "0",
            "CallerB should preserve the bit via Identity"
        );
    }
}
