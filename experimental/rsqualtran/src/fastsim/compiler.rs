//! Compiler: translates an [`L1Module`] AST into a [`CompiledModule`] for the VM.
//!
//! The output is a dense `Vec<CompiledSubroutine>` indexed by subroutine ID (`sub_id`).
//! Each subroutine is one of: [`SubroutineBody::Impl`] (instruction sequence),
//! [`SubroutineBody::Extern`] (a leaf gate), or [`SubroutineBody::Cast`] (type
//! reinterpretation).
//!
//! ## String Interning
//!
//! All string identifiers (bloq keys, variable names, register names) are interned
//! into a shared [`InternTable`] at compile time. The compiled IR uses [`InternId`]
//! everywhere, eliminating string hashing, comparison, and cloning on the VM
//! hot path.
//!
//! ## Slot Assignment
//!
//! Each subroutine has a [`SlotInfo`] that maps its interned variable names to dense
//! local slot indices (0, 1, 2, …). The VM uses a flat `Vec` per call frame,
//! indexed by slot index.
//!
//! ## Subroutine Dispatch
//!
//! [`Instruction::Call::callee`] stores the callee's `sub_id` directly. Dispatch is
//! a single bounds-checked array index with no hashing. Aliases are resolved at
//! compile time.

use std::collections::HashMap;

#[cfg(feature = "py")]
use pyo3::prelude::*;

use crate::nodes::*;

/// An interned string identifier (index into [`InternTable`]).
pub type InternId = u32;
/// A subroutine index (index into [`CompiledModule::subroutines`]).
pub type SubId = usize;
/// A slot index within a subroutine's call frame.
pub type SlotIdx = usize;

/// Bidirectional map between strings and dense [`InternId`]s.
///
/// Populated during compilation. All compiled IR references use [`InternId`]s
/// instead of heap-allocated strings.
#[derive(Debug)]
pub struct InternTable {
    strings: Vec<String>,
    lookup: HashMap<String, InternId>,
}

impl InternTable {
    fn new() -> Self {
        InternTable {
            strings: Vec::new(),
            lookup: HashMap::new(),
        }
    }

    /// Returns the intern ID for `s`, inserting it if not already present.
    fn intern(&mut self, s: &str) -> InternId {
        if let Some(&id) = self.lookup.get(s) {
            return id;
        }
        let id = self.strings.len() as InternId;
        self.strings.push(s.to_string());
        self.lookup.insert(s.to_string(), id);
        id
    }

    /// Resolves an intern ID back to its string. Panics if `id` is out of range.
    pub fn resolve(&self, id: InternId) -> &str {
        &self.strings[id as usize]
    }

    /// Returns the intern ID for `s`, or `None` if it was never interned.
    pub fn get_id(&self, s: &str) -> Option<InternId> {
        self.lookup.get(s).copied()
    }
}

/// A register in a subroutine's signature.
#[derive(Debug, Clone)]
pub struct RegisterInfo {
    /// Interned register name.
    pub name: InternId,
    /// Total number of bits (including shape multiplier, e.g. `QBit[8]` → 8).
    pub n_bits: usize,
    /// Whether this register is input, output, passthrough, or a type cast.
    pub direction: RegisterDirection,
    /// Interned dtype name.
    pub dtype_name: InternId,
    /// Shape dimensions (e.g. [2,2,2] for QBit[2,2,2]). None for scalars.
    pub shape: Option<Vec<usize>>,
}

/// The direction of a register in a signature.
#[derive(Debug, Clone, PartialEq)]
pub enum RegisterDirection {
    /// Thru: present on both left and right sides with the same type.
    Thru,
    /// Output-only: right side only (e.g. `Allocate`).
    RightOnly,
    /// Input-only: left side only (e.g. `Free`).
    LeftOnly,
    /// Type cast: both sides present with different types but equal bit count.
    Cast,
}

/// Maps a callee register to bits from the caller's variables.
#[derive(Debug, Clone)]
pub struct ArgMapping {
    /// Slot index in the callee's call frame (or the current frame for `Return`).
    pub callee_slot: SlotIdx,
    /// Caller slot indices supplying individual bits to the callee register.
    pub bits: Vec<SlotIdx>,
    /// Element sub-slots of the callee register, if it is a shaped register.
    /// Used by the VM to distribute incoming bits to individual element slots.
    pub callee_elements: Option<Vec<SlotIdx>>,
}

/// A callee-to-caller slot pair used in [`OutputMapping::Direct`].
#[derive(Debug, Clone)]
pub struct SlotPair {
    pub callee: SlotIdx,
    pub caller: SlotIdx,
    /// Element sub-slots of the caller register (if shaped). The VM distributes
    /// the multi-bit result to these individual element slots.
    pub caller_elements: Option<Vec<SlotIdx>>,
}

/// Maps a callee's output registers back to the caller's lvalue slots.
#[derive(Debug, Clone)]
pub enum OutputMapping {
    /// Each output register maps 1:1 to a caller slot via [`SlotPair`].
    Direct(Vec<SlotPair>),
    /// Multiple output registers concatenate into a single caller slot.
    Concat {
        callee_slots: Vec<SlotIdx>,
        caller_slot: SlotIdx,
        /// If the caller slot has element children, listed here for distribution.
        caller_elements: Option<Vec<SlotIdx>>,
    },
}

/// A compiled VM instruction.
///
/// `Call` and `Return` handle general subroutine dispatch with argument/output
/// mapping. The `Inline*` variants are inlined extern gates that operate
/// directly on the caller's slots, bypassing call frame allocation.
#[derive(Debug, Clone)]
pub enum Instruction {
    Call {
        /// Index into `CompiledModule.subroutines` (the callee's `sub_id`).
        callee: SubId,
        arg_mappings: Vec<ArgMapping>,
        output_mapping: OutputMapping,
    },
    Return {
        ret_mappings: Vec<ArgMapping>,
    },

    InlineX {
        q: SlotIdx,
        lvalue_slot: SlotIdx,
    },
    InlineCNOT {
        ctrl: SlotIdx,
        target: SlotIdx,
        ctrl_lvalue: SlotIdx,
        target_lvalue: SlotIdx,
    },
    InlineToffoli {
        ctrl0: SlotIdx,
        ctrl1: SlotIdx,
        target: SlotIdx,
        ctrl_lvalue: SlotIdx,
        ctrl0_lvalue: Option<SlotIdx>,
        ctrl1_lvalue: Option<SlotIdx>,
        target_lvalue: SlotIdx,
    },
    InlineAnd {
        ctrl0: SlotIdx,
        ctrl1: SlotIdx,
        target_lvalue: SlotIdx,
        ctrl_lvalue: SlotIdx,
        ctrl0_lvalue: Option<SlotIdx>,
        ctrl1_lvalue: Option<SlotIdx>,
        cv: [bool; 2],
    },
    InlineAndDag {
        ctrl0: SlotIdx,
        ctrl1: SlotIdx,
        target: SlotIdx,
        ctrl_lvalue: SlotIdx,
        ctrl0_lvalue: Option<SlotIdx>,
        ctrl1_lvalue: Option<SlotIdx>,
        cv: [bool; 2],
    },
    InlineTwoBitCSwap {
        ctrl: SlotIdx,
        x: SlotIdx,
        y: SlotIdx,
        ctrl_lvalue: SlotIdx,
        x_lvalue: SlotIdx,
        y_lvalue: SlotIdx,
    },
    InlineZ {
        q: SlotIdx,
        lvalue_slot: SlotIdx,
    },
    InlineS {
        q: SlotIdx,
        lvalue_slot: SlotIdx,
    },
    InlineT {
        q: SlotIdx,
        lvalue_slot: SlotIdx,
    },
    InlineCZ {
        q0: SlotIdx,
        q1: SlotIdx,
        lvalue_slot: SlotIdx,
        q0_lvalue: Option<SlotIdx>,
        q1_lvalue: Option<SlotIdx>,
    },
    InlineCCZ {
        q0: SlotIdx,
        q1: SlotIdx,
        q2: SlotIdx,
        lvalue_slot: SlotIdx,
        q0_lvalue: Option<SlotIdx>,
        q1_lvalue: Option<SlotIdx>,
        q2_lvalue: Option<SlotIdx>,
    },
    InlineAllocate {
        n_bits: usize,
        lvalue_slot: SlotIdx,
        /// Per-element lvalue slots for distributing individual bits.
        element_lvalue_slots: Vec<SlotIdx>,
    },
    InlineZeroState {
        lvalue_slot: SlotIdx,
    },
    InlineOneState {
        lvalue_slot: SlotIdx,
    },
    InlineFree {
        n_bits: usize,
        in_bits: Vec<SlotIdx>,
    },
    InlineZeroEffect {
        q: SlotIdx,
    },
    InlineOneEffect {
        q: SlotIdx,
    },
}

/// A recognized extern gate with slot indices pre-resolved into its own call frame.
///
/// Used for non-inlined calls that go through the normal `Call` dispatch path.
#[derive(Debug, Clone)]
pub enum ExternGate {
    XGate {
        q_slot: SlotIdx,
    },
    CNOT {
        ctrl_slot: SlotIdx,
        target_slot: SlotIdx,
    },
    Toffoli {
        ctrl_slot: SlotIdx,
        target_slot: SlotIdx,
    },
    And {
        ctrl_slot: SlotIdx,
        target_slot: SlotIdx,
        cv: [bool; 2],
    },
    AndDag {
        ctrl_slot: SlotIdx,
        target_slot: SlotIdx,
        cv: [bool; 2],
    },
    TwoBitCSwap {
        ctrl_slot: SlotIdx,
        x_slot: SlotIdx,
        y_slot: SlotIdx,
    },
    ZGate {
        q_slot: SlotIdx,
    },
    SGate {
        q_slot: SlotIdx,
    },
    TGate {
        q_slot: SlotIdx,
    },
    CZGate {
        q_slot: SlotIdx,
    },
    CCZGate {
        q_slot: SlotIdx,
    },
    // State preparation gates
    Allocate {
        n_bits: usize,
        out_slot: SlotIdx,
    },
    ZeroState {
        q_slot: SlotIdx,
    },
    OneState {
        q_slot: SlotIdx,
    },
    // Measurement / deallocation gates
    Free {
        n_bits: usize,
        in_slot: SlotIdx,
    },
    ZeroEffect {
        q_slot: SlotIdx,
    },
    OneEffect {
        q_slot: SlotIdx,
    },
}

/// A cast operation with pre-resolved slot indices.
#[derive(Debug, Clone)]
pub enum CastOp {
    Split {
        total_bits: usize,
        reg_slot: SlotIdx,
    },
    Join {
        total_bits: usize,
        reg_slot: SlotIdx,
    },
    /// A multi-register partition (bit-preserving rewiring across registers).
    ///
    /// Models `_PartitionBase` bloqs (`Partition`/`Split2`/`Join2` and their
    /// adjoints), which the L1 layer emits as multi-register `qcast` nodes.
    ///
    /// The invariant these bloqs uphold is that the ordered concatenation of
    /// the LEFT (input) registers' bits equals the ordered concatenation of the
    /// RIGHT (output) registers' bits. Execution therefore gathers all input
    /// bits (in signature order) and re-distributes them to the output
    /// registers (in signature order), regardless of which side is "lumped".
    Partition {
        /// Ordered `(slot, n_bits)` for each LEFT/input register.
        input_slots: Vec<(SlotIdx, usize)>,
        /// Ordered `(slot, n_bits)` for each RIGHT/output register.
        output_slots: Vec<(SlotIdx, usize)>,
        /// Total bits; equals the summed input bits and the summed output bits.
        total_bits: usize,
    },
}

/// Maps interned variable names to dense local slot indices for a single subroutine.
///
/// The VM allocates a flat `Vec` of size `n_slots` per call frame.
#[derive(Debug, Clone)]
pub struct SlotInfo {
    /// Intern ID → local slot index for variables used in this subroutine.
    pub intern_to_slot: HashMap<InternId, SlotIdx>,
    /// Total number of slots in this call frame.
    pub n_slots: usize,
    /// Parent slot → ordered element sub-slots (row-major). The VM uses this
    /// to distribute a multi-bit value into individual element slots.
    pub element_slots: HashMap<SlotIdx, Vec<SlotIdx>>,
}

impl SlotInfo {
    fn new() -> Self {
        SlotInfo {
            intern_to_slot: HashMap::new(),
            n_slots: 0,
            element_slots: HashMap::new(),
        }
    }

    /// Returns the slot index for `intern_id`, assigning a new one if absent.
    fn get_or_assign(&mut self, intern_id: InternId) -> SlotIdx {
        if let Some(&slot) = self.intern_to_slot.get(&intern_id) {
            slot
        } else {
            let slot = self.n_slots;
            self.intern_to_slot.insert(intern_id, slot);
            self.n_slots += 1;
            slot
        }
    }

    /// Register element slots for a parent slot.
    fn register_element_slot(
        &mut self,
        parent_slot: SlotIdx,
        element_slot: SlotIdx,
        offset: usize,
    ) {
        let elements = self
            .element_slots
            .entry(parent_slot)
            .or_insert_with(Vec::new);
        // Ensure the vector is large enough
        if elements.len() <= offset {
            elements.resize(offset + 1, usize::MAX);
        }
        elements[offset] = element_slot;
    }
}

/// The body of a compiled subroutine: instructions, an extern gate, or a cast.
#[derive(Debug, Clone)]
pub enum SubroutineBody {
    Impl(Vec<Instruction>),
    Extern(ExternGate),
    Cast(CastOp),
}

/// A compiled subroutine with its signature, body, and slot assignments.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "py", pyo3::pyclass(from_py_object))]
pub struct CompiledSubroutine {
    /// Interned bloq_key ID.
    pub bloq_key: InternId,
    pub signature: Vec<RegisterInfo>,
    pub body: SubroutineBody,
    /// Slot assignments for this subroutine's variables.
    pub slots: SlotInfo,
}

/// The compiled module: a dense array of subroutines with compile-time-resolved dispatch.
#[derive(Debug)]
#[cfg_attr(feature = "py", pyo3::pyclass)]
pub struct CompiledModule {
    /// Subroutines indexed by `sub_id`.
    pub subroutines: Vec<CompiledSubroutine>,
    /// Bloq key intern ID → `sub_id`. Used for entrypoint lookup; not on the hot path.
    pub intern_to_sub: HashMap<InternId, SubId>,
    /// Shared intern table for resolving IDs back to strings.
    pub intern_table: InternTable,
}

impl CompiledModule {
    /// Returns the subroutine for the given bloq key, or `None` if not found.
    pub fn get_subroutine(&self, key: &str) -> Option<&CompiledSubroutine> {
        let id = self.intern_table.get_id(key)?;
        let sub_idx = self.intern_to_sub.get(&id)?;
        Some(&self.subroutines[*sub_idx])
    }

    /// Returns `true` if a subroutine exists for the given bloq key.
    pub fn has_subroutine(&self, key: &str) -> bool {
        self.get_subroutine(key).is_some()
    }
}

#[cfg(feature = "py")]
#[pymethods]
impl CompiledModule {
    #[pyo3(name = "get_subroutine")]
    pub fn py_get_subroutine(&self, key: &str) -> Option<CompiledSubroutine> {
        self.get_subroutine(key).cloned()
    }

    #[pyo3(name = "has_subroutine")]
    pub fn py_has_subroutine(&self, key: &str) -> bool {
        self.has_subroutine(key)
    }

    /// Return the signature of a subroutine as a list of register descriptors.
    ///
    /// Each element is a 5-tuple `(name, n_bits, direction, dtype, shape)`:
    ///
    /// - `name: String` — Register name (e.g. `"cube"`, `"aux"`).
    /// - `n_bits: usize` — Total bit count, including shape (e.g. `QBit[2,2,2]` → 8).
    /// - `direction: String` — One of `"Thru"`, `"RightOnly"`, `"LeftOnly"`, or `"Cast"`.
    /// - `dtype: String` — Base data-type name (e.g. `"QBit"`, `"QUInt"`, `"QInt"`).
    /// - `shape: Option<Vec<usize>>` — Shape dimensions for array registers
    ///   (e.g. `[2,2,2]`), or `None` for scalars.
    #[pyo3(name = "get_subroutine_signature")]
    pub fn py_get_subroutine_signature(
        &self,
        key: &str,
    ) -> PyResult<Vec<(String, usize, String, String, Option<Vec<usize>>)>> {
        let sub = self.get_subroutine(key).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Subroutine '{}' not found", key))
        })?;
        let mut sig = Vec::new();
        for reg in &sub.signature {
            let name = self.intern_table.resolve(reg.name).to_string();
            let dtype = self.intern_table.resolve(reg.dtype_name).to_string();
            let dir = match reg.direction {
                RegisterDirection::Thru => "Thru",
                RegisterDirection::RightOnly => "RightOnly",
                RegisterDirection::LeftOnly => "LeftOnly",
                RegisterDirection::Cast => "Cast",
            }
            .to_string();
            sig.push((name, reg.n_bits, dir, dtype, reg.shape.clone()));
        }
        Ok(sig)
    }
}

/// Phase 1 output for a subroutine: signature and slot assignments.
/// Consumed in Phase 2 to resolve callee references and build argument mappings.
#[derive(Debug, Clone)]
struct Phase1Info {
    bloq_key: InternId,
    signature: Vec<RegisterInfo>,
    slots: SlotInfo,
}

/// Compiles an [`L1Module`] AST into a [`CompiledModule`].
///
/// Two-phase compilation:
/// 1. Assign `sub_id`s, compile signatures, and initialize slot assignments.
/// 2. Compile each qdef body with callee references and argument mappings pre-resolved.
pub fn compile(module: &L1Module) -> Result<CompiledModule, String> {
    let mut intern = InternTable::new();
    let mut intern_to_sub: HashMap<InternId, SubId> = HashMap::new();

    // Phase 1: assign sub_ids, compile signatures, initialize slot assignments.
    let mut phase1_infos = Vec::with_capacity(module.qdefs.len());
    for (idx, qdef) in module.qdefs.iter().enumerate() {
        let (bloq_key_str, qsig) = match qdef {
            QDefNode::Impl(n) => (&n.bloq_key, &n.qsignature),
            QDefNode::Extern(n) => (&n.bloq_key, &n.qsignature),
            QDefNode::Cast(n) => (&n.bloq_key, &n.qsignature),
        };
        let key_id = intern.intern(bloq_key_str);
        intern_to_sub.insert(key_id, idx);

        let signature = compile_signature(qsig, &mut intern)?;
        let mut slots = SlotInfo::new();
        for reg in &signature {
            let parent_slot = slots.get_or_assign(reg.name);
            // For shaped registers, also assign element slots
            if let Some(ref shape) = reg.shape {
                let reg_name = intern.resolve(reg.name).to_string();
                let total_elements: usize = shape.iter().product();
                for flat_idx in 0..total_elements {
                    let element_name = format_element_name(&reg_name, shape, flat_idx);
                    let elem_intern = intern.intern(&element_name);
                    let elem_slot = slots.get_or_assign(elem_intern);
                    slots.register_element_slot(parent_slot, elem_slot, flat_idx);
                }
            }
        }
        phase1_infos.push(Phase1Info {
            bloq_key: key_id,
            signature,
            slots,
        });
    }

    // Phase 2: compile each qdef body with resolved callee references and mappings.
    let mut subroutines = Vec::with_capacity(module.qdefs.len());
    for (idx, qdef) in module.qdefs.iter().enumerate() {
        let phase1 = &phase1_infos[idx];
        let compiled = compile_qdef(
            qdef,
            phase1.bloq_key,
            phase1.signature.clone(),
            phase1.slots.clone(),
            &mut intern,
            &intern_to_sub,
            &phase1_infos,
            module,
        )?;
        subroutines.push(compiled);
    }

    Ok(CompiledModule {
        subroutines,
        intern_to_sub,
        intern_table: intern,
    })
}

fn compile_qdef(
    qdef: &QDefNode,
    bloq_key: InternId,
    signature: Vec<RegisterInfo>,
    slots: SlotInfo,
    intern: &mut InternTable,
    intern_to_sub: &HashMap<InternId, SubId>,
    phase1_infos: &[Phase1Info],
    module: &L1Module,
) -> Result<CompiledSubroutine, String> {
    match qdef {
        QDefNode::Impl(impl_node) => compile_impl(
            impl_node,
            bloq_key,
            signature,
            slots,
            intern,
            intern_to_sub,
            phase1_infos,
            module,
        ),
        QDefNode::Extern(extern_node) => {
            compile_extern(extern_node, bloq_key, signature, slots, intern)
        }
        QDefNode::Cast(cast_node) => compile_cast(cast_node, bloq_key, signature, slots, intern),
    }
}

fn compile_impl(
    node: &QDefImplNode,
    bloq_key: InternId,
    signature: Vec<RegisterInfo>,
    mut slots: SlotInfo,
    intern: &mut InternTable,
    intern_to_sub: &HashMap<InternId, SubId>,
    phase1_infos: &[Phase1Info],
    module: &L1Module,
) -> Result<CompiledSubroutine, String> {
    let mut instructions = Vec::new();

    // Build a shape map from signature registers for ND index resolution.
    let shape_map: HashMap<InternId, Vec<usize>> = signature
        .iter()
        .filter_map(|reg| reg.shape.as_ref().map(|s| (reg.name, s.clone())))
        .collect();

    // Build a per-subroutine alias map overlaying the global bloq key map.
    // Aliases are scoped to this subroutine to avoid cross-subroutine collisions.
    let mut local_intern_to_sub = intern_to_sub.clone();
    for stmt in &node.body {
        if let StatementNode::Alias(alias_node) = stmt {
            let alias_id = intern.intern(&alias_node.alias);
            let target_id = intern.intern(&alias_node.bloq_key);
            let sub_idx = *local_intern_to_sub.get(&target_id).ok_or_else(|| {
                format!(
                    "In '{}': alias '{}' references unknown target '{}'",
                    node.bloq_key, alias_node.alias, alias_node.bloq_key
                )
            })?;
            local_intern_to_sub.insert(alias_id, sub_idx);
        }
    }

    // Pre-scan: walk every qarg reference and lvalue in the body to ensure
    // all element slots are registered in `slots` BEFORE the compilation loop
    // that builds output mappings and inline instructions. Element slots are
    // registered lazily by `resolve_nested_qarg` when indexed references like
    // `reg[0]` are encountered, but output mapping construction needs the
    // element_slots HashMap to be already populated for the lvalue being mapped.
    for stmt in &node.body {
        match stmt {
            StatementNode::Call(call) => {
                for qarg in &call.qargs {
                    resolve_nested_qarg(&qarg.value, intern, &mut slots, &shape_map);
                }
                for lv in &call.lvalues {
                    let id = intern.intern(lv);
                    slots.get_or_assign(id);
                }
            }
            StatementNode::Return(ret) => {
                for qarg in &ret.ret_mapping {
                    resolve_nested_qarg(&qarg.value, intern, &mut slots, &shape_map);
                }
            }
            StatementNode::Alias(_) => {} // already handled
        }
    }

    for stmt in &node.body {
        match stmt {
            StatementNode::Call(call) => {
                let callee_intern_id = intern.intern(&call.bloq_key);
                let callee_sub_id =
                    *local_intern_to_sub.get(&callee_intern_id).ok_or_else(|| {
                        format!(
                            "In '{}': call target '{}' not found in compiled module",
                            node.bloq_key, call.bloq_key
                        )
                    })?;
                let callee_info = &phase1_infos[callee_sub_id];
                let arg_mappings = compile_call_qargs(
                    &call.qargs,
                    intern,
                    &mut slots,
                    &callee_info.slots,
                    &call.bloq_key,
                    &node.bloq_key,
                    &shape_map,
                )?;
                let lvalues: Vec<InternId> = call
                    .lvalues
                    .iter()
                    .map(|lv| {
                        let id = intern.intern(lv);
                        slots.get_or_assign(id);
                        id
                    })
                    .collect();

                let output_regs: Vec<&RegisterInfo> = callee_info
                    .signature
                    .iter()
                    .filter(|r| {
                        r.direction == RegisterDirection::Thru
                            || r.direction == RegisterDirection::RightOnly
                            || r.direction == RegisterDirection::Cast
                    })
                    .collect();

                let output_mapping = if !lvalues.is_empty() {
                    if lvalues.len() == output_regs.len() {
                        let mut pairs = Vec::with_capacity(lvalues.len());
                        for (lvalue_id, reg) in lvalues.iter().zip(output_regs.iter()) {
                            let callee_slot = *callee_info.slots.intern_to_slot.get(&reg.name)
                                .ok_or_else(|| format!(
                                    "In '{}': call to '{}': output register '{}' has no slot in callee",
                                    node.bloq_key, call.bloq_key, intern.resolve(reg.name)
                                ))?;
                            let caller_slot =
                                *slots.intern_to_slot.get(lvalue_id).ok_or_else(|| {
                                    format!(
                                        "In '{}': call to '{}': lvalue '{}' has no slot in caller",
                                        node.bloq_key,
                                        call.bloq_key,
                                        intern.resolve(*lvalue_id)
                                    )
                                })?;
                            pairs.push(SlotPair {
                                callee: callee_slot,
                                caller: caller_slot,
                                caller_elements: slots.element_slots.get(&caller_slot).cloned(),
                            });
                        }
                        OutputMapping::Direct(pairs)
                    } else if lvalues.len() == 1 && output_regs.len() > 1 {
                        let mut callee_slots = Vec::with_capacity(output_regs.len());
                        for reg in &output_regs {
                            let callee_slot = *callee_info.slots.intern_to_slot.get(&reg.name)
                                .ok_or_else(|| format!(
                                    "In '{}': call to '{}': output register '{}' has no slot in callee (concat case)",
                                    node.bloq_key, call.bloq_key, intern.resolve(reg.name)
                                ))?;
                            callee_slots.push(callee_slot);
                        }
                        let caller_slot = *slots.intern_to_slot.get(&lvalues[0])
                            .ok_or_else(|| format!(
                                "In '{}': call to '{}': lvalue '{}' has no slot in caller (concat case)",
                                node.bloq_key, call.bloq_key, intern.resolve(lvalues[0])
                            ))?;
                        OutputMapping::Concat {
                            callee_slots,
                            caller_slot,
                            caller_elements: slots.element_slots.get(&caller_slot).cloned(),
                        }
                    } else {
                        return Err(format!(
                            "In '{}': call to '{}': lvalue count ({}) does not match output register count ({}). \
                             Lvalues: {:?}, Output regs: {:?}",
                            node.bloq_key,
                            call.bloq_key,
                            lvalues.len(),
                            output_regs.len(),
                            lvalues.iter().map(|id| intern.resolve(*id)).collect::<Vec<_>>(),
                            output_regs.iter().map(|r| intern.resolve(r.name)).collect::<Vec<_>>()
                        ));
                    }
                } else {
                    OutputMapping::Direct(Vec::new())
                };

                let mut inlined = None;
                if let QDefNode::Extern(extern_node) = &module.qdefs[callee_sub_id] {
                    inlined = try_inline_extern(
                        extern_node,
                        callee_info,
                        &arg_mappings,
                        &output_mapping,
                        intern,
                        &slots,
                    );
                }

                if let Some(instr) = inlined {
                    instructions.push(instr);
                } else {
                    instructions.push(Instruction::Call {
                        callee: callee_sub_id,
                        arg_mappings,
                        output_mapping,
                    });
                }
            }
            StatementNode::Return(ret) => {
                let ret_mappings = compile_return_qargs(
                    &ret.ret_mapping,
                    intern,
                    &mut slots,
                    &node.bloq_key,
                    &shape_map,
                )?;
                instructions.push(Instruction::Return { ret_mappings });
            }
            StatementNode::Alias(_) => {
                // Handled in the alias collection pass above.
            }
        }
    }

    Ok(CompiledSubroutine {
        bloq_key,
        signature,
        body: SubroutineBody::Impl(instructions),
        slots,
    })
}

/// Attempts to inline an extern gate call. Returns `Some(Instruction::Inline*)`
/// if the gate is recognized and all arguments are simple enough to inline,
/// or `None` to fall back to a normal `Call` instruction.
fn try_inline_extern(
    extern_node: &QDefExternNode,
    callee_info: &Phase1Info,
    arg_mappings: &[ArgMapping],
    output_mapping: &OutputMapping,
    intern: &InternTable,
    caller_slots: &SlotInfo,
) -> Option<Instruction> {
    let gate = match_extern_gate(
        &extern_node.cobject_from,
        &callee_info.signature,
        &callee_info.slots,
        intern,
    )
    .ok()?;

    let get_arg = |slot: SlotIdx| -> Option<&ArgMapping> {
        arg_mappings.iter().find(|m| m.callee_slot == slot)
    };

    let get_single_bit = |slot: SlotIdx| -> Option<SlotIdx> {
        let arg = get_arg(slot)?;
        if arg.bits.len() == 1 {
            Some(arg.bits[0])
        } else {
            None
        }
    };

    let get_lvalue = |slot: SlotIdx| -> Option<SlotIdx> {
        match output_mapping {
            OutputMapping::Direct(pairs) => {
                pairs.iter().find(|p| p.callee == slot).map(|p| p.caller)
            }
            OutputMapping::Concat { .. } => None,
        }
    };

    // Returns the element sub-slots of the caller's lvalue slot, if any.
    fn get_lvalue_elements(
        callee_slot: SlotIdx,
        output_mapping: &OutputMapping,
        caller_slots: &SlotInfo,
    ) -> Option<Vec<SlotIdx>> {
        let parent = match output_mapping {
            OutputMapping::Direct(pairs) => pairs
                .iter()
                .find(|p| p.callee == callee_slot)
                .map(|p| p.caller),
            OutputMapping::Concat { .. } => None,
        }?;
        caller_slots.element_slots.get(&parent).cloned()
    }

    match gate {
        ExternGate::XGate { q_slot } => {
            let q = get_single_bit(q_slot)?;
            let lvalue_slot = get_lvalue(q_slot)?;
            Some(Instruction::InlineX { q, lvalue_slot })
        }
        ExternGate::CNOT {
            ctrl_slot,
            target_slot,
        } => {
            let ctrl = get_single_bit(ctrl_slot)?;
            let target = get_single_bit(target_slot)?;
            let ctrl_lvalue = get_lvalue(ctrl_slot)?;
            let target_lvalue = get_lvalue(target_slot)?;
            Some(Instruction::InlineCNOT {
                ctrl,
                target,
                ctrl_lvalue,
                target_lvalue,
            })
        }
        ExternGate::Toffoli {
            ctrl_slot,
            target_slot,
        } => {
            let ctrl_bits = &get_arg(ctrl_slot)?.bits;
            if ctrl_bits.len() != 2 {
                return None;
            }
            let ctrl0 = ctrl_bits[0];
            let ctrl1 = ctrl_bits[1];
            let target = get_single_bit(target_slot)?;
            let ctrl_lvalue = get_lvalue(ctrl_slot)?;
            let target_lvalue = get_lvalue(target_slot)?;
            // Get per-element lvalue slots for ctrl (optional)
            let ctrl_elem_lvalues = get_lvalue_elements(ctrl_slot, output_mapping, caller_slots);
            let (ctrl0_lv, ctrl1_lv) = match ctrl_elem_lvalues {
                Some(ref elems) if elems.len() == 2 => (Some(elems[0]), Some(elems[1])),
                _ => (None, None),
            };
            Some(Instruction::InlineToffoli {
                ctrl0,
                ctrl1,
                target,
                ctrl_lvalue,
                ctrl0_lvalue: ctrl0_lv,
                ctrl1_lvalue: ctrl1_lv,
                target_lvalue,
            })
        }
        ExternGate::And {
            ctrl_slot,
            target_slot,
            cv,
        } => {
            let ctrl_bits = &get_arg(ctrl_slot)?.bits;
            if ctrl_bits.len() != 2 {
                return None;
            }
            let ctrl0 = ctrl_bits[0];
            let ctrl1 = ctrl_bits[1];
            let ctrl_lvalue = get_lvalue(ctrl_slot)?;
            let target_lvalue = get_lvalue(target_slot)?;
            let ctrl_elem_lvalues = get_lvalue_elements(ctrl_slot, output_mapping, caller_slots);
            let (ctrl0_lv, ctrl1_lv) = match ctrl_elem_lvalues {
                Some(ref elems) if elems.len() == 2 => (Some(elems[0]), Some(elems[1])),
                _ => (None, None),
            };
            Some(Instruction::InlineAnd {
                ctrl0,
                ctrl1,
                target_lvalue,
                ctrl_lvalue,
                ctrl0_lvalue: ctrl0_lv,
                ctrl1_lvalue: ctrl1_lv,
                cv,
            })
        }
        ExternGate::AndDag {
            ctrl_slot,
            target_slot,
            cv,
        } => {
            let ctrl_bits = &get_arg(ctrl_slot)?.bits;
            if ctrl_bits.len() != 2 {
                return None;
            }
            let ctrl0 = ctrl_bits[0];
            let ctrl1 = ctrl_bits[1];
            let target = get_single_bit(target_slot)?;
            let ctrl_lvalue = get_lvalue(ctrl_slot)?;
            let ctrl_elem_lvalues = get_lvalue_elements(ctrl_slot, output_mapping, caller_slots);
            let (ctrl0_lv, ctrl1_lv) = match ctrl_elem_lvalues {
                Some(ref elems) if elems.len() == 2 => (Some(elems[0]), Some(elems[1])),
                _ => (None, None),
            };
            Some(Instruction::InlineAndDag {
                ctrl0,
                ctrl1,
                target,
                ctrl_lvalue,
                ctrl0_lvalue: ctrl0_lv,
                ctrl1_lvalue: ctrl1_lv,
                cv,
            })
        }
        ExternGate::TwoBitCSwap {
            ctrl_slot,
            x_slot,
            y_slot,
        } => {
            let ctrl = get_single_bit(ctrl_slot)?;
            let x = get_single_bit(x_slot)?;
            let y = get_single_bit(y_slot)?;
            let ctrl_lvalue = get_lvalue(ctrl_slot)?;
            let x_lvalue = get_lvalue(x_slot)?;
            let y_lvalue = get_lvalue(y_slot)?;
            Some(Instruction::InlineTwoBitCSwap {
                ctrl,
                x,
                y,
                ctrl_lvalue,
                x_lvalue,
                y_lvalue,
            })
        }
        ExternGate::ZGate { q_slot } => {
            let q = get_single_bit(q_slot)?;
            let lvalue_slot = get_lvalue(q_slot)?;
            Some(Instruction::InlineZ { q, lvalue_slot })
        }
        ExternGate::SGate { q_slot } => {
            let q = get_single_bit(q_slot)?;
            let lvalue_slot = get_lvalue(q_slot)?;
            Some(Instruction::InlineS { q, lvalue_slot })
        }
        ExternGate::TGate { q_slot } => {
            let q = get_single_bit(q_slot)?;
            let lvalue_slot = get_lvalue(q_slot)?;
            Some(Instruction::InlineT { q, lvalue_slot })
        }
        ExternGate::CZGate { q_slot } => {
            let q_bits = &get_arg(q_slot)?.bits;
            if q_bits.len() != 2 {
                return None;
            }
            let q0 = q_bits[0];
            let q1 = q_bits[1];
            let lvalue_slot = get_lvalue(q_slot)?;
            let q_elem_lvalues = get_lvalue_elements(q_slot, output_mapping, caller_slots);
            let (q0_lv, q1_lv) = match q_elem_lvalues {
                Some(ref elems) if elems.len() == 2 => (Some(elems[0]), Some(elems[1])),
                _ => (None, None),
            };
            Some(Instruction::InlineCZ {
                q0,
                q1,
                lvalue_slot,
                q0_lvalue: q0_lv,
                q1_lvalue: q1_lv,
            })
        }
        ExternGate::CCZGate { q_slot } => {
            let q_bits = &get_arg(q_slot)?.bits;
            if q_bits.len() != 3 {
                return None;
            }
            let q0 = q_bits[0];
            let q1 = q_bits[1];
            let q2 = q_bits[2];
            let lvalue_slot = get_lvalue(q_slot)?;
            let q_elem_lvalues = get_lvalue_elements(q_slot, output_mapping, caller_slots);
            let (q0_lv, q1_lv, q2_lv) = match q_elem_lvalues {
                Some(ref elems) if elems.len() == 3 => {
                    (Some(elems[0]), Some(elems[1]), Some(elems[2]))
                }
                _ => (None, None, None),
            };
            Some(Instruction::InlineCCZ {
                q0,
                q1,
                q2,
                lvalue_slot,
                q0_lvalue: q0_lv,
                q1_lvalue: q1_lv,
                q2_lvalue: q2_lv,
            })
        }
        ExternGate::Allocate { n_bits, out_slot } => {
            let lvalue_slot = get_lvalue(out_slot)?;
            let element_lvalue_slots =
                get_lvalue_elements(out_slot, output_mapping, caller_slots).unwrap_or_default();
            Some(Instruction::InlineAllocate {
                n_bits,
                lvalue_slot,
                element_lvalue_slots,
            })
        }
        ExternGate::ZeroState { q_slot } => {
            let lvalue_slot = get_lvalue(q_slot)?;
            Some(Instruction::InlineZeroState { lvalue_slot })
        }
        ExternGate::OneState { q_slot } => {
            let lvalue_slot = get_lvalue(q_slot)?;
            Some(Instruction::InlineOneState { lvalue_slot })
        }
        ExternGate::Free { n_bits, in_slot } => {
            let in_bits = get_arg(in_slot)?.bits.clone();
            Some(Instruction::InlineFree { n_bits, in_bits })
        }
        ExternGate::ZeroEffect { q_slot } => {
            let q = get_single_bit(q_slot)?;
            Some(Instruction::InlineZeroEffect { q })
        }
        ExternGate::OneEffect { q_slot } => {
            let q = get_single_bit(q_slot)?;
            Some(Instruction::InlineOneEffect { q })
        }
    }
}

fn compile_extern(
    node: &QDefExternNode,
    bloq_key: InternId,
    signature: Vec<RegisterInfo>,
    slots: SlotInfo,
    intern: &InternTable,
) -> Result<CompiledSubroutine, String> {
    let gate = match_extern_gate(&node.cobject_from, &signature, &slots, intern)?;

    Ok(CompiledSubroutine {
        bloq_key,
        signature,
        body: SubroutineBody::Extern(gate),
        slots,
    })
}

fn compile_cast(
    node: &QCastNode,
    bloq_key: InternId,
    signature: Vec<RegisterInfo>,
    slots: SlotInfo,
    intern: &InternTable,
) -> Result<CompiledSubroutine, String> {
    let body = infer_cast_type(&node.bloq_key, &node.qsignature, &signature, &slots, intern)?;

    Ok(CompiledSubroutine {
        bloq_key,
        signature,
        body,
        slots,
    })
}

/// Resolves an already-interned register name to its slot index.
fn resolve_sig_register_slot(
    name: &str,
    slots: &SlotInfo,
    intern: &InternTable,
) -> Result<SlotIdx, String> {
    let intern_id = intern.get_id(name)
        .ok_or_else(|| format!(
            "Register name '{}' was never interned (should have been interned as part of signature)",
            name
        ))?;
    let slot = *slots
        .intern_to_slot
        .get(&intern_id)
        .ok_or_else(|| format!("Register '{}' has no slot assignment", name))?;
    Ok(slot)
}

/// Matches a `CObjectNode` to a known extern gate, resolving each register to
/// its slot index.
fn match_extern_gate(
    cobject: &CObjectNode,
    signature: &[RegisterInfo],
    slots: &SlotInfo,
    intern: &InternTable,
) -> Result<ExternGate, String> {
    match cobject.name.as_str() {
        "qualtran.bloqs.basic_gates.XGate" => {
            assert_no_cargs(cobject);
            let q_slot = resolve_sig_register_slot("q", slots, intern)?;
            Ok(ExternGate::XGate { q_slot })
        }
        "qualtran.bloqs.basic_gates.CNOT" => {
            assert_no_cargs(cobject);
            let ctrl_slot = resolve_sig_register_slot("ctrl", slots, intern)?;
            let target_slot = resolve_sig_register_slot("target", slots, intern)?;
            Ok(ExternGate::CNOT {
                ctrl_slot,
                target_slot,
            })
        }
        "qualtran.bloqs.basic_gates.Toffoli" => {
            assert_no_cargs(cobject);
            let ctrl_slot = resolve_sig_register_slot("ctrl", slots, intern)?;
            let target_slot = resolve_sig_register_slot("target", slots, intern)?;
            Ok(ExternGate::Toffoli {
                ctrl_slot,
                target_slot,
            })
        }
        "qualtran.bloqs.mcmt.And" => {
            let ctrl_slot = resolve_sig_register_slot("ctrl", slots, intern)?;
            let target_slot = resolve_sig_register_slot("target", slots, intern)?;
            let cv = extract_and_cv(cobject)?;
            if cobject.cargs.len() < 3 {
                return Err(format!(
                    "And gate '{}' requires 3 cargs (cv1, cv2, uncompute), got {}",
                    cobject,
                    cobject.cargs.len()
                ));
            }
            if cobject.cargs.len() > 3 {
                return Err(format!(
                    "And gate '{}' has unexpected extra cargs (expected 3, got {})",
                    cobject,
                    cobject.cargs.len()
                ));
            }
            if is_carg_true(&cobject.cargs[2]) {
                Ok(ExternGate::AndDag {
                    ctrl_slot,
                    target_slot,
                    cv,
                })
            } else if is_carg_false(&cobject.cargs[2]) {
                Ok(ExternGate::And {
                    ctrl_slot,
                    target_slot,
                    cv,
                })
            } else {
                Err(format!(
                    "And gate '{}': uncompute carg (3rd) must be True or False, got '{}'",
                    cobject, cobject.cargs[2]
                ))
            }
        }
        "qualtran.bloqs.basic_gates.TwoBitCSwap" => {
            assert_no_cargs(cobject);
            let ctrl_slot = resolve_sig_register_slot("ctrl", slots, intern)?;
            let x_slot = resolve_sig_register_slot("x", slots, intern)?;
            let y_slot = resolve_sig_register_slot("y", slots, intern)?;
            Ok(ExternGate::TwoBitCSwap {
                ctrl_slot,
                x_slot,
                y_slot,
            })
        }
        "qualtran.bloqs.basic_gates.ZGate" => {
            assert_no_cargs(cobject);
            let q_slot = resolve_sig_register_slot("q", slots, intern)?;
            Ok(ExternGate::ZGate { q_slot })
        }
        "qualtran.bloqs.basic_gates.SGate" => {
            if !cobject.cargs.is_empty() {
                if cobject.cargs.len() != 1 {
                    return Err(format!(
                        "SGate '{}' expects 0 or 1 cargs (is_adjoint), got {}",
                        cobject,
                        cobject.cargs.len()
                    ));
                }
                if is_carg_true(&cobject.cargs[0]) {
                    return Err(format!(
                        "SGate adjoint (is_adjoint=True) is not yet supported in fastsim: '{}'",
                        cobject
                    ));
                }
                // is_adjoint=False is fine, equivalent to no cargs
            }
            let q_slot = resolve_sig_register_slot("q", slots, intern)?;
            Ok(ExternGate::SGate { q_slot })
        }
        "qualtran.bloqs.basic_gates.TGate" => {
            if !cobject.cargs.is_empty() {
                if cobject.cargs.len() != 1 {
                    return Err(format!(
                        "TGate '{}' expects 0 or 1 cargs (is_adjoint), got {}",
                        cobject,
                        cobject.cargs.len()
                    ));
                }
                if is_carg_true(&cobject.cargs[0]) {
                    return Err(format!(
                        "TGate adjoint (is_adjoint=True) is not yet supported in fastsim: '{}'",
                        cobject
                    ));
                }
                // is_adjoint=False is fine, equivalent to no cargs
            }
            let q_slot = resolve_sig_register_slot("q", slots, intern)?;
            Ok(ExternGate::TGate { q_slot })
        }
        "qualtran.bloqs.basic_gates.CZGate" => {
            assert_no_cargs(cobject);
            let q_slot = resolve_sig_register_slot("q", slots, intern)?;
            Ok(ExternGate::CZGate { q_slot })
        }
        "qualtran.bloqs.basic_gates.CCZGate" => {
            assert_no_cargs(cobject);
            let q_slot = resolve_sig_register_slot("q", slots, intern)?;
            Ok(ExternGate::CCZGate { q_slot })
        }
        "qualtran.bloqs.basic_gates.ZeroState" => {
            assert_no_cargs(cobject);
            let q_slot = resolve_sig_register_slot("q", slots, intern)?;
            Ok(ExternGate::ZeroState { q_slot })
        }
        "qualtran.bloqs.basic_gates.ZeroEffect" => {
            assert_no_cargs(cobject);
            let q_slot = resolve_sig_register_slot("q", slots, intern)?;
            Ok(ExternGate::ZeroEffect { q_slot })
        }
        "qualtran.bloqs.basic_gates.OneState" => {
            assert_no_cargs(cobject);
            let q_slot = resolve_sig_register_slot("q", slots, intern)?;
            Ok(ExternGate::OneState { q_slot })
        }
        "qualtran.bloqs.basic_gates.OneEffect" => {
            assert_no_cargs(cobject);
            let q_slot = resolve_sig_register_slot("q", slots, intern)?;
            Ok(ExternGate::OneEffect { q_slot })
        }
        "qualtran.bloqs.bookkeeping.Allocate" => {
            validate_allocate_free_cargs(cobject)?;
            let out_reg = signature
                .iter()
                .find(|r| r.direction == RegisterDirection::RightOnly)
                .ok_or_else(|| {
                    format!("Allocate '{}': no RightOnly register in signature", cobject)
                })?;
            let out_slot = *slots.intern_to_slot.get(&out_reg.name).unwrap_or_else(|| {
                panic!(
                    "Allocate '{}': output register has no slot assignment",
                    cobject
                );
            });
            Ok(ExternGate::Allocate {
                n_bits: out_reg.n_bits,
                out_slot,
            })
        }
        "qualtran.bloqs.bookkeeping.Free" => {
            validate_allocate_free_cargs(cobject)?;
            let in_reg = signature
                .iter()
                .find(|r| r.direction == RegisterDirection::LeftOnly)
                .ok_or_else(|| format!("Free '{}': no LeftOnly register in signature", cobject))?;
            let in_slot = *slots.intern_to_slot.get(&in_reg.name).unwrap_or_else(|| {
                panic!("Free '{}': input register has no slot assignment", cobject);
            });
            Ok(ExternGate::Free {
                n_bits: in_reg.n_bits,
                in_slot,
            })
        }
        _ => Err(format!(
            "Unrecognized extern gate: {} (from '{}')",
            cobject.name, cobject
        )),
    }
}

/// Returns `true` if `carg` represents the boolean value `True`.
fn is_carg_true(carg: &CArgNode) -> bool {
    match &carg.value {
        CValueNode::CObject(obj) => obj.name == "True",
        CValueNode::Literal(lit) => match &lit.value {
            LiteralVal::String(s) => s == "True",
            _ => false,
        },
        _ => false,
    }
}

/// Returns `true` if `carg` represents the boolean value `False`.
fn is_carg_false(carg: &CArgNode) -> bool {
    match &carg.value {
        CValueNode::CObject(obj) => obj.name == "False",
        CValueNode::Literal(lit) => match &lit.value {
            LiteralVal::String(s) => s == "False",
            _ => false,
        },
        _ => false,
    }
}

/// Asserts that a CObjectNode has no classical arguments.
/// Panics if unexpected cargs are present, preventing silent misconfiguration.
fn assert_no_cargs(cobject: &CObjectNode) {
    if !cobject.cargs.is_empty() {
        panic!(
            "Extern gate '{}' has unexpected cargs: {}. \
             This gate does not support parameters in fastsim.",
            cobject.name, cobject
        );
    }
}

/// Extracts `[cv1, cv2]` control values from the first two cargs of an And gate.
///
/// Cargs[0] and cargs[1] must be integer literals (0 or 1).
/// cv=1 means positive control (default), cv=0 means negative control.
fn extract_and_cv(cobject: &CObjectNode) -> Result<[bool; 2], String> {
    if cobject.cargs.len() < 2 {
        return Err(format!(
            "And gate '{}' requires at least 2 cargs (cv1, cv2), got {}",
            cobject,
            cobject.cargs.len()
        ));
    }
    let cv1 = extract_int_from_carg(&cobject.cargs[0])?;
    let cv2 = extract_int_from_carg(&cobject.cargs[1])?;
    match (cv1, cv2) {
        (0, 0) => Ok([false, false]),
        (0, 1) => Ok([false, true]),
        (1, 0) => Ok([true, false]),
        (1, 1) => Ok([true, true]),
        _ => Err(format!(
            "And gate '{}': cv values must be 0 or 1, got cv1={}, cv2={}",
            cobject, cv1, cv2
        )),
    }
}

/// Validates cargs for Allocate and Free gates.
///
/// These gates accept either:
/// - 0 cargs (legacy format, dtype comes from the signature)
/// - 2 cargs: `(dtype, dirty)` where dtype is a CObject (redundant with
///   signature, ignored) and dirty must be `False`
///
/// `dirty=True` is not supported because fastsim asserts that freed
/// registers are zero.
fn validate_allocate_free_cargs(cobject: &CObjectNode) -> Result<(), String> {
    match cobject.cargs.len() {
        0 => Ok(()),
        2 => {
            // cargs[0] is the dtype (e.g. QUInt(64)) — redundant with
            // signature, no validation needed.
            // cargs[1] is dirty (must be False).
            if is_carg_true(&cobject.cargs[1]) {
                return Err(format!(
                    "{} with dirty=True is not supported in fastsim: '{}'",
                    cobject.name, cobject
                ));
            }
            if !is_carg_false(&cobject.cargs[1]) {
                return Err(format!(
                    "{} dirty carg (2nd) must be True or False, got '{}': '{}'",
                    cobject.name, cobject.cargs[1], cobject
                ));
            }
            Ok(())
        }
        n => Err(format!(
            "{} expects 0 or 2 cargs (dtype, dirty), got {}: '{}'",
            cobject.name, n, cobject
        )),
    }
}

/// Compiles call-site arguments into [`ArgMapping`]s.
///
/// Resolves each `QArgNode` to caller-side bit slots and maps them to
/// the corresponding callee register slot.
fn compile_call_qargs(
    qargs: &[QArgNode],
    intern: &mut InternTable,
    caller_slots: &mut SlotInfo,
    callee_slots: &SlotInfo,
    callee_name: &str,
    caller_name: &str,
    shape_map: &HashMap<InternId, Vec<usize>>,
) -> Result<Vec<ArgMapping>, String> {
    qargs
        .iter()
        .map(|qarg| {
            let register_name = intern.intern(&qarg.key);
            let bits = resolve_nested_qarg(&qarg.value, intern, caller_slots, shape_map);
            let callee_slot =
                *callee_slots
                    .intern_to_slot
                    .get(&register_name)
                    .ok_or_else(|| {
                        format!(
                    "In '{}': call to '{}': argument '{}' does not match any register in callee",
                    caller_name, callee_name, qarg.key
                )
                    })?;
            Ok(ArgMapping {
                callee_slot,
                bits,
                callee_elements: callee_slots.element_slots.get(&callee_slot).cloned(),
            })
        })
        .collect()
}

/// Compiles return-statement arguments into [`ArgMapping`]s.
///
/// Maps each return argument to its signature register slot within the
/// current subroutine's frame.
fn compile_return_qargs(
    qargs: &[QArgNode],
    intern: &mut InternTable,
    slots: &mut SlotInfo,
    caller_name: &str,
    shape_map: &HashMap<InternId, Vec<usize>>,
) -> Result<Vec<ArgMapping>, String> {
    qargs
        .iter()
        .map(|qarg| {
            let register_name = intern.intern(&qarg.key);
            let bits = resolve_nested_qarg(&qarg.value, intern, slots, shape_map);
            let callee_slot = *slots.intern_to_slot.get(&register_name).ok_or_else(|| {
                format!(
                    "In '{}': return mapping '{}' does not match any signature register",
                    caller_name, qarg.key
                )
            })?;
            Ok(ArgMapping {
                callee_slot,
                bits,
                callee_elements: None,
            })
        })
        .collect()
}

/// Recursively resolves a [`NestedQArgValue`] into a flat list of caller [`SlotIdx`]s.
///
/// - **Indexed references** (e.g. `cube[0,0,1]`): computes the element name and
///   looks up (or assigns) the corresponding element slot.
/// - **Whole-variable references** on shaped registers (e.g. `cube`): expands to
///   all element slots in row-major order.
/// - **Lists**: recursively flattens each item.
fn resolve_nested_qarg(
    value: &NestedQArgValue,
    intern: &mut InternTable,
    slots: &mut SlotInfo,
    shape_map: &HashMap<InternId, Vec<usize>>,
) -> Vec<SlotIdx> {
    match value {
        NestedQArgValue::Leaf(leaf) => {
            let name_id = intern.intern(&leaf.name);
            let parent_slot = slots.get_or_assign(name_id);
            if leaf.idx.is_empty() {
                // Whole-variable reference.
                // If shaped, expand to all element slots in row-major order.
                if let Some(shape) = shape_map.get(&name_id) {
                    let total: usize = shape.iter().product();
                    let mut result = Vec::with_capacity(total);
                    for flat_idx in 0..total {
                        let elem_name = format_element_name(&leaf.name, shape, flat_idx);
                        let elem_id = intern.intern(&elem_name);
                        let elem_slot = slots.get_or_assign(elem_id);
                        // Ensure element slot is registered
                        slots.register_element_slot(parent_slot, elem_slot, flat_idx);
                        result.push(elem_slot);
                    }
                    result
                } else {
                    // Scalar variable — return the slot directly
                    vec![parent_slot]
                }
            } else if leaf.idx.len() == 1 {
                // 1D indexing (e.g. reg[3])
                let flat_idx = leaf.idx[0] as usize;
                let elem_name = format!("{}[{}]", leaf.name, flat_idx);
                let elem_id = intern.intern(&elem_name);
                let elem_slot = slots.get_or_assign(elem_id);
                // Register this element slot under the parent
                slots.register_element_slot(parent_slot, elem_slot, flat_idx);
                vec![elem_slot]
            } else {
                // Multi-dimensional indexing (e.g. cube[0,0,1])
                let shape = shape_map.get(&name_id).unwrap_or_else(|| {
                    panic!(
                        "Multi-dimensional index on '{}' but no shape found in signature",
                        leaf.name
                    );
                });
                assert_eq!(
                    leaf.idx.len(),
                    shape.len(),
                    "Index dimensions ({}) do not match shape dimensions ({}) for '{}'",
                    leaf.idx.len(),
                    shape.len(),
                    leaf.name
                );
                // Compute the flat index from ND indices using row-major order
                let flat_idx = nd_indices_to_flat(&leaf.idx, shape);
                let elem_name = format_element_name(&leaf.name, shape, flat_idx);
                let elem_id = intern.intern(&elem_name);
                let elem_slot = slots.get_or_assign(elem_id);
                slots.register_element_slot(parent_slot, elem_slot, flat_idx);
                vec![elem_slot]
            }
        }
        NestedQArgValue::List(items) => items
            .iter()
            .flat_map(|item| resolve_nested_qarg(item, intern, slots, shape_map))
            .collect(),
    }
}

/// Converts multi-dimensional indices to a flat row-major index.
///
/// For shape `[d0, d1, d2]` and indices `[i0, i1, i2]`:
/// `flat = i0 * (d1 * d2) + i1 * d2 + i2`
fn nd_indices_to_flat(indices: &[i64], shape: &[usize]) -> usize {
    assert_eq!(
        indices.len(),
        shape.len(),
        "nd_indices_to_flat: index count ({}) != shape dimensions ({})",
        indices.len(),
        shape.len()
    );
    let mut flat = 0usize;
    let mut stride = 1usize;
    for (idx, &dim) in indices.iter().zip(shape.iter()).rev() {
        let i = *idx as usize;
        assert!(
            i < dim,
            "nd_indices_to_flat: index {} out of bounds for dimension {}",
            i,
            dim
        );
        flat += i * stride;
        stride *= dim;
    }
    flat
}

/// Converts a flat index back to ND indices (row-major order).
fn flat_index_to_nd_indices(flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0usize; shape.len()];
    let mut remaining = flat;
    for i in (0..shape.len()).rev() {
        indices[i] = remaining % shape[i];
        remaining /= shape[i];
    }
    indices
}

/// Formats an element name for a shaped register.
///
/// For 1D shape `[8]`: `"reg[3]"`
/// For ND shape `[2,2,2]`: `"cube[0,1,0]"`
fn format_element_name(base_name: &str, shape: &[usize], flat_idx: usize) -> String {
    if shape.len() == 1 {
        format!("{}[{}]", base_name, flat_idx)
    } else {
        let nd_indices = flat_index_to_nd_indices(flat_idx, shape);
        let idx_str: Vec<String> = nd_indices.iter().map(|i| i.to_string()).collect();
        format!("{}[{}]", base_name, idx_str.join(","))
    }
}

/// Compiles a quantum signature into [`RegisterInfo`] entries with interned names.
fn compile_signature(
    qsig: &[QSignatureEntry],
    intern: &mut InternTable,
) -> Result<Vec<RegisterInfo>, String> {
    qsig.iter()
        .map(|entry| {
            let name_id = intern.intern(&entry.name);
            let (direction, n_bits, dtype_name_str, shape) = match &entry.dtype {
                SignatureDType::Single(t) => {
                    let nb = compute_dtype_bits(t)?;
                    let dn = t.dtype.name.clone();
                    let sh = extract_shape(t);
                    Ok((RegisterDirection::Thru, nb, dn, sh))
                }
                SignatureDType::Pair((left, right)) => match (left, right) {
                    (None, Some(t)) => {
                        let nb = compute_dtype_bits(t)?;
                        let dn = t.dtype.name.clone();
                        let sh = extract_shape(t);
                        Ok((RegisterDirection::RightOnly, nb, dn, sh))
                    }
                    (Some(t), None) => {
                        let nb = compute_dtype_bits(t)?;
                        let dn = t.dtype.name.clone();
                        let sh = extract_shape(t);
                        Ok((RegisterDirection::LeftOnly, nb, dn, sh))
                    }
                    (Some(left_t), Some(right_t)) => {
                        let nb_left = compute_dtype_bits(left_t)?;
                        let nb_right = compute_dtype_bits(right_t)?;
                        if nb_left != nb_right {
                            return Err(format!(
                                "Cast register '{}': left side has {} bits but right side has {} bits. \
                                 Both sides of a cast must have the same total bit count.",
                                entry.name, nb_left, nb_right
                            ));
                        }
                        let dn = right_t.dtype.name.clone();
                        let sh = extract_shape(right_t);
                        Ok((RegisterDirection::Cast, nb_left, dn, sh))
                    }
                    (None, None) => Err("Invalid signature: both left and right are None".to_string()),
                },
            }?;
            let dtype_name = intern.intern(&dtype_name_str);
            Ok(RegisterInfo {
                name: name_id,
                n_bits,
                direction,
                dtype_name,
                shape,
            })
        })
        .collect()
}

/// Extracts the shape from a [`QDTypeNode`] as `Option<Vec<usize>>`.
///
/// Returns `Some(shape)` if the dtype has a shape (e.g. `QBit[2,2,2]`),
/// or `None` for scalar types (e.g. `QBit`, `QInt(8)`).
fn extract_shape(dtype: &QDTypeNode) -> Option<Vec<usize>> {
    dtype
        .shape
        .as_ref()
        .map(|s| s.iter().map(|&d| d as usize).collect())
}

/// Computes the total bit count for a [`QDTypeNode`], including shape.
pub fn compute_dtype_bits(dtype: &QDTypeNode) -> Result<usize, String> {
    let base_name = &dtype.dtype.name;
    let base_bits = match base_name.as_str() {
        "QBit" => 1,
        "QInt" | "QUInt" | "QAny" | "QMontgomeryUInt" | "QFxp" => {
            if let Some(first_carg) = dtype.dtype.cargs.first() {
                extract_int_from_carg(first_carg)?
            } else {
                return Err(format!("Missing bit width for {}", base_name));
            }
        }
        _ => {
            return Err(format!("Unknown quantum data type: {}", base_name));
        }
    };

    // Handle shape: QBit[8] means 8 qubits
    if let Some(ref shape) = dtype.shape {
        let total: usize = shape.iter().map(|&s| s as usize).product();
        Ok(base_bits * total)
    } else {
        Ok(base_bits)
    }
}

/// Extracts an integer value from a [`CArgNode`].
fn extract_int_from_carg(carg: &CArgNode) -> Result<usize, String> {
    match &carg.value {
        CValueNode::Literal(lit) => match &lit.value {
            LiteralVal::Int(v) => Ok(*v as usize),
            _ => Err(format!("Expected integer literal, got {:?}", lit.value)),
        },
        _ => Err(format!("Expected literal carg, got {:?}", carg.value)),
    }
}

/// Infers the [`SubroutineBody`] for a qcast node.
///
/// Single-register casts (`Pair(Some, Some)`) are Split, Join, or identity.
/// Multi-register casts model `_PartitionBase` bloqs: a bit-preserving
/// rewiring where every register is strictly one-sided (LEFT or RIGHT), and
/// the concatenated LEFT bits equal the concatenated RIGHT bits.
/// Single-register allocate/free signatures are rejected — those must use
/// `extern qdef`.
fn infer_cast_type(
    bloq_key: &str,
    qsig: &[QSignatureEntry],
    signature: &[RegisterInfo],
    slots: &SlotInfo,
    intern: &InternTable,
) -> Result<SubroutineBody, String> {
    if qsig.is_empty() {
        return Err(format!("Cast {} has empty signature", bloq_key));
    }
    assert!(
        !signature.is_empty(),
        "Cast {} has non-empty qsig but empty compiled signature",
        bloq_key
    );

    // Multi-register cast: a `_PartitionBase` bloq (Partition/Split2/Join2 and
    // adjoints). Each register must be strictly one-sided; inputs are LEFT
    // registers, outputs are RIGHT registers, both in signature order.
    if signature.len() > 1 {
        return infer_partition_cast(bloq_key, signature, slots, intern);
    }

    let entry = &qsig[0];
    match &entry.dtype {
        SignatureDType::Pair((left, right)) => match (left, right) {
            (None, Some(_)) => Err(format!(
                "qcast '{}' has allocate signature (| -> T). \
                     Allocation must use 'extern qdef' (e.g. ZeroState, Allocate), not 'qcast'.",
                bloq_key
            )),
            (Some(_), None) => Err(format!(
                "qcast '{}' has free signature (T -> |). \
                     Deallocation must use 'extern qdef' (e.g. ZeroEffect, Free), not 'qcast'.",
                bloq_key
            )),
            (Some(left_t), Some(right_t)) => {
                let left_n = compute_dtype_bits(left_t)?;
                let right_n = compute_dtype_bits(right_t)?;
                if left_n != right_n {
                    return Err(format!(
                        "qcast '{}': left bitsize ({}) != right bitsize ({}). \
                         Casts must preserve total bit count.",
                        bloq_key, left_n, right_n,
                    ));
                }
                let reg_slot = *slots
                    .intern_to_slot
                    .get(&signature[0].name)
                    .unwrap_or_else(|| {
                        panic!("Cast {}: register has no slot assignment", bloq_key);
                    });
                // Determine Split vs Join from the bloq_key prefix.
                // Identity casts use Join semantics (no-op on bits).
                let cast_op = if bloq_key.starts_with("Split") {
                    CastOp::Split {
                        total_bits: left_n,
                        reg_slot,
                    }
                } else if bloq_key.starts_with("Join") {
                    CastOp::Join {
                        total_bits: left_n,
                        reg_slot,
                    }
                } else {
                    CastOp::Join {
                        total_bits: left_n,
                        reg_slot,
                    }
                };
                Ok(SubroutineBody::Cast(cast_op))
            }
            _ => Err(format!("Invalid cast signature for {}", bloq_key)),
        },
        _ => Err(format!(
            "Cast {} expected Pair signature, got Single",
            bloq_key
        )),
    }
}

/// Builds a [`CastOp::Partition`] for a multi-register `_PartitionBase` cast.
///
/// Requires every register to be strictly one-sided (`LeftOnly` or
/// `RightOnly`). Inputs are the LEFT registers and outputs are the RIGHT
/// registers, each collected in signature order. The summed input bits must
/// equal the summed output bits (the partition invariant); any deviation is a
/// compile error rather than a silently-accepted mismatch.
fn infer_partition_cast(
    bloq_key: &str,
    signature: &[RegisterInfo],
    slots: &SlotInfo,
    intern: &InternTable,
) -> Result<SubroutineBody, String> {
    let mut input_slots: Vec<(SlotIdx, usize)> = Vec::new();
    let mut output_slots: Vec<(SlotIdx, usize)> = Vec::new();

    for reg in signature {
        let slot = *slots.intern_to_slot.get(&reg.name).ok_or_else(|| {
            format!(
                "Partition cast '{}': register '{}' has no slot assignment",
                bloq_key,
                intern.resolve(reg.name),
            )
        })?;
        match reg.direction {
            RegisterDirection::LeftOnly => input_slots.push((slot, reg.n_bits)),
            RegisterDirection::RightOnly => output_slots.push((slot, reg.n_bits)),
            RegisterDirection::Thru | RegisterDirection::Cast => {
                return Err(format!(
                    "Partition cast '{}': register '{}' is two-sided ({:?}), but multi-register \
                     partitions require every register to be strictly one-sided (LEFT or RIGHT).",
                    bloq_key,
                    intern.resolve(reg.name),
                    reg.direction,
                ));
            }
        }
    }

    if input_slots.is_empty() || output_slots.is_empty() {
        return Err(format!(
            "Partition cast '{}': expected registers on both sides, found {} input(s) and {} output(s).",
            bloq_key,
            input_slots.len(),
            output_slots.len(),
        ));
    }

    let input_bits: usize = input_slots.iter().map(|(_, n)| n).sum();
    let output_bits: usize = output_slots.iter().map(|(_, n)| n).sum();
    if input_bits != output_bits {
        return Err(format!(
            "Partition cast '{}': total input bits ({}) != total output bits ({}). \
             Partitions must preserve total bit count.",
            bloq_key, input_bits, output_bits,
        ));
    }

    Ok(SubroutineBody::Cast(CastOp::Partition {
        input_slots,
        output_slots,
        total_bits: input_bits,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;

    #[test]
    fn test_compile_negate() {
        let source = std::fs::read_to_string("example_qlts/negate.qlt").unwrap();
        let (module, errors) = parser::parse_l1_module(&source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);

        let compiled = compile(&module).expect("Compilation failed");

        // Should have the top-level entry point
        assert!(compiled.has_subroutine("Negate"));
        assert!(compiled.has_subroutine("BitwiseNot(8)"));
        assert!(compiled.has_subroutine("AddK(k=1)"));
        assert!(compiled.has_subroutine("X(oneach=8)"));
        assert!(compiled.has_subroutine("XorK(1)"));
        assert!(compiled.has_subroutine("X"));
        assert!(compiled.has_subroutine("And"));
        assert!(compiled.has_subroutine("CNOT"));
        assert!(compiled.has_subroutine("And_dag"));
        assert!(compiled.has_subroutine("Split(QInt(8))"));
        assert!(compiled.has_subroutine("Join(QInt(8))"));
        assert!(compiled.has_subroutine("Allocate(QInt(8))"));
        assert!(compiled.has_subroutine("Free(QInt(8))"));

        // Allocate and Free should be compiled as extern gates, not casts
        let alloc_sub = compiled.get_subroutine("Allocate(QInt(8))").unwrap();
        assert!(matches!(
            alloc_sub.body,
            SubroutineBody::Extern(ExternGate::Allocate { n_bits: 8, .. })
        ));
        let free_sub = compiled.get_subroutine("Free(QInt(8))").unwrap();
        assert!(matches!(
            free_sub.body,
            SubroutineBody::Extern(ExternGate::Free { n_bits: 8, .. })
        ));

        // Verify X is an extern gate
        let x_sub = compiled.get_subroutine("X").unwrap();
        assert!(matches!(
            x_sub.body,
            SubroutineBody::Extern(ExternGate::XGate { .. })
        ));

        // Verify Split is a Cast
        let split_sub = compiled.get_subroutine("Split(QInt(8))").unwrap();
        assert!(matches!(
            split_sub.body,
            SubroutineBody::Cast(CastOp::Split { total_bits: 8, .. })
        ));
    }

    #[test]
    fn test_compile_zero_one_state() {
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

qdef PrepOne
[
    q: | -> QBit,
] {
    q                    = OneState         []
                           return           [q=q]
}

qdef MeasZero
[
    q: QBit -> |,
] {
    |                    = ZeroEffect       [q=q]
}

qdef MeasOne
[
    q: QBit -> |,
] {
    |                    = OneEffect        [q=q]
}

extern qdef ZeroState
from qualtran.bloqs.basic_gates.ZeroState
[q: | -> QBit]

extern qdef ZeroEffect
from qualtran.bloqs.basic_gates.ZeroEffect
[q: QBit -> |]

extern qdef OneState
from qualtran.bloqs.basic_gates.OneState
[q: | -> QBit]

extern qdef OneEffect
from qualtran.bloqs.basic_gates.OneEffect
[q: QBit -> |]
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);

        let compiled = compile(&module).expect("Compilation failed");

        // Verify ZeroState is compiled as an extern gate
        let zs = compiled.get_subroutine("ZeroState").unwrap();
        assert!(matches!(
            zs.body,
            SubroutineBody::Extern(ExternGate::ZeroState { .. })
        ));

        // Verify ZeroEffect is compiled as an extern gate
        let ze = compiled.get_subroutine("ZeroEffect").unwrap();
        assert!(matches!(
            ze.body,
            SubroutineBody::Extern(ExternGate::ZeroEffect { .. })
        ));

        // Verify OneState is compiled as an extern gate
        let os = compiled.get_subroutine("OneState").unwrap();
        assert!(matches!(
            os.body,
            SubroutineBody::Extern(ExternGate::OneState { .. })
        ));

        // Verify OneEffect is compiled as an extern gate
        let oe = compiled.get_subroutine("OneEffect").unwrap();
        assert!(matches!(
            oe.body,
            SubroutineBody::Extern(ExternGate::OneEffect { .. })
        ));
    }

    #[test]
    fn test_compile_multi_register_partition() {
        // A multi-register partition: one lumped QAny(8) input split into two
        // output registers (a QUInt(3) and a QBit[5]). This models a
        // `_PartitionBase` bloq and must compile to `CastOp::Partition`.
        let source = r#"
# Qualtran-L1
# 1.0.0

qcast Partition
[
    x: QAny(8) -> |,
    lo: | -> QUInt(3),
    hi: | -> QBit[5],
]
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);

        let compiled = compile(&module).expect("Compilation failed");
        let sub = compiled.get_subroutine("Partition").unwrap();

        match &sub.body {
            SubroutineBody::Cast(CastOp::Partition {
                input_slots,
                output_slots,
                total_bits,
            }) => {
                assert_eq!(*total_bits, 8);
                // One input register (x, 8 bits).
                assert_eq!(input_slots.len(), 1);
                assert_eq!(input_slots[0].1, 8);
                // Two output registers in signature order: lo (3), hi (5).
                assert_eq!(output_slots.len(), 2);
                assert_eq!(output_slots[0].1, 3);
                assert_eq!(output_slots[1].1, 5);
            }
            other => panic!("Expected CastOp::Partition, got {:?}", other),
        }
    }

    #[test]
    fn test_qcast_partition_bit_mismatch_rejected() {
        // Total input bits (8) != total output bits (3+4=7); must be rejected.
        let source = r#"
# Qualtran-L1
# 1.0.0

qcast Partition
[
    x: QAny(8) -> |,
    lo: | -> QUInt(3),
    hi: | -> QBit[4],
]
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);

        let result = compile(&module);
        assert!(result.is_err(), "Partition with bit mismatch should be rejected");
        let err = result.unwrap_err();
        assert!(
            err.contains("total input bits") && err.contains("total output bits"),
            "Error: {}",
            err
        );
    }

    #[test]
    fn test_qcast_allocate_rejected() {
        let source = r#"
# Qualtran-L1
# 1.0.0

qcast Allocate(QInt(8))
[reg: | -> QInt(8)]
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);

        let result = compile(&module);
        assert!(result.is_err(), "qcast Allocate should be rejected");
        let err = result.unwrap_err();
        assert!(
            err.contains("Allocation must use 'extern qdef'"),
            "Error: {}",
            err
        );
    }

    #[test]
    fn test_qcast_free_rejected() {
        let source = r#"
# Qualtran-L1
# 1.0.0

qcast Free(QInt(8))
[reg: QInt(8) -> |]
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);

        let result = compile(&module);
        assert!(result.is_err(), "qcast Free should be rejected");
        let err = result.unwrap_err();
        assert!(
            err.contains("Deallocation must use 'extern qdef'"),
            "Error: {}",
            err
        );
    }

    #[test]
    fn test_qcast_bitsize_mismatch_rejected() {
        let source = r#"
# Qualtran-L1
# 1.0.0

qcast BadCast
[reg: QUInt(8) -> QUInt(16)]
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);

        let result = compile(&module);
        assert!(
            result.is_err(),
            "qcast with mismatched bitsizes should be rejected"
        );
        let err = result.unwrap_err();
        assert!(
            err.contains("left side has 8 bits but right side has 16 bits"),
            "Error: {}",
            err
        );
    }

    #[test]
    fn test_compile_cswap() {
        let source = std::fs::read_to_string("example_qlts/cswap.qlt").unwrap();
        let (module, errors) = parser::parse_l1_module(&source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);

        let compiled = compile(&module).expect("Compilation failed");

        assert!(compiled.has_subroutine("CSwap"));
        assert!(compiled.has_subroutine("TwoBitCSwap"));
        assert!(compiled.has_subroutine("Split(5)"));
        assert!(compiled.has_subroutine("Join(5)"));

        // Verify TwoBitCSwap is an extern gate
        let tbs = compiled.get_subroutine("TwoBitCSwap").unwrap();
        assert!(matches!(
            tbs.body,
            SubroutineBody::Extern(ExternGate::TwoBitCSwap { .. })
        ));
    }

    #[test]
    fn test_unrecognized_extern() {
        // Create a minimal module with an unrecognized extern
        let source = r#"
# Qualtran-L1
# 1.0.0

extern qdef BadGate
from unknown.module.BadGate
[q: QBit]
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);

        let result = compile(&module);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Unrecognized extern gate"), "Error: {}", err);
    }

    #[test]
    fn test_intern_table_roundtrip() {
        let mut intern = InternTable::new();
        let id_a = intern.intern("hello");
        let id_b = intern.intern("world");
        let id_a2 = intern.intern("hello");
        assert_eq!(id_a, id_a2, "Re-interning same string must return same ID");
        assert_ne!(id_a, id_b, "Different strings must get different IDs");
        assert_eq!(intern.resolve(id_a), "hello");
        assert_eq!(intern.resolve(id_b), "world");
        assert_eq!(intern.get_id("hello"), Some(id_a));
        assert_eq!(intern.get_id("nonexistent"), None);
    }

    #[test]
    fn test_slot_assignment() {
        let source = std::fs::read_to_string("example_qlts/negate.qlt").unwrap();
        let (module, errors) = parser::parse_l1_module(&source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);

        let compiled = compile(&module).expect("Compilation failed");

        // Every subroutine should have at least as many slots as signature registers
        for sub in &compiled.subroutines {
            assert!(
                sub.slots.n_slots >= sub.signature.len(),
                "Subroutine '{}' has {} slots but {} signature registers",
                compiled.intern_table.resolve(sub.bloq_key),
                sub.slots.n_slots,
                sub.signature.len()
            );
            // Every signature register should have a slot
            for reg in &sub.signature {
                assert!(
                    sub.slots.intern_to_slot.contains_key(&reg.name),
                    "Subroutine '{}': signature register '{}' has no slot assignment",
                    compiled.intern_table.resolve(sub.bloq_key),
                    compiled.intern_table.resolve(reg.name),
                );
            }
        }
    }

    #[test]
    fn test_alias_basic_resolution() {
        // A single alias resolving to a known subroutine compiles successfully.
        let source = r#"
# Qualtran-L1
# 1.0.0

qdef Caller
[
    q: QBit,
] {
    myalias              = Target
    q                    = myalias          [q=q]
                           return           [q=q]
}

qdef Target
[
    q: QBit,
] {
    q                    = X                [q=q]
                           return           [q=q]
}

extern qdef X
from qualtran.bloqs.basic_gates.XGate
[q: QBit]
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        let compiled = compile(&module).expect("Compilation failed");

        // The alias "myalias" should not appear in the global intern_to_sub map
        // since aliases are now scoped per-subroutine.
        if let Some(alias_id) = compiled.intern_table.get_id("myalias") {
            assert!(
                !compiled.intern_to_sub.contains_key(&alias_id),
                "Alias should NOT be in the global intern_to_sub map"
            );
        }
    }

    #[test]
    fn test_alias_scoping_same_name_different_targets() {
        // Two subroutines both define alias "op" but pointing to different targets.
        // This is the exact bug pattern from newton-raphson-15.qlt where multiple
        // COutOfPlaceAdder variants each define `oopa` pointing to different
        // OutOfPlaceAdder variants.
        let source = r#"
# Qualtran-L1
# 1.0.0

qdef CallerA
[
    q: QBit,
] {
    op                   = TargetA
    q                    = op               [q=q]
                           return           [q=q]
}

qdef CallerB
[
    q: QBit,
] {
    op                   = TargetB
    q                    = op               [q=q]
                           return           [q=q]
}

qdef TargetA
[
    q: QBit,
] {
    q                    = X                [q=q]
                           return           [q=q]
}

qdef TargetB
[
    q: QBit,
] {
                           return           [q=q]
}

extern qdef X
from qualtran.bloqs.basic_gates.XGate
[q: QBit]
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        let compiled = compile(&module).expect("Compilation failed");

        // CallerA's call should target TargetA (index of TargetA in subroutines)
        let caller_a = compiled.get_subroutine("CallerA").unwrap();
        let caller_b = compiled.get_subroutine("CallerB").unwrap();

        let target_a_idx = *compiled
            .intern_to_sub
            .get(&compiled.intern_table.get_id("TargetA").unwrap())
            .unwrap();
        let target_b_idx = *compiled
            .intern_to_sub
            .get(&compiled.intern_table.get_id("TargetB").unwrap())
            .unwrap();

        // Verify the two targets are actually different subroutines.
        assert_ne!(
            target_a_idx, target_b_idx,
            "TargetA and TargetB should be different subroutines"
        );

        // Extract the call target from CallerA's compiled instructions.
        if let SubroutineBody::Impl(instrs) = &caller_a.body {
            let call_target = instrs
                .iter()
                .find_map(|i| {
                    if let Instruction::Call { callee, .. } = i {
                        Some(*callee)
                    } else {
                        None
                    }
                })
                .expect("CallerA should have a Call instruction");
            assert_eq!(
                call_target, target_a_idx,
                "CallerA's alias 'op' should resolve to TargetA, not TargetB"
            );
        } else {
            panic!("CallerA should be an Impl subroutine");
        }

        // Extract the call target from CallerB's compiled instructions.
        if let SubroutineBody::Impl(instrs) = &caller_b.body {
            let call_target = instrs
                .iter()
                .find_map(|i| {
                    if let Instruction::Call { callee, .. } = i {
                        Some(*callee)
                    } else {
                        None
                    }
                })
                .expect("CallerB should have a Call instruction");
            assert_eq!(
                call_target, target_b_idx,
                "CallerB's alias 'op' should resolve to TargetB, not TargetA"
            );
        } else {
            panic!("CallerB should be an Impl subroutine");
        }
    }

    #[test]
    fn test_alias_does_not_leak_between_subroutines() {
        // SubA defines alias "op" → Target, but SubB tries to call "op" without
        // defining it. This must fail — aliases must not leak.
        let source = r#"
# Qualtran-L1
# 1.0.0

qdef SubA
[
    q: QBit,
] {
    op                   = Target
    q                    = op               [q=q]
                           return           [q=q]
}

qdef SubB
[
    q: QBit,
] {
    q                    = op               [q=q]
                           return           [q=q]
}

qdef Target
[
    q: QBit,
] {
                           return           [q=q]
}
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);

        let result = compile(&module);
        assert!(
            result.is_err(),
            "SubB calling undefined alias 'op' should fail"
        );
        let err = result.unwrap_err();
        assert!(err.contains("call target 'op' not found"), "Error: {}", err);
    }

    #[test]
    fn test_alias_unknown_target_rejected() {
        // An alias pointing to a non-existent subroutine must produce an error.
        let source = r#"
# Qualtran-L1
# 1.0.0

qdef Caller
[
    q: QBit,
] {
    op                   = DoesNotExist
    q                    = op               [q=q]
                           return           [q=q]
}
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);

        let result = compile(&module);
        assert!(result.is_err(), "Alias to non-existent target should fail");
        let err = result.unwrap_err();
        assert!(
            err.contains("alias 'op' references unknown target 'DoesNotExist'"),
            "Error: {}",
            err
        );
    }

    #[test]
    fn test_and_cv_default() {
        let source = r#"
# Qualtran-L1
# 1.0.0

extern qdef And
from qualtran.bloqs.mcmt.And(1, 1, False)
[ctrl: QBit[2], target: | -> QBit]
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        let compiled = compile(&module).expect("Compilation failed");
        let sub = compiled.get_subroutine("And").unwrap();
        match &sub.body {
            SubroutineBody::Extern(ExternGate::And { cv, .. }) => {
                assert_eq!(*cv, [true, true]);
            }
            other => panic!("Expected And extern gate, got {:?}", other),
        }
    }

    #[test]
    fn test_and_cv_negative_controls() {
        let source = r#"
# Qualtran-L1
# 1.0.0

extern qdef And
from qualtran.bloqs.mcmt.And(0, 0, False)
[ctrl: QBit[2], target: | -> QBit]
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        let compiled = compile(&module).expect("Compilation failed");
        let sub = compiled.get_subroutine("And").unwrap();
        match &sub.body {
            SubroutineBody::Extern(ExternGate::And { cv, .. }) => {
                assert_eq!(*cv, [false, false]);
            }
            other => panic!("Expected And extern gate, got {:?}", other),
        }
    }

    #[test]
    fn test_and_cv_mixed() {
        let source = r#"
# Qualtran-L1
# 1.0.0

extern qdef And
from qualtran.bloqs.mcmt.And(0, 1, False)
[ctrl: QBit[2], target: | -> QBit]
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        let compiled = compile(&module).expect("Compilation failed");
        let sub = compiled.get_subroutine("And").unwrap();
        match &sub.body {
            SubroutineBody::Extern(ExternGate::And { cv, .. }) => {
                assert_eq!(*cv, [false, true]);
            }
            other => panic!("Expected And extern gate, got {:?}", other),
        }
    }

    #[test]
    fn test_and_dag_cv() {
        let source = r#"
# Qualtran-L1
# 1.0.0

extern qdef And_dag
from qualtran.bloqs.mcmt.And(0, 1, True)
[ctrl: QBit[2], target: QBit -> |]
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        let compiled = compile(&module).expect("Compilation failed");
        let sub = compiled.get_subroutine("And_dag").unwrap();
        match &sub.body {
            SubroutineBody::Extern(ExternGate::AndDag { cv, .. }) => {
                assert_eq!(*cv, [false, true]);
            }
            other => panic!("Expected AndDag extern gate, got {:?}", other),
        }
    }

    #[test]
    fn test_and_missing_cargs_rejected() {
        let source = r#"
# Qualtran-L1
# 1.0.0

extern qdef And
from qualtran.bloqs.mcmt.And
[ctrl: QBit[2], target: | -> QBit]
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        let result = compile(&module);
        assert!(result.is_err(), "Expected error for And with no cargs");
    }

    #[test]
    #[should_panic(expected = "unexpected cargs")]
    fn test_xgate_with_cargs_rejected() {
        let source = r#"
# Qualtran-L1
# 1.0.0

extern qdef X
from qualtran.bloqs.basic_gates.XGate(42)
[q: QBit @ oplus]
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        let _ = compile(&module);
    }

    #[test]
    fn test_allocate_with_cargs_accepted() {
        let source = r#"
# Qualtran-L1
# 1.0.0

extern qdef Allocate(QUInt(64))
from qualtran.bloqs.bookkeeping.Allocate(QUInt(64), False)
[reg: | -> QUInt(64)]
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        let compiled = compile(&module).expect("Compilation failed");
        let sub = compiled.get_subroutine("Allocate(QUInt(64))").unwrap();
        assert!(matches!(
            sub.body,
            SubroutineBody::Extern(ExternGate::Allocate { n_bits: 64, .. })
        ));
    }

    #[test]
    fn test_allocate_no_cargs_accepted() {
        let source = r#"
# Qualtran-L1
# 1.0.0

extern qdef Allocate(QInt(8))
from qualtran.bloqs.bookkeeping.Allocate
[reg: | -> QInt(8)]
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        let compiled = compile(&module).expect("Compilation failed");
        let sub = compiled.get_subroutine("Allocate(QInt(8))").unwrap();
        assert!(matches!(
            sub.body,
            SubroutineBody::Extern(ExternGate::Allocate { n_bits: 8, .. })
        ));
    }

    #[test]
    fn test_allocate_dirty_rejected() {
        let source = r#"
# Qualtran-L1
# 1.0.0

extern qdef Allocate(QUInt(8))
from qualtran.bloqs.bookkeeping.Allocate(QUInt(8), True)
[reg: | -> QUInt(8)]
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        let result = compile(&module);
        assert!(result.is_err(), "Expected error for dirty Allocate");
        assert!(
            result.unwrap_err().contains("dirty=True"),
            "Error should mention dirty=True"
        );
    }

    #[test]
    fn test_free_dirty_rejected() {
        let source = r#"
# Qualtran-L1
# 1.0.0

extern qdef Free(QUInt(8))
from qualtran.bloqs.bookkeeping.Free(QUInt(8), True)
[reg: QUInt(8) -> |]
"#;
        let (module, errors) = parser::parse_l1_module(source);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);
        let result = compile(&module);
        assert!(result.is_err(), "Expected error for dirty Free");
        assert!(
            result.unwrap_err().contains("dirty=True"),
            "Error should mention dirty=True"
        );
    }
}
