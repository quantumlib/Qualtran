# `fastsim` Module Reference Documentation

The `fastsim` module provides a high-performance bytecode compiler and virtual machine (VM) executor for simulating quantum programs in the Qualtran Intermediate Representation (`.qlt`) format on classical basis-state inputs.

---

## Architectural Overview

The execution pipeline consists of three main components:
1. **[compiler](compiler.rs)** translates a parsed AST (`L1Module`) into an optimized `CompiledModule`.
2. **[vm](vm.rs)** executes the compiled program on classical basis states using `VmSimulator`. The simulator retains and reuses memory allocations across execution runs to ensure high throughput during batched simulation.
3. **[gates](gates.rs)** implements the classical logic routines for fundamental quantum gates (such as `X`, `CNOT`, `Toffoli`, `And`, `CSwap`, `Z`, `S`, `T`, `CZ`, `CCZ`).

---

## Module Breakdown & Public API

### 1. The Virtual Machine Simulator (`vm`)

The `vm` module provides the execution environment (`VmSimulator`), state inspection (`SimState`), and utilities for converting classical values to and from bit vectors.

#### `VmSimulator<'a>`
The primary entrypoint for executing compiled subroutines. `VmSimulator` manages the simulation state and preserves internal buffers across multiple runs for maximum performance.

```rust
pub struct VmSimulator<'a> { /* private fields */ }
```

##### Public Methods

- `pub fn new(compiled: &'a CompiledModule, entrypoint: &str) -> Result<Self, String>`
  Initializes a new VM simulator for a given compiled module and entrypoint subroutine name.

- `pub fn execute_run(&mut self, input_values: &[(String, Vec<bool>)]) -> Result<(), String>`
  Resets the simulator state and executes the entrypoint subroutine with the provided inputs.

- `pub fn extract_outputs(&self) -> Vec<(String, String, String)>`
  Extracts the final output registers as a list of tuples: `(name, value_string, dtype_string)`.

- `pub fn phase_exponent(&self) -> f64`
  Retrieves the accumulated global phase exponent (the value `x` such that the global phase is `exp(iπx)`).

##### Usage Example
```rust
use _rsqlt::fastsim::vm::VmSimulator;

let mut simulator = VmSimulator::new(&compiled_module, "EntrypointSubroutine")?;
simulator.execute_run(&[("x".to_string(), vec![false, true, true])])?;
let outputs = simulator.extract_outputs();
let phase_exponent = simulator.phase_exponent();
```

---

#### `SimState`
Represents the classical basis state of the simulation, tracking allocated bits and the global phase exponent accumulator.

The phase exponent tracks the value `x` in `exp(iπx)`. To recover the
complex phase, compute `exp(iπ * phase_exponent)`.

```rust
pub struct SimState {
    pub bits: Vec<bool>,
    pub phase_exponent: f64,
}
```

##### Public Methods

- `pub fn new(capacity_bits: usize) -> Self`
  Allocates a new simulation state with pre-reserved bit capacity.

- `pub fn from_bits(bits: impl IntoIterator<Item = bool>) -> Self`
  Creates a state initialized with a specific sequence of bits.

- `pub fn get_bit(&self, index: usize) -> bool`
  Retrieves the bit value at the specified index.

- `pub fn set_bit(&mut self, index: usize, val: bool)`
  Sets the bit value at the specified index.

- `pub fn flip_bit(&mut self, index: usize)`
  Inverts the bit value at the specified index.

- `pub fn push_bit(&mut self, val: bool) -> usize`
  Appends a new bit to the state and returns its index.

- `pub fn extend_bits(&mut self, vals: impl IntoIterator<Item = bool>)`
  Appends multiple bits to the state.

- `pub fn extend_false(&mut self, n: usize)`
  Allocates `n` new bits initialized to `false`.

- `pub fn clear(&mut self)`
  Resets the bit vector and phase exponent accumulator.

- `pub fn len(&self) -> usize`
  Returns the current number of allocated bits in the state.

- `pub fn is_empty(&self) -> bool`
  Returns `true` if no bits are currently allocated.

---

#### Bit Conversion Utilities
Functions for converting between native Rust integers, decimal strings, and big-endian bit vectors (`Vec<bool>`).

```rust
pub fn int_to_bits(value: i64, n_bits: usize) -> Vec<bool>
pub fn bits_to_uint(bits: &[bool]) -> u64
pub fn bits_to_int(bits: &[bool]) -> i64
pub fn decimal_str_to_bits(s: &str, n_bits: usize) -> Vec<bool>
pub fn signed_decimal_str_to_bits(s: &str, n_bits: usize) -> Vec<bool>
pub fn bits_to_uint_str(bits: &[bool]) -> String
pub fn bits_to_int_str(bits: &[bool]) -> String
```

---

### 2. The Bytecode Compiler (`compiler`)

The `compiler` module converts an AST representation into an executable `CompiledModule`.

#### `compile`
The primary entrypoint for compiling an `L1Module`.

```rust
pub fn compile(module: &L1Module) -> Result<CompiledModule, String>
```

---

#### `CompiledModule`
Contains the fully compiled program ready for execution by `VmSimulator`.

```rust
pub struct CompiledModule { /* fields */ }
```

##### Public Methods

- `pub fn get_subroutine(&self, key: &str) -> Option<&CompiledSubroutine>`
  Retrieves a compiled subroutine by name.

- `pub fn has_subroutine(&self, key: &str) -> bool`
  Checks if a subroutine exists in the module.

---

### 3. Built-in Gate Implementations (`gates`)

The `gates` module exposes public functions for directly applying fundamental quantum operations to a `SimState`.

```rust
pub fn apply_x(state: &mut SimState, bits: &[usize])
pub fn apply_cnot(state: &mut SimState, ctrl_bits: &[usize], target_bits: &[usize])
pub fn apply_and(state: &mut SimState, ctrl_bits: &[usize]) -> usize
pub fn apply_and_dag(state: &mut SimState, ctrl_bits: &[usize], target_bit: usize)
pub fn apply_twobitcswap(state: &mut SimState, ctrl_bits: &[usize], x_bits: &[usize], y_bits: &[usize])
pub fn apply_z(state: &mut SimState, bits: &[usize])
pub fn apply_s(state: &mut SimState, bits: &[usize])
pub fn apply_t(state: &mut SimState, bits: &[usize])
pub fn apply_cz(state: &mut SimState, q1_bits: &[usize], q2_bits: &[usize])
pub fn apply_ccz(state: &mut SimState, q1_bits: &[usize], q2_bits: &[usize], q3_bits: &[usize])
```
