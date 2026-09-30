# The Kickmix Circuit File Format (.kmx)

This document describes the kickmix circuit file format (.kmx).
A kickmix circuit file is a human-readable specification of a quantum circuit composed of
phased classical reversible arithmetic, with support for feedback and X-basis demolition measurements.
These circuits are designed to be efficient to simulate and easy to analyze.

## Index

- [Version Differences](#Version_Differences)
- [Encoding](#Encoding)
- [Syntax](#Syntax)
- [Semantics](#Semantics)
- [Examples](#Examples)


## Version Differences

Version 1 of the kickmix format was defined in order to have a well defined format
to target for the zero knowledge proof in the paper https://arxiv.org/abs/2603.28846
(see the [zenodo upload of that paper](https://doi.org/10.5281/ZENODO.19597130)
for version 1 of this document).

Version 2 (the version described in this document) was released later, to support
a Kaggle competition.
Key additions were the ability to name registers (the `REGISTER` instruction
can take an additional optional token specifying the name) and the ability
to perform fine-grained rotations (the `Z_POW` instruction was added).
(This release also specified the complementary [binary kickmix format](kickmix_binary_format.md),
which uses less space and is an order of magnitude faster to parse.)

## Encoding

Kickmix circuit files are always encoded using UTF-8.
Furthermore, the only place in the file where non-ASCII characters are permitted is inside of comments.

## Syntax

A kickmix circuit file is made up of a series of lines.
Each line is either blank or an instruction.
Also, each line may be indented with spacing characters and may end with a comment indicated by a hash (`#`).
Comments and indentation are purely decorative; they carry no semantic significance.

Here is a formal definition of the above paragraph.
Entries like `/this/` are regular expressions.
Entries like `<this>` are named expressions.
Entries like `'this'` are literal string expressions.
The `::=` operator means "defined as".
The `|` binary operator means "or".
The `?` suffix operator means "zero-or-one".
The `*` suffix operator means "zero-or-many".
Parens are used to group expressions.
Adjacent expressions are combined by concatenation.

```
<CIRCUIT> ::= <LINE>*
<LINE> ::= <INDENT> <INSTRUCTION>? <COMMENT>? '\n'
<INDENT> ::= /[ \t]*/
<COMMENT> ::= '#' /[^\n]*/
```

An *instruction* is composed of a name,
then some number of space-separated targets,
then an optional "if" specifying a condition bit.
For example, the line `CX q5 q6 if b7` is an instruction with a
name (`CX`), two qubit targets (`q5`, `q6`), and a condition (`b7`).

```
<INSTRUCTION> ::= <NAME> <TARGET>* <CONDITION>? <QUOTED_TEXT>?
<CONDITION> ::= 'if' /[ \t]+/ <BIT_ID>
<TARGET> ::= <QUBIT_ID> | <BIT_ID> | <REGISTER_ID> | <EXPONENT>
```

An instruction *name* starts with a letter and then contains a series of letters, digits, and underscores.
Names are always upper case.
Qubit ids, bit ids, and register ids are non-negative integers prefixed by a type-identifying character
('q' for qubit, 'b' for bit, and 'r' for register).
Exponents are specified in decimal without scientific notation
(they are *not* floating point numbers).

```
<NAME> ::= /[A-Z][A-Z0-9_]*/ 
<QUBIT_ID> ::= 'q' <ID>
<BIT_ID> ::= 'b' <ID>
<REGISTER_ID> ::= 'r' <ID>
<ID> ::= /[0-9]+/
<EXPONENT> ::= /[-+]/? /[0-9]+/ '.' /[0-9]+/
```

The format specifies no maximum id, but in typical usage simulators are expected to allocate
memory proportional to the largest id.
Using ostentatious ids can result in simulations not fitting into memory.

The `REGISTER` instruction can optionally included quoted text specifying a register name.
Quoted text starts and ends with a double-quote character (`"`, ASCII code 34).
A few characters cannot appear directly in quoted text, and must instead be escaped:

- Backslash (`\`, ASCII code 92) must be encoded as `\B`.
- Double-quote (`"`, ASCII code 34) must be encoded as `\Q`.
- Pound (`#`, ASCII code 35) must be encoded as `\P`.
- New line (ASCII code 10) must be encoded as `\n`.
- Line feed (ASCII code 13) must be encoded as `\r`.

```
<QUOTED_TEXT> ::= '"' /[^"#\n\r]*/ '"'
```

A parser should fail if a backslash character appears and is followed a character other than B, Q, P, n, or r.


## Semantics

A kickmix circuit file is executed by executing each of its instructions, one by one, from start to finish.

### Instruction Types

For a complete list of instructions see the adjacent file [kickmix_instruction_set.md](kickmix_instruction_set.md).

Generally speaking, the instructions that can appear can be divided up into three groups:

1. Quantum Operations
2. Classical Operations
3. Control Flow Operations
4. Metadata Operations

A *quantum operation* is a quantum channel to apply to the quantum state of the system.
Available types of quantum operations include reversible classical gates (e.g. the controlled-not gate `CX`),
phasing gates (e.g. the controlled-controlled-Z gate `CCZ`),
and dissipative gates (e.g. the X basis demolition measurement `HMR`).
Measurement operations write to a classical bit in addition to acting on the quantum state.
All quantum operations (including measurements) can be controlled by a classical condition that determines if the operation is skipped or not.

A *classical operation* is an operation applied to classical bits.
These can be irreversible operations, like `BIT_STORE1`.
All classical operations can be controlled by a classical condition that determines if the operation is skipped or not.

*Control flow operations* make global modifications to how operations are being interpreted.
These include `PUSH_CONDITION`, which adds a thing that must be true in order for non-context operations to not be skipped,
and `POP_CONDITION` which removes the most recent pushed condition.

*Metadata operations*, like `APPEND_TO_REGISTER`, are used to define *registers*.
These have no effect on simulations but provide a useful abstraction for reading and
writing multiple qubits/bits as if they were one single value when performing simulations.

### State Space

A simulator executing a kickmix circuit is expected to store three things:

1. **The Qubits**.
   The simulator should track a boolean for each qubit index mentioned by the circuit.
   By construction, no supported operation creates superpositions, so qubit trajectories
   can be followed while only storing the qubits as bits.
2. **The Phase**.
   Phasing operations like `Z` must negate the tracked phase (depending on the
   values of qubits targeted). Fuzz tests of a kickmix circuit must verify that the
   phase behaved as expected for the circuit.
3. **The Bits**.
   The simulator must track a boolean for each bit index mentioned by the circuit.
4. **The Condition Stack**.
   The simulator must track booleans pushed onto and popped off of the condition stack,
   to determine if operations are occurring.

### Fuzz Testing

Kickmix circuits support the operations that they support because they enable efficient fuzz testing.
Generating random inputs, and verifying that they produce the correct output (and phase) is an effective way
to verify the approximate correctness of kickmix circuits.

Quantum operations that correspond to classical reversible operations, like `CX`, are trivially compatible with
fuzz testing.
Adding phasing operations like `Z` doesn't fundamentally change this; the phase is essentially just an extra bit to track and check.
The only operation that isn't *obviously* compatible with fuzz testing is `HMR` (the X basis demolition measurement).
Fuzz tests can simulate this operation by generating a random measurement result and negating the tracked phase if the target qubit is ON
and the measurement result is also ON (this is called the ``phase kickback'' of the operation.)
This correctly simulates the case where the target qubit is a Z-basis function of other qubits
(i.e. measurement based uncomputation).
It doesn't correctly simulate other cases, but it forces fuzz tests of those cases to fail because it's
impossible to fix the phase kickback using the available instructions.

## Examples

### 3-Qubit Increment Circuit

```
# Conveniently define the 3-qubit register
APPEND_TO_REGISTER q0 r0
APPEND_TO_REGISTER q1 r0
APPEND_TO_REGISTER q2 r0
REGISTER r0 "target"

# Perform the increment
CCX q0 q1 q2
CX q0 q1
X q0
```

### 8-qubit Increment Circuit using ancilla qubits

```
# The 8 qubit register to increment
APPEND_TO_REGISTER q0 r0
APPEND_TO_REGISTER q1 r0
APPEND_TO_REGISTER q2 r0
APPEND_TO_REGISTER q3 r0
APPEND_TO_REGISTER q4 r0
APPEND_TO_REGISTER q5 r0
APPEND_TO_REGISTER q6 r0
APPEND_TO_REGISTER q7 r0
REGISTER r0 "target"

# Note: qubits q12 thrugh q16 are used as workspace

# Compute prefix ANDs into q12..q16.
R q12
CCX q0 q1 q12
R q13
CCX q2 q12 q13
R q14
CCX q3 q13 q14
R q15
CCX q4 q14 q15
R q16
CCX q5 q15 q16

# Flips q7 if q0..q6 were all ON.
CCX q16 q6 q7

# Flips q6 if q0..q5 were all ON.
CX q16 q6
# Measurement based uncomputation of q16.
HMR q16 b0
CZ q5 q15 if b0

# Flips q5 if q0..q4 were all ON.
CX q15 q5
# Measurement based uncomputation of q15.
HMR q15 b0
CZ q4 q14 if b0

CX q14 q4
HMR q14 b0
CZ q3 q13 if b0

CX q13 q3
HMR q13 b0
CZ q2 q12 if b0

CX q12 q2
HMR q12 b0
CZ q0 q1 if b0

CX q0 q1
X q0
```
