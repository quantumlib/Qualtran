This document reviews the instructions that can appear in a kickmix circuit.

Note that kickmix circuits are extremely low level and minimalist.
The design goal was not expressiveness, but rather to make kickmix circuits dead simple to analyze.
The instructions form a tiny assembly language that doesn't even have the notion of a "jump" instruction,
so it's not possible to create loops.

# Index

- [Quantum Instructions](#quantum-instructions)
    - [X instruction](#x-instruction)
    - [CX instruction](#cx-instruction)
    - [CCX instruction](#ccx-instruction)
    - [Z instruction](#z-instruction)
    - [CZ instruction](#cz-instruction)
    - [CCZ instruction](#ccz-instruction)
    - [Z_POW instruction](#z-pow-instruction)
    - [NEG instruction](#neg-instruction)
    - [SWAP instruction](#swap-instruction)
    - [R instruction](#r-instruction)
    - [HMR instruction](#hmr-instruction)
- [Classical Instructions](#classical-instructions)
    - [BIT_INVERT instruction](#bit_invert-instruction)
    - [BIT_STORE0 instruction](#bit_store0-instruction)
    - [BIT_STORE1 instruction](#bit_store1-instruction)
- [Control Flow Instructions](#control-flow-instructions)
    - [PUSH_CONDITION instruction](#push_condition-instruction)
    - [POP_CONDITION instruction](#pop-condition-instruction)
- [Metadata Instructions](#metadata-instructions)
    - [REGISTER instruction](#register-instruction)
    - [APPEND_TO_REGISTER instruction](#append_to_register-instruction)
    - [DEBUG_PRINT instruction](#debug_print-instruction)

# Quantum Instructions

## X instruction

The NOT gate.
Bit flips the target single qubit.
Sends |0⟩ to |1⟩ and |1⟩ to |0⟩.

Examples:

```
# Unconditional bit flip.
# Applies a bit flip to qubit q0.
X q0

# Conditional bit flip.
# If bit b3 is OFF, no operation is performed.
# If bit b3 is ON, applies an X gate to qubit q1.
X q1 if b3
```

Arguments:

```
1 Qubit
Optional Condition Bit
```

## CX instruction

The Controlled-NOT gate.
Bit flips the target qubit in the parts of the superposition where
the control qubit is on.
Sends |00⟩ to |00⟩, |01⟩ to |01⟩, |10⟩ to |11⟩, and |11⟩ to |10⟩.

Examples:

```
# Bit flip q1 in the parts of the superposition where q0 is ON.
CX q0 q1

# A classically-conditioned controlled-NOT:
# If bit b7 is OFF, no operation is performed.
# If bit b7 is ON, apply a CX to qubit q2 and qubit q3.
CX q2 q3 if b7
```

Arguments:

```
2 Qubits
Optional Condition Bit
```


## CCX instruction

The Doubly-Controlled-NOT gate.
Bit flips the target qubit in the parts of the superposition where
both control qubits are on.
Sends
|000⟩ to |000⟩,
|001⟩ to |001⟩,
|010⟩ to |010⟩,
|011⟩ to |011⟩,
|100⟩ to |100⟩,
|101⟩ to |101⟩,
|110⟩ to |111⟩, and
|111⟩ to |110⟩.

Examples:

```
# Bit flip q2 in the parts of the superposition
# where q0 is ON and also q1 is ON:
CCX q0 q1 q2

# A classically-conditioned CCX:
# If bit b7 is OFF, no operation is performed.
# If bit b7 is ON, apply a CCX to qubits q2, q3, q4.
CX q2 q3 q4 if b7
```

Arguments:

```
3 Qubits
Optional Condition Bit
```


## Z instruction

The phase flip operation.
Negates the amplitudes of parts of the superposition where the target qubit is ON.
Sends |0⟩ to |0⟩ and |1⟩ to -|1⟩.

Examples:

```
# Phase flip q0.
Z q0

# If the bit b2 is OFF, do nothing.
# If the bit b2 is ON, phase flip q1.
Z q1 if b2
```

Arguments:

```
1 Qubit
Optional Condition Bit
```


## CZ instruction

The controlled phase flip operation.
Negates the amplitudes of parts of the superposition where both target qubits are ON.
Sends
|00⟩ to |00⟩,
|01⟩ to |01⟩,
|10⟩ to |10⟩,  and
|11⟩ to -|11⟩.

Examples:

```
# Controlled phase flip of q0 and q1.
CZ q0 q1

# If the bit b2 is OFF, do nothing.
# If the bit b2 is ON, apply CZ to q1 and q2.
CZ q1 q2 if b2
```

Arguments:

```
2 Qubits
Optional Condition Bit
```


## CCZ instruction

The doubly-controlled phase flip operation.
Negates the amplitudes of parts of the superposition where all three target qubits are ON.
Sends
|000⟩ to |000⟩,
|001⟩ to |001⟩,
|010⟩ to |010⟩,
|011⟩ to |011⟩,
|100⟩ to |100⟩,
|101⟩ to |101⟩,
|110⟩ to |110⟩, and
|111⟩ to -|111⟩.

Examples:

```
# Negate the amplitudes of states where q0 and q1 and q2 are ON.
CCZ q0 q1 q2

# If the bit b3 is OFF, do nothing.
# If the bit b3 is ON, apply CCZ to q0 and q1 and q2.
CCZ q0 q1 q2 if b3
```

Arguments:

```
3 Qubits
Optional Condition Bit
```


## Z_POW instruction

Fine-grained single-qubit phasing.
Phases |1> by exp(i π t) where t is the power that the Z gate is raised to.

Parsers and simulators *may* round the exponent to the nearest multiple of 2⁻¹²⁷.
Parsers and simulators **MUST NOT** perturb the given fraction by more than a 2⁻¹²⁸'th of a turn.
In particular, note that this precision requirement implies it isn't appropriate to store the exponent as a double precision floating point number.

Examples:

```
# Equivalent to Z q0 (a 180 degree rotation):
Z_POW q0 1.0

# Apply an "S" gate (square root of Z; 90 degree rotation) to q1:
Z_POW q1 0.5

# ... but only if b1 is true:
Z_POW q1 0.5 if b1

# Apply an inverse "S" gate (-90 degree rotation) to q2:
Z_POW q2 -0.5

# Other things equivalent to an inverse "S" gate on q2:
Z_POW q2 1.5
Z_POW q2 -2.5

# Phase q3 by exactly one 2⁻¹²⁸'th of a turn, plus 90 degrees:
Z_POW q3 0.5000000000000000000000000000000000000058774717541114375398436826861112283890933277838604376075437585313920862972736358642578125

# Phase q3 by approximately one 2⁻¹²⁸'th of a turn (off by less than one 2⁻¹²⁹'th of a turn):
Z_POW q3 0.00000000000000000000000000000000000000588
```

Arguments:

```
1 Qubit
1 Exponent
Optional Condition Bit
```


## NEG instruction

The global phase negation instruction.
Negates the amplitude of all states.

This instruction is a strange inclusion because it would never be needed
on a real quantum computer (because negating all amplitudes has no
observational effect).
The reason this instruction is present is to make fuzz testing of kickmix
circuits possible.
In particular, when a kickmix circuit is being fuzz tested by trying random
classical trajectories, each sampled trajectory will have an associated phase
(-1 or +1). When the phase ends up as -1, this is  an indication that phase
kickback from an HMR instruction was incorrectly handled. If that kickback
can be fixed by a NEG instruction then it is actually benign, but it's hard
to distinguish this case from the fatal phases being fixed by Z/CZ/CCZ instructions.
So kickmix  circuits are expected to go the extra mile and make themselves easy to
verify, by using NEG.

Examples:

```
# Negate the global phase
NEG

# Negate the global phase if the bit b1 is ON
NEG if b1

# Prepare a qubit in the 1 state, then clear it with
# measurement-based uncomputation.
R 0
X 0
HMR q0 b0
NEG if b0  # Corrects phase kickback from the HMR.
```

Arguments:

```
Optional Condition Bit
```    

## SWAP instruction

Exchanges two qubits.
Sends
|00⟩ to |00⟩,
|01⟩ to |10⟩,
|10⟩ to |01⟩, and
|11⟩ to |11⟩.

Examples:

```
# Swap the values of qubit q0 and qubit q1.
SWAP q0 q1

# Do nothing if bit b2 is OFF.
# If b2 is ON, swap the values of q0 and q1.
SWAP q0 q1 if b2

# A decomposition of `SWAP q0 q1 if b2` into CX gates.
CX q0 q1
CX q1 q0 if b2
CX q0 q1
```

Arguments:

```
2 Qubits
Optional Condition Bit
```


## R instruction

The reset gate.
Discards the target qubit and replaces it with one in the |0⟩ state.

Beware that, if the target qubit is in the |1> state, this operation randomizes the phase.
So, ironically, you only want to apply this gate to qubits that are already in the |0⟩ state.
It's perhaps more of an assertion that the qubit *should* be 0.
Alternatively, you can think of it as a decoration that makes circuit diagrams look nicer.

Examples:

```
# Reset qubit q0 to |0>.
R q0

# If bit b2 is OFF, do nothing.
# If bit b2 is ON, reset q1 to |0>.
R q1 if b2

# Randomize the global phase for no reason, ruining the
# ability to easily fuzz test the circuit.
R q0
X q0
R q0
```

Arguments:

```
1 Qubit
Optional Condition Bit
```

## HMR instruction

The **H**adamard then **M**easure then **R**eset instruction (pronounced "hammer").
Measures a qubit in the X basis, stores the result in a bit,
and resets the qubit to the |0> state.

When a qubit is a Z-basis function of other qubits, the effect of this operation is
to negate the phase of states where the qubit was |1> if the measurement returns TRUE
(this is the "phase kickback" from the operation).
The circuit is expected to correct this phase kickback using Z, CZ, CCZ, and NEG
instructions.

In a kickmix circuit, it is a mistake to apply the HMR instruction to a qubit
that isn't a Z-basis function of other qubits.
In an actual quantum computer, this would create complex interference effects.
In a kickmix simulator, this case will be incorrectly simulated by returning
a random result from the measurement and negating  the phases of trajectories
where the qubit was |1> if the measurement result is true.
Although this is incorrect, it forces phase verification tests to fail probabilistically.
In this way, incorrectly used HMR operations will fail loudly instead of silently.
They can be detected by fuzz testing.

Examples:

```
# Measures q3 in the X basis, storing the result in b0.
# Also resets q3 to the 0 state:
HMR q3 b0

# If b2 is OFF, does nothing (including not writing to b1).
# If b2 is ON, performs HMR q3 b1. 
HMR q3 b1 if b2

# Initializes q2 to be q0 AND q1, then uncomputing it with measurement
# based uncomputation using HMR + phase feedback.
R q2
CCX q0 q1 q2
HMR q1 b0
CZ q0 q1 if b0
```

Arguments:

```
1 Qubit
1 Bit (output location)
Optional Condition Bit
```

# Classical Instructions

## BIT_INVERT instruction

The classical NOT gate.
Flips a target bit.

Examples:

```
# Flips the value of b0
BIT_INVERT b0

# Flips the value of b1 if b2 is ON.
# In other words, performs b1 ^= b2.
BIT_INVERT b1 if b2

# Set b3 to (b4 ^ b5) & b6.
BIT_INVERT b4 if b5
BIT_INVERT b6
BIT_INVERT b4
BIT_STORE1 b3
BIT_STORE0 b3 if b6
BIT_STORE0 b3 if b4
BIT_INVERT b3
BIT_INVERT b4
BIT_INVERT b6
BIT_INVERT b4 if b5
```

Arguments:

```
1 Bit
Optional Condition Bit
```

## BIT_STORE0 instruction

Sets the target bit to OFF.

Examples:

```
# Clear b0.
BIT_STORE0 b0

# Clear b1 if b2 is ON.
BIT_STORE0 b1 if b2

# Set b3 to (b4 ^ b5) & b6.
BIT_INVERT b4 if b5
BIT_INVERT b6
BIT_INVERT b4
BIT_STORE1 b3
BIT_STORE0 b3 if b6
BIT_STORE0 b3 if b4
BIT_INVERT b3
BIT_INVERT b4
BIT_INVERT b6
BIT_INVERT b4 if b5
```

Arguments:

```
1 Bit
Optional Condition Bit
```

## BIT_STORE1 instruction

Sets the target bit to ON.

Examples:

```
# Set b0.
BIT_STORE1 b0

# Set b1 if b2 is ON.
BIT_STORE1 b1 if b2

# Set b3 to (b4 ^ b5) & b6.
BIT_INVERT b4 if b5
BIT_INVERT b6
BIT_INVERT b4
BIT_STORE1 b3
BIT_STORE0 b3 if b6
BIT_STORE0 b3 if b4
BIT_INVERT b3
BIT_INVERT b4
BIT_INVERT b6
BIT_INVERT b4 if b5
```

Arguments:

```
1 Bit
Optional Condition Bit
```


# Control Flow Instructions

## PUSH_CONDITION instruction

Pushes a bit's value onto the condition stack.
Basically: the equivalent of `if (condition) {` in C.

Most instructions won't occur unless all values on the condition stack are TRUE.
There are four exceptions: `PUSH_CONDITION`, `POP_CONDITION`, `REGISTER`, and `APPEND_TO_REGISTER`.
`POP_CONDITION` has to ignore the condition stack because otherwise it would be impossible
to clear the stack.
`PUSH_CONDITION` has to ignore the condition stack because otherwise it would desync
from `POP_CONDITION`.
`REGISTER` and `APPEND_TO_REGISTER` ignore the condition stack
so that it's not necessary to run a simulation in order to identify the shape of the
registers in a circuit.

Note: it's the bit's *current* value that is pushed onto the stack.
Writing to a bit after pushing it won't change the state of the condition stack.

Examples:

```
# Perform `HMR q0 b3` if b0 and b1 and b2 are all ON
PUSH_CONDITION if b0
PUSH_CONDITION if b1
HMR q0 b3 if b2
POP_CONDITION
POP_CONDITION
```

Arguments:

```
1 Condition Bit
```


## POP_CONDITION instruction

Pops a bit off of the condition stack.
Basically: the equivalent of `}` at the end of an `if` block in C.

If the condition stack is empty, this operation has no effect.

Examples:

```
# Performs `HMR q0 b3` if b0 and b1 and b2 are all ON
PUSH_CONDITION if b0
PUSH_CONDITION if b1
HMR q0 b3 if b2
POP_CONDITION
POP_CONDITION
```

Arguments:

```
None
```


# Metadata Instructions

## REGISTER instruction

The `REGISTER` instruction declares the existence of a register, and optionally names the register.
Registers can also be inferred from instructions like `APPEND_TO_REGISTER`.

Registers have no effect on simulation, but make it easier to refer to parts of the statespace.
For example, a user may perform a simulation by initializing the qubits of register #1 to represent
a desired integer value and then check that the contents of register #2 have been initialized correctly
by the action of the circuit (rather than specifying these things a single qubit/bit at a time).

The `REGISTER` instruction is mainly useful for ensuring a register will still exist in corner cases
where the register could be empty (e.g. circuit generation code asked to produce a zero-qubit incrementer).

**This operation ignores conditions from `PUSH_CONDITION`**.

Examples:

```
# Declare an unnamed empty register r2 (also ensures r0 and r1 exist).
REGISTER r2

# Declare a four bit register with b0 as the least significant bit.
APPEND_TO_REGISTER b0 r0
APPEND_TO_REGISTER b1 r0
APPEND_TO_REGISTER b2 r0
APPEND_TO_REGISTER b3 r0

# Declare a three qubit register with q9 as the least significant bit.
APPEND_TO_REGISTER q9 r1
APPEND_TO_REGISTER q6 r1
APPEND_TO_REGISTER q10 r1
# Gives a name to register r1
REGISTER r1 "example_register"
```

Arguments:

```
1 register
optional name
```


## APPEND_TO_REGISTER instruction

Declares that a bit or qubit is part of a register.
Bits/qubits are appended to register in order of increasing significance.

**This operation ignores conditions from `PUSH_CONDITION`**.

Examples:

```
# Declare a four bit register with b0 as the least significant bit.
APPEND_TO_REGISTER b0 r0
APPEND_TO_REGISTER b1 r0
APPEND_TO_REGISTER b2 r0
APPEND_TO_REGISTER b3 r0

# Declare a three qubit register with q9 as the least significant bit.
APPEND_TO_REGISTER q9 r1
APPEND_TO_REGISTER q6 r1
APPEND_TO_REGISTER q10 r1

# Declare an empty register r2.
REGISTER r2
```

Arguments:

```
1 qubit|bit
1 register
```


## DEBUG_PRINT instruction

This instruction is a request to please print some information about simulator state.
Simulators are free to interpret this instruction however they want, including ignoring it.

```
optional qubit
optional bit
optional register
```

Examples:

```
# Please print something related to qubit q0.
# Perhaps its value from the various trajectories being simulated?
DEBUG_PRINT q0

# Please print something related to bit b0's tracked value.
# Perhaps its value from the various trajectories being simulated?
DEBUG_PRINT b0

# Please print something related to register 0's value.
# Perhaps the values of its qubits and bits, from the various
# trajectories being simulated, interpreted as a 2s complement
# little endian integer?
DEBUG_PRINT r0

# Please print something about nothing.
# ...maybe print a newline character...?
DEBUG_PRINT
```