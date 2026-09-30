# The Kickmix Binary Circuit File Format (.kmb)

This document describes the kickmix binary circuit file format (.kmb).
A .kmb file stores broadly similar information to [a .kmx file](kickmix_file_format.md),
but in a binary format optimized for performance rather than for human readability.

## Index

- [File Layout](#File-Layout)
    - [Header](#Header)
    - [Packets](#Packets)
    - [Packet Types](#Packet-Types)
        - [`KMB_REGISTER_TARGETS` packet](#KMB_REGISTER_TARGETS-packet)
        - [`KMB_OPERATION_TYPES` packet](#KMB_OPERATION_TYPES-packet)
        - [`KMB_QUBIT#_DATA_FOR_#_TARGET_QUBIT_GATES` packets (multiple)](#KMB_QUBITX_DATA_FOR_X_TARGET_QUBIT_GATES-packets)
        - [`KMB_BIT0_DATA_FOR_1_TARGET_BIT_GATES` packet](#KMB_BIT0_DATA_FOR_1_TARGET_BIT_GATES-packet)
        - [`KMB_BIT0_DATA_FOR_1_CONDITION_BIT_GATES` packet](#KMB_BIT0_DATA_FOR_1_CONDITION_BIT_GATES-packet)
        - [`KMB_ANGLE_DATA_FOR_ROTATION_GATES` packet](#KMB_ANGLE_DATA_FOR_ROTATION_GATES-packet)
  - [Operation Types](#Operation_Types)
  - [SOA Layout of Operation Target Data](#SOA-Layout-of-Operation-Target-Data)
- [Validation](#Validation)
- [Example Files](#Example-Files)


## File Layout

Every kmb file starts with a fixed size header.
After the header, all data comes in "packets".
A packet starts by noting the length and type of its payload, and then finishes with its payload.


### Header

The header begins with magic bytes that can be used to identify an unknown file as a kmb file.
These bytes should be `d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf`.
The header then lists a version (which currently should be 2),
and various summary statistics about the circuit such as the number of qubits and the number of operations.

```
// Note: all values are little endian
struct Header {
    char magic[16];       // Expect: d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf
    uint32_t version;     // Expect: 2
    uint32_t num_qubits;  // Largest used qubit id plus 1.
    uint32_t num_bits;    // Largest used bit id plus 1.
    uint32_t num_registers;
    uint64_t num_ops;
    uint64_t num_ops_with_3_target_qubits;
    uint64_t num_ops_with_2_target_qubits;
    uint64_t num_ops_with_1_target_qubits;
    uint64_t num_ops_with_1_target_bit;
    uint64_t num_ops_with_1_condition_bit;
    uint64_t num_ops_with_angle;
}
```

### Packets

After the header, all data comes in "packets".
A packet starts by noting the length and type of its payload, and then finishes with its payload:

```
struct Packet {
    uint64_t payload_type;
    uint64_t payload_length;
    char payload[payload_length];
}
```

### Packet Types

The following payload types are specified:

```
enum payload_type_u64 : uint64_t {
   KMB_REGISTER_TARGETS                     = 0,
   KMB_OPERATION_TYPES                      = 1,
   KMB_QUBIT0_DATA_FOR_3_TARGET_QUBIT_GATES = 2,
   KMB_QUBIT1_DATA_FOR_3_TARGET_QUBIT_GATES = 3,
   KMB_QUBIT2_DATA_FOR_3_TARGET_QUBIT_GATES = 4,
   KMB_QUBIT0_DATA_FOR_2_TARGET_QUBIT_GATES = 5,
   KMB_QUBIT1_DATA_FOR_2_TARGET_QUBIT_GATES = 6,
   KMB_QUBIT0_DATA_FOR_1_TARGET_QUBIT_GATES = 7,
   KMB_BIT0_DATA_FOR_1_TARGET_BIT_GATES     = 8,
   KMB_BIT0_DATA_FOR_1_CONDITION_BIT_GATES  = 9,
   KMB_ANGLE_DATA_FOR_ROTATION_GATES        = 10,
}
```

#### KMB_REGISTER_TARGETS packet

```
// Note: all values are little endian
struct Packet_REGISTER_TARGETS {
   uint64_t payload_type = (uint64_t)KMB_REGISTER_TARGETS;
   uint64_t payload_length;
   uint32_t name_length;
   uint32_t register_length;
   char name_payload[name_length];
   uint32_t register_payload[register_length];
}
```

This packet lists the name, bits, and/or qubits of a register.
The number of characters in the name is `name_length`.
The number of bit ids and/or qubit ids in the register is `register_length`.
Each bit-or-qubit id is specified by a big endian uint32_t.
The high bit of the uint32_t is set if the id is a qubit, rather than a bit.
For example, 0x3 is bit id 3 and 0x80000002 is qubit id 2.
Beware that registers may alias (i.e. the same qubit or bit can appear in multiple registers).

#### KMB_OPERATION_TYPES packet

```
// Note: all values are little endian
struct Packet_KMB_OPERATION_TYPES {
   uint64_t payload_type = (uint64_t)KMB_OPERATION_TYPES;
   uint64_t payload_length;
   OperationType payload[payload_length];
}
```

This packet lists the types of operations performed by the circuit, in the order that they are performed.

#### KMB_QUBITX_DATA_FOR_X_TARGET_QUBIT_GATES packets

The packet types `KMB_QUBIT0_DATA_FOR_3_TARGET_QUBIT_GATES`,
`KMB_QUBIT1_DATA_FOR_3_TARGET_QUBIT_GATES`,
`KMB_QUBIT2_DATA_FOR_3_TARGET_QUBIT_GATES`,
`KMB_QUBIT0_DATA_FOR_2_TARGET_QUBIT_GATES`,
`KMB_QUBIT1_DATA_FOR_2_TARGET_QUBIT_GATES`, and
`KMB_QUBIT0_DATA_FOR_1_TARGET_QUBIT_GATES`
all have identical structure:
```
// Note: all values are little endian
struct packet_KMB_QUBIT_DATA {
   uint64_t payload_type;
   uint64_t payload_length;
   uint32_t payload[payload_length / 4];
}
```

Each `uint32_t` in the payload is a qubit identifier.

#### KMB_BIT0_DATA_FOR_1_TARGET_BIT_GATES packet

```
// Note: all values are little endian
struct packet_KMB_BIT0_DATA_FOR_1_TARGET_BIT_GATES {
   payload_type_u64 payload_type;
   uint64_t payload_length;
   uint32_t payload[payload_length / 4];
}
```

Each `uint32_t` in the payload is a bit identifier.


#### KMB_BIT0_DATA_FOR_1_CONDITION_BIT_GATES packet

```
// Note: all values are little endian
struct packet_KMB_BIT0_DATA_FOR_1_CONDITION_BIT_GATES {
   payload_type_u64 payload_type;
   uint64_t payload_length;
   uint32_t payload[payload_length / 4];
}
```

Each `uint32_t` in the payload is a bit identifier.

#### KMB_ANGLE_DATA_FOR_ROTATION_GATES packet

Each pair of `uint64_t`s in the payload is an angle, rounded to the
nearest multiple of 2**-128.

```
struct FixedPrecisionAngle128 {
    // The angle is equal to (words[0] / 2**128 + words[1] / 2**64) half turns.
    // Note: all values are little endian
    uint64_t words[2];
}
struct packet_KMB_ANGLE_DATA_FOR_ROTATION_GATES {
   payload_type_u64 payload_type;
   uint64_t payload_length;
   FixedPrecisionAngle128 payload[payload_length / 16];
}
```


### Operation Types

In the operation type data, each operation type is indicated by a single byte.
Here is an enum of specified operation types and their associated byte:

```
const uint8_t OP_HAS_1Q = 0x10;
const uint8_t OP_HAS_2Q = 0x20;
const uint8_t OP_HAS_3Q = OP_HAS_2Q | OP_HAS_1Q;
const uint8_t OP_HAS_1B = 0x40;
const uint8_t OP_HAS_COND = 0x80;
enum OperationType : uint8_t {
    NEG                  = 0x0,
    NEG_IF               = 0x0 | OP_HAS_COND,
    POP_CONDITION        = 0x1,
    PUSH_CONDITION       = 0x1 | OP_HAS_COND,

    Z                    = 0x0 | OP_HAS_1Q,
    Z_IF                 = 0x0 | OP_HAS_1Q | OP_HAS_COND,
    X                    = 0x1 | OP_HAS_1Q,
    X_IF                 = 0x1 | OP_HAS_1Q | OP_HAS_COND,
    R                    = 0x2 | OP_HAS_1Q,
    R_IF                 = 0x2 | OP_HAS_1Q | OP_HAS_COND,
    Z_POW                = 0x3 | OP_HAS_1Q,
    Z_POW_IF             = 0x3 | OP_HAS_1Q | OP_HAS_COND,

    HMR                  = 0x0 | OP_HAS_1Q | OP_HAS_1B,
    HMR_IF               = 0x0 | OP_HAS_1Q | OP_HAS_1B | OP_HAS_COND,

    CZ                   = 0x0 | OP_HAS_2Q,
    CZ_IF                = 0x0 | OP_HAS_2Q | OP_HAS_COND,
    CX                   = 0x1 | OP_HAS_2Q,
    CX_IF                = 0x1 | OP_HAS_2Q | OP_HAS_COND,
    SWAP                 = 0x2 | OP_HAS_2Q,
    SWAP_IF              = 0x2 | OP_HAS_2Q | OP_HAS_COND,

    CCZ                  = 0x0 | OP_HAS_3Q,
    CCZ_IF               = 0x0 | OP_HAS_3Q | OP_HAS_COND,
    CCX                  = 0x1 | OP_HAS_3Q,
    CCX_IF               = 0x1 | OP_HAS_3Q | OP_HAS_COND,

    BIT_STORE0           = 0x0 | OP_HAS_1B,
    BIT_STORE0_IF        = 0x0 | OP_HAS_1B | OP_HAS_COND,
    BIT_STORE1           = 0x1 | OP_HAS_1B,
    BIT_STORE1_IF        = 0x1 | OP_HAS_1B | OP_HAS_COND,
    BIT_INVERT           = 0x2 | OP_HAS_1B,
    BIT_INVERT_IF        = 0x2 | OP_HAS_1B | OP_HAS_COND,

    DEBUG_PRINT_EMPTY    = 0x7,
    DEBUG_PRINT_EMPTY_IF = 0x7 | OP_HAS_COND,
    DEBUG_PRINT_C        = 0x7 | OP_HAS_1B,
    DEBUG_PRINT_C_IF     = 0x7 | OP_HAS_1B | OP_HAS_COND,
    DEBUG_PRINT_Q        = 0x7 | OP_HAS_1Q,
    DEBUG_PRINT_Q_IF     = 0x7 | OP_HAS_1Q | OP_HAS_COND,
};
```

See [kickmix_instruction_set.md](kickmix_instruction_set.md) for more information on each kind of operation.
Note that specific bit positions are used to indicate what kinds of data the operation acts on.
This makes it possible to efficiently compute statistics, such as the number of 3 qubit gates,
using only the operation type data.

### SOA Layout of Operation Target Data

The targets for operations are stored in struct-of-array style (SOA), rather than array-of-struct style.
For example, a 3 qubit operation isn't stored in the file as an operation type identifier
followed by three qubit identifiers.
Instead, each qubit identifier goes into a separate array
(the first-qubit-target-of-3-qubit-gate array, the second-qubit-target-of-3-qubit-gate array, and the third-qubit-target-of-3-qubit-gate array).
This allows data to be read in bulk, and also for certain important properties to be verified in bulk (such as the requirement that a three qubit gate must act on three different qubits).

The main downside of this layout is that it doesn't support constant time random access into the operations of the circuit,
because different operations pull data from different arrays.

For concreteness, here is a partial example of consuming struct-of-array kmb data in order to print out a [human readable .kmx file](kickmix_file_format.md):

```
struct KmbStructOfArrays {
    size_t num_ops;                 // from header
    OpType *type;                   // from KMB_OPERATION_TYPES packet
    uint32_t *qqq0;                 // from KMB_QUBIT0_DATA_FOR_3_TARGET_QUBIT_GATES packet
    uint32_t *qqq1;                 // from KMB_QUBIT0_DATA_FOR_3_TARGET_QUBIT_GATES packet
    uint32_t *qqq2;                 // from KMB_QUBIT0_DATA_FOR_3_TARGET_QUBIT_GATES packet
    uint32_t *qq0;                  // from KMB_QUBIT0_DATA_FOR_2_TARGET_QUBIT_GATES packet
    uint32_t *qq1;                  // from KMB_QUBIT1_DATA_FOR_2_TARGET_QUBIT_GATES packet
    uint32_t *q0;                   // from KMB_QUBIT0_DATA_FOR_1_TARGET_QUBIT_GATES packet
    uint32_t *b0;                   // from KMB_BIT0_DATA_FOR_1_TARGET_BIT_GATES packet
    uint32_t *bc;                   // from KMB_BIT0_DATA_FOR_1_CONDITION_BIT_GATES packet
    FixedPrecisionAngle128 *angles; // from KMB_ANGLE_DATA_FOR_ROTATION_GATES packet
}

void print_kmb_as_kmx(const KmbStructOfArrays *circuit) {
    const uint32_t *qqq0 = circuit->qqq0;
    const uint32_t *qqq1 = circuit->qqq1;
    const uint32_t *qqq2 = circuit->qqq2;
    const uint32_t *qq0 = circuit->qq0;
    const uint32_t *qq1 = circuit->qq1;
    const uint32_t *q0 = circuit->q0;
    const uint32_t *b0 = circuit->b0;
    const uint32_t *bc = circuit->bc;
    const FixedPrecisionAngle128 *angles = circuit->angles;
    for (size_t k = 0; k < circuit->num_ops; k++) {
        switch (type[k]) {
        case OpType::X:      printf("X q%u\n", *q0++);                               break;
        case OpType::X_IF:   printf("X q%u if b%u\n", *q0++, *b0++);                 break;
        case OpType::HMR:    printf("HMR q%u b%u\n", *q0++, *bc++);                  break;
        case OpType::HMR_IF: printf("HMR q%u b%u if b%u\n", *q0++, *bc++, *b0++);    break;
        case OpType::CCX:    printf("CCX q%u q%u q%u\n", *qqq0++, *qqq1++, *qqq2++); break;
        // ... and so forth for all other operation types ...
        }
    }
}
```

## Validation

1. **Validate packet order**. For a version 2 .kmb file, the payload packets must appear in exactly the following order:
   - A `KMB_REGISTER_TARGETS` packet for each register (as indicated by `header.num_registers`).
   - Exactly one `KMB_OPERATION_TYPES` packet with a payload length `header.num_operations`.
   - Exactly one `KMB_QUBIT0_DATA_FOR_3_TARGET_QUBIT_GATES` packet.
   - Exactly one `KMB_QUBIT1_DATA_FOR_3_TARGET_QUBIT_GATES` packet.
   - Exactly one `KMB_QUBIT2_DATA_FOR_3_TARGET_QUBIT_GATES` packet.
   - Exactly one `KMB_QUBIT0_DATA_FOR_2_TARGET_QUBIT_GATES` packet.
   - Exactly one `KMB_QUBIT1_DATA_FOR_2_TARGET_QUBIT_GATES` packet.
   - Exactly one `KMB_QUBIT0_DATA_FOR_1_TARGET_QUBIT_GATES` packet.
   - Exactly one `KMB_BIT0_DATA_FOR_1_TARGET_BIT_GATES` packet.
   - Exactly one `KMB_BIT0_DATA_FOR_1_CONDITION_BIT_GATES` packet.
   - Exactly one `KMB_ANGLE_DATA_FOR_ROTATION_GATES` packet.
   
   A parser MUST verify that the packets come in this exact order, and fail with an error if they didn't.
    
2. **Validate payload lengths**.
    For most packets, the payload length of the packet is a function of a value listed in the header.
    For example, the payload length of `KMB_QUBIT0_DATA_FOR_3_TARGET_QUBIT_GATES` is `4 * header.num_ops_with_3_target_qubits`.
    A parser MUST verify that the payload lengths are consistent with the values in the header, and fail with an error if they are inconsistent.

3. **Validate header.num_qubits**.
    The value of `num_qubits` in the header MUST be 1 more than the largest qubit id that is used anywhere the circuit (either in an operation target or in a register).
    If the circuit uses no qubit ids, the value must be 0.
    Furthermore, `num_qubits` must be at most half of a billion (num_qubits <= 500000000).
   A parser MUST verify that the num_qubits value was correct, and fail with an error if it wasn't.

4. **Validate header.num_bits**.
   The value of `num_bits` in the header MUST be 1 more than the largest bit id that is used anywhere the circuit (either in an operation target or in a register).
   If the circuit uses no bit ids, the value must be 0.
   Furthermore, `num_bits` must be at most half of a billion (num_bits <= 500000000).
   A parser MUST verify that the num_bits value was correct, and fail with an error if it wasn't.

5. **Validate header.num_ops_with_3_target_qubits**.
   The value of `num_ops_with_3_target_qubits` in the header MUST be equal to the number of operations
   in the circuit that target 3 qubits (CCX, CCZ, CCX_IF, and CCZ_IF).

6. **Validate header.num_ops_with_2_target_qubits**.
   The value of `num_ops_with_2_target_qubits` in the header MUST be equal to the number of operations
   in the circuit that target 2 qubits (CX, CX_IF, CZ, etc).

7. **Validate header.num_ops_with_1_target_qubit**.
   The value of `num_ops_with_1_target_qubit` in the header MUST be equal to the number of operations
   in the circuit that target 1 qubit (X, X_IF, R, R_IF, HMR, etc).

8. **Validate header.num_ops_with_1_target_bit**.
   The value of `num_ops_with_1_target_bit` in the header MUST be equal to the number of operations
   in the circuit that target 1 bit (BIT_STORE0, BIT_STORE0_IF, HMR, etc).

9. **Validate header.num_ops_with_1_condition_bit**.
   The value of `num_ops_with_1_condition_bit` in the header MUST be equal to the number of operations
   in the circuit that have 1 condition bit (X_IF, CX_IF, CCX_IF, BIT_STORE0_IF, PUSH_CONDITION, etc).

10. **Validate header.num_angles**.
   The value of `num_angles` in the header MUST be equal to the number of operations
   in the circuit that have a rotation angle (Z_POW and Z_POW_IF).

11. **Validate disjoint 2 qubit gate targets**.
   Two qubit operations get their first and second qubit from `KMB_QUBIT0_DATA_FOR_2_TARGET_QUBIT_GATES` and `KMB_QUBIT1_DATA_FOR_2_TARGET_QUBIT_GATES` respectively.
   These two qubits must never be identical to each other.
   A parser MUST verify that the qubit targets are never identical, and fail with an error if any were.

12. **Validate disjoint 3 qubit gate targets**.
   Three qubit operations get their first, second, and third qubit from `KMB_QUBIT0_DATA_FOR_3_TARGET_QUBIT_GATES`, `KMB_QUBIT1_DATA_FOR_3_TARGET_QUBIT_GATES`, and `KMB_QUBIT2_DATA_FOR_3_TARGET_QUBIT_GATES` respectively.
   These three qubits must all be different from each other.
   A parser MUST verify that the three qubit targets are disjoint, and fail with an error if they weren't.


## Example Files

These examples can be reproduced by running

```bash
cat ${KMX_FILE_PATH} \
    | kickmix kmx2kmb \
    | hexdump -v -e '32/1 "%02x " "\n"'
````

### The Empty Circuit

Hex dump of .kmb data for performing no operations to no registers:

```
d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf 02 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 02 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 03 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 04 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 05 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 06 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 07 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 09 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 0a 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00                        
```

Equivalent .kmx data:

```
# (empty)
```

### Three qubit incrementer:

Hex dump of .kmb data for performing register0 += 1:

```
d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf 02 00 00 00 03 00 00 00 00 00 00 00 01 00 00 00
03 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
14 00 00 00 00 00 00 00 00 00 00 00 03 00 00 00 00 00 00 80 01 00 00 80 02 00 00 80 01 00 00 00
00 00 00 00 03 00 00 00 00 00 00 00 31 21 11 02 00 00 00 00 00 00 00 04 00 00 00 00 00 00 00 00
00 00 00 03 00 00 00 00 00 00 00 04 00 00 00 00 00 00 00 01 00 00 00 04 00 00 00 00 00 00 00 04
00 00 00 00 00 00 00 02 00 00 00 05 00 00 00 00 00 00 00 04 00 00 00 00 00 00 00 00 00 00 00 06
00 00 00 00 00 00 00 04 00 00 00 00 00 00 00 01 00 00 00 07 00 00 00 00 00 00 00 04 00 00 00 00
00 00 00 00 00 00 00 08 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 09 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 0a 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00                           
```

Equivalent .kmx data:

```
APPEND_TO_REGISTER q0 r0
APPEND_TO_REGISTER q1 r0
APPEND_TO_REGISTER q2 r0
CCX q0 q1 q2
CX q0 q1
X q0
```


## 8 qubit adder (classical offset, 2 clean ancillae, 6 dirty ancillae):

Hex dump of .kmb data for performing register0 += register1:

```
d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf 02 00 00 00 10 00 00 00 10 00 00 00 02 00 00 00
93 00 00 00 00 00 00 00 10 00 00 00 00 00 00 00 19 00 00 00 00 00 00 00 62 00 00 00 00 00 00 00
0b 00 00 00 00 00 00 00 4e 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
28 00 00 00 00 00 00 00 00 00 00 00 08 00 00 00 00 00 00 80 01 00 00 80 02 00 00 80 03 00 00 80
04 00 00 80 05 00 00 80 06 00 00 80 07 00 00 80 00 00 00 00 00 00 00 00 28 00 00 00 00 00 00 00
00 00 00 00 08 00 00 00 00 00 00 00 01 00 00 00 02 00 00 00 03 00 00 00 04 00 00 00 05 00 00 00
06 00 00 00 07 00 00 00 01 00 00 00 00 00 00 00 93 00 00 00 00 00 00 00 91 91 91 91 91 91 91 91
21 21 21 21 21 21 12 a1 91 12 22 91 31 91 21 50 c2 91 12 22 91 31 91 21 50 c2 91 12 22 91 31 91
21 50 c2 91 12 22 91 31 91 21 50 c2 91 12 91 31 91 21 91 91 31 91 21 50 81 90 90 80 20 10 01 91
50 c2 21 21 21 21 21 21 11 11 11 11 11 11 11 11 90 90 90 90 90 91 91 91 91 91 91 31 31 31 31 31
91 91 91 91 91 91 91 91 91 91 91 91 a1 31 31 31 31 31 91 91 91 91 91 91 91 91 91 91 91 91 90 90
90 90 90 11 11 11 11 11 11 11 11 02 00 00 00 00 00 00 00 40 00 00 00 00 00 00 00 01 00 00 00 02
00 00 00 03 00 00 00 04 00 00 00 05 00 00 00 06 00 00 00 0e 00 00 00 0d 00 00 00 0c 00 00 00 0b
00 00 00 0a 00 00 00 0a 00 00 00 0b 00 00 00 0c 00 00 00 0d 00 00 00 0e 00 00 00 03 00 00 00 00
00 00 00 40 00 00 00 00 00 00 00 09 00 00 00 09 00 00 00 09 00 00 00 09 00 00 00 08 00 00 00 09
00 00 00 05 00 00 00 04 00 00 00 03 00 00 00 02 00 00 00 01 00 00 00 01 00 00 00 02 00 00 00 03
00 00 00 04 00 00 00 05 00 00 00 04 00 00 00 00 00 00 00 40 00 00 00 00 00 00 00 08 00 00 00 08
00 00 00 08 00 00 00 08 00 00 00 09 00 00 00 07 00 00 00 0f 00 00 00 0e 00 00 00 0d 00 00 00 0c
00 00 00 0b 00 00 00 0b 00 00 00 0c 00 00 00 0d 00 00 00 0e 00 00 00 0f 00 00 00 05 00 00 00 00
00 00 00 64 00 00 00 00 00 00 00 01 00 00 00 02 00 00 00 03 00 00 00 04 00 00 00 05 00 00 00 06
00 00 00 00 00 00 00 08 00 00 00 09 00 00 00 08 00 00 00 09 00 00 00 08 00 00 00 09 00 00 00 08
00 00 00 09 00 00 00 08 00 00 00 09 00 00 00 05 00 00 00 01 00 00 00 02 00 00 00 03 00 00 00 04
00 00 00 05 00 00 00 06 00 00 00 00 00 00 00 06 00 00 00 00 00 00 00 64 00 00 00 00 00 00 00 0a
00 00 00 0b 00 00 00 0c 00 00 00 0d 00 00 00 0e 00 00 00 0f 00 00 00 08 00 00 00 09 00 00 00 01
00 00 00 09 00 00 00 02 00 00 00 09 00 00 00 03 00 00 00 09 00 00 00 04 00 00 00 05 00 00 00 06
00 00 00 08 00 00 00 0a 00 00 00 0b 00 00 00 0c 00 00 00 0d 00 00 00 0e 00 00 00 0f 00 00 00 0a
00 00 00 07 00 00 00 00 00 00 00 88 01 00 00 00 00 00 00 00 00 00 00 01 00 00 00 02 00 00 00 03
00 00 00 04 00 00 00 05 00 00 00 06 00 00 00 07 00 00 00 08 00 00 00 08 00 00 00 09 00 00 00 09
00 00 00 09 00 00 00 09 00 00 00 08 00 00 00 09 00 00 00 09 00 00 00 09 00 00 00 09 00 00 00 08
00 00 00 09 00 00 00 09 00 00 00 09 00 00 00 09 00 00 00 08 00 00 00 09 00 00 00 09 00 00 00 09
00 00 00 09 00 00 00 08 00 00 00 09 00 00 00 08 00 00 00 08 00 00 00 09 00 00 00 09 00 00 00 09
00 00 00 09 00 00 00 05 00 00 00 08 00 00 00 08 00 00 00 07 00 00 00 08 00 00 00 00 00 00 00 01
00 00 00 02 00 00 00 03 00 00 00 04 00 00 00 05 00 00 00 06 00 00 00 07 00 00 00 0a 00 00 00 0b
00 00 00 0c 00 00 00 0d 00 00 00 0e 00 00 00 00 00 00 00 01 00 00 00 02 00 00 00 03 00 00 00 04
00 00 00 05 00 00 00 0a 00 00 00 0b 00 00 00 0c 00 00 00 0d 00 00 00 0e 00 00 00 0f 00 00 00 0a
00 00 00 0b 00 00 00 0c 00 00 00 0d 00 00 00 0e 00 00 00 0f 00 00 00 0a 00 00 00 0b 00 00 00 0c
00 00 00 0d 00 00 00 0e 00 00 00 0f 00 00 00 00 00 00 00 01 00 00 00 02 00 00 00 03 00 00 00 04
00 00 00 05 00 00 00 0a 00 00 00 0b 00 00 00 0c 00 00 00 0d 00 00 00 0e 00 00 00 00 00 00 00 01
00 00 00 02 00 00 00 03 00 00 00 04 00 00 00 05 00 00 00 06 00 00 00 07 00 00 00 08 00 00 00 00
00 00 00 2c 00 00 00 00 00 00 00 0f 00 00 00 0a 00 00 00 0f 00 00 00 0b 00 00 00 0f 00 00 00 0c
00 00 00 0f 00 00 00 0d 00 00 00 0f 00 00 00 0f 00 00 00 0e 00 00 00 09 00 00 00 00 00 00 00 38
01 00 00 00 00 00 00 00 00 00 00 01 00 00 00 02 00 00 00 03 00 00 00 04 00 00 00 05 00 00 00 06
00 00 00 07 00 00 00 00 00 00 00 00 00 00 00 01 00 00 00 01 00 00 00 0f 00 00 00 01 00 00 00 02
00 00 00 02 00 00 00 0f 00 00 00 02 00 00 00 03 00 00 00 03 00 00 00 0f 00 00 00 03 00 00 00 04
00 00 00 04 00 00 00 0f 00 00 00 04 00 00 00 05 00 00 00 05 00 00 00 05 00 00 00 06 00 00 00 06
00 00 00 0f 00 00 00 05 00 00 00 05 00 00 00 05 00 00 00 06 00 00 00 0f 00 00 00 0a 00 00 00 0b
00 00 00 0c 00 00 00 0d 00 00 00 0e 00 00 00 00 00 00 00 01 00 00 00 02 00 00 00 03 00 00 00 04
00 00 00 05 00 00 00 00 00 00 00 01 00 00 00 02 00 00 00 03 00 00 00 04 00 00 00 05 00 00 00 01
00 00 00 02 00 00 00 03 00 00 00 04 00 00 00 05 00 00 00 06 00 00 00 00 00 00 00 01 00 00 00 02
00 00 00 03 00 00 00 04 00 00 00 05 00 00 00 06 00 00 00 00 00 00 00 01 00 00 00 02 00 00 00 03
00 00 00 04 00 00 00 05 00 00 00 0a 00 00 00 0b 00 00 00 0c 00 00 00 0d 00 00 00 0e 00 00 00 0a
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00                                                   
```

Equivalent .kmx data:

```
APPEND_TO_REGISTER q0 r0
APPEND_TO_REGISTER q1 r0
APPEND_TO_REGISTER q2 r0
APPEND_TO_REGISTER q3 r0
APPEND_TO_REGISTER q4 r0
APPEND_TO_REGISTER q5 r0
APPEND_TO_REGISTER q6 r0
APPEND_TO_REGISTER q7 r0
APPEND_TO_REGISTER b0 r1
APPEND_TO_REGISTER b1 r1
APPEND_TO_REGISTER b2 r1
APPEND_TO_REGISTER b3 r1
APPEND_TO_REGISTER b4 r1
APPEND_TO_REGISTER b5 r1
APPEND_TO_REGISTER b6 r1
APPEND_TO_REGISTER b7 r1
X q0 if b0
X q1 if b1
X q2 if b2
X q3 if b3
X q4 if b4
X q5 if b5
X q6 if b6
X q7 if b7
CX q1 q10
CX q2 q11
CX q3 q12
CX q4 q13
CX q5 q14
CX q6 q15
R q8
CX q0 q8 if b0
X q8 if b0
R q9
SWAP q8 q9
X q9 if b1
CCX q1 q9 q8
X q9 if b1
CX q9 q1
HMR q9 b15
BIT_INVERT b10 if b15
X q8 if b1
R q9
SWAP q8 q9
X q9 if b2
CCX q2 q9 q8
X q9 if b2
CX q9 q2
HMR q9 b15
BIT_INVERT b11 if b15
X q8 if b2
R q9
SWAP q8 q9
X q9 if b3
CCX q3 q9 q8
X q9 if b3
CX q9 q3
HMR q9 b15
BIT_INVERT b12 if b15
X q8 if b3
R q9
SWAP q8 q9
X q9 if b4
CCX q4 q9 q8
X q9 if b4
CX q9 q4
HMR q9 b15
BIT_INVERT b13 if b15
X q8 if b4
R q9
X q8 if b5
CCX q5 q8 q9
X q8 if b5
CX q8 q5
X q9 if b5
X q9 if b6
CCX q6 q9 q7
X q9 if b6
CX q9 q6
HMR q9 b15
PUSH_CONDITION if b15
Z q5 if b5
Z q8 if b5
NEG if b5
CZ q5 q8
Z q8
POP_CONDITION
X q7 if b6
HMR q8 b15
BIT_INVERT b14 if b15
CX q1 q10
CX q2 q11
CX q3 q12
CX q4 q13
CX q5 q14
CX q6 q15
X q0
X q1
X q2
X q3
X q4
X q5
X q6
X q7
Z q10 if b10
Z q11 if b11
Z q12 if b12
Z q13 if b13
Z q14 if b14
X q0 if b0
X q1 if b1
X q2 if b2
X q3 if b3
X q4 if b4
X q5 if b5
CCX q14 q5 q15
CCX q13 q4 q14
CCX q12 q3 q13
CCX q11 q2 q12
CCX q10 q1 q11
X q10 if b0
X q11 if b1
X q12 if b2
X q13 if b3
X q14 if b4
X q15 if b5
X q10 if b1
X q11 if b2
X q12 if b3
X q13 if b4
X q14 if b5
X q15 if b6
CX q0 q10 if b0
CCX q10 q1 q11
CCX q11 q2 q12
CCX q12 q3 q13
CCX q13 q4 q14
CCX q14 q5 q15
X q10 if b1
X q11 if b2
X q12 if b3
X q13 if b4
X q14 if b5
X q15 if b6
X q0 if b0
X q1 if b1
X q2 if b2
X q3 if b3
X q4 if b4
X q5 if b5
Z q10 if b10
Z q11 if b11
Z q12 if b12
Z q13 if b13
Z q14 if b14
X q0
X q1
X q2
X q3
X q4
X q5
X q6
X q7
```

## A Circuit with Every Operation

Hex dump of .kmb data:

```
d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf 02 00 00 00 19 00 00 00 1a 00 00 00 02 00 00 00
25 00 00 00 00 00 00 00 04 00 00 00 00 00 00 00 06 00 00 00 00 00 00 00 0c 00 00 00 00 00 00 00
0a 00 00 00 00 00 00 00 10 00 00 00 00 00 00 00 02 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
10 00 00 00 00 00 00 00 00 00 00 00 02 00 00 00 00 00 00 80 00 00 00 00 00 00 00 00 00 00 00 00
11 00 00 00 00 00 00 00 09 00 00 00 00 00 00 00 74 65 73 74 0d 0a 5c 22 23 01 00 00 00 00 00 00
00 25 00 00 00 00 00 00 00 07 11 91 21 a1 31 b1 00 80 10 90 20 a0 30 b0 50 d0 12 92 42 c2 40 c0
41 c1 22 a2 81 41 11 81 41 11 01 01 13 93 02 00 00 00 00 00 00 00 10 00 00 00 00 00 00 00 00 00
00 00 01 00 00 00 06 00 00 00 07 00 00 00 03 00 00 00 00 00 00 00 10 00 00 00 00 00 00 00 01 00
00 00 02 00 00 00 07 00 00 00 08 00 00 00 04 00 00 00 00 00 00 00 10 00 00 00 00 00 00 00 02 00
00 00 03 00 00 00 08 00 00 00 09 00 00 00 05 00 00 00 00 00 00 00 18 00 00 00 00 00 00 00 00 00
00 00 01 00 00 00 05 00 00 00 06 00 00 00 0d 00 00 00 0f 00 00 00 06 00 00 00 00 00 00 00 18 00
00 00 00 00 00 00 01 00 00 00 02 00 00 00 06 00 00 00 07 00 00 00 0e 00 00 00 10 00 00 00 07 00
00 00 00 00 00 00 30 00 00 00 00 00 00 00 00 00 00 00 01 00 00 00 05 00 00 00 06 00 00 00 09 00
00 00 0a 00 00 00 0b 00 00 00 0c 00 00 00 14 00 00 00 15 00 00 00 17 00 00 00 18 00 00 00 08 00
00 00 00 00 00 00 28 00 00 00 00 00 00 00 07 00 00 00 11 00 00 00 08 00 00 00 09 00 00 00 0b 00
00 00 0c 00 00 00 0e 00 00 00 0f 00 00 00 17 00 00 00 18 00 00 00 09 00 00 00 00 00 00 00 40 00
00 00 00 00 00 00 00 00 00 00 01 00 00 00 02 00 00 00 03 00 00 00 04 00 00 00 05 00 00 00 06 00
00 00 12 00 00 00 13 00 00 00 0a 00 00 00 0d 00 00 00 10 00 00 00 14 00 00 00 15 00 00 00 16 00
00 00 19 00 00 00 0a 00 00 00 00 00 00 00 20 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 40 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 20                              
```

Equivalent .kmx data:

```
APPEND_TO_REGISTER q0 r0
APPEND_TO_REGISTER b0 r0
REGISTER r1 "test\r\n\B\Q\P"
DEBUG_PRINT

X q0
X q1 if b0
CX q0 q1
CX q1 q2 if b1
CCX q0 q1 q2
CCX q1 q2 q3 if b2
NEG
NEG if b3

Z q5
Z q6 if b4
CZ q5 q6
CZ q6 q7 if b5
CCZ q6 q7 q8
CCZ q7 q8 q9 if b6
HMR q9 b7
HMR q10 b17 if b18
R q11
R q12 if b19
BIT_INVERT b8
BIT_INVERT b9 if b10
BIT_STORE0 b11
BIT_STORE0 b12 if b13
BIT_STORE1 b14
BIT_STORE1 b15 if b16

SWAP q13 q14
SWAP q15 q16 if b20

PUSH_CONDITION if b21
BIT_STORE1 b23
X q20
PUSH_CONDITION if b22
BIT_STORE1 b24
X q21
POP_CONDITION
POP_CONDITION

Z_POW q23 0.5
Z_POW q24 0.25 if b25
```
