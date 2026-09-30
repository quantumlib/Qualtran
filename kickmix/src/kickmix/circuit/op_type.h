#ifndef KICKMIX_CIRCUIT_OP_TYPE_H
#define KICKMIX_CIRCUIT_OP_TYPE_H

#include <cstdint>
#include <ostream>

namespace kickmix {

/// Qubit target count flags embedded into operation types.
///
/// An operation targets no qubits if (op_type & OP_HAS_3Q) == 0.
/// An operation targets one qubit if (op_type & OP_HAS_3Q) == OP_HAS_1Q.
/// An operation targets two qubits if (op_type & OP_HAS_3Q) == OP_HAS_2Q.
/// An operation targets three qubits if (op_type & OP_HAS_3Q) == OP_HAS_3Q.
constexpr uint8_t OP_HAS_1Q = 0x10;
constexpr uint8_t OP_HAS_2Q = 0x20;
constexpr uint8_t OP_HAS_3Q = OP_HAS_2Q | OP_HAS_1Q;

/// Bit target flag embedded into operation types.
///
/// An operation targets no bits if (op_type & OP_HAS_1B) == 0.
/// An operation targets one bit if (op_type & OP_HAS_1B) == OP_HAS_1B.
constexpr uint8_t OP_HAS_1B = 0x40;

/// Bit condition flag embedded into operation types.
///
/// An operation has no condition bit if (op_type & OP_HAS_COND) == 0.
/// An operation has a condition bit if (op_type & OP_HAS_COND) == OP_HAS_COND.
constexpr uint8_t OP_HAS_COND = 0x80;

/// Operation types supported by the kickmix simulator.
enum class OpType : uint8_t {
    NEG = 0x0,                // Flip global phase.
    POP_CONDITION = 0x1,      // Unconditionally removes a condition from the condition stack. Operations other than
                              // PUSH_CONDITION/POP_CONDITION only occur when the condition stack is satisfied.
    DEBUG_PRINT_EMPTY = 0x7,  // Asks simulator to print unspecified debugging information.

    BIT_STORE0 = 0x0 | OP_HAS_1B,     // Stores 0 into a bit.
    BIT_STORE1 = 0x1 | OP_HAS_1B,     // Stores 1 into a bit.
    BIT_INVERT = 0x2 | OP_HAS_1B,     // Inverts a bit.
    DEBUG_PRINT_C = 0x7 | OP_HAS_1B,  // Asks simulator to print debugging information about a bit.

    Z = 0x0 | OP_HAS_1Q,              // Phase flips a qubit (negates global phase if qubit is 1).
    X = 0x1 | OP_HAS_1Q,              // Bit flips a qubit (applies a NOT gate to it).
    R = 0x2 | OP_HAS_1Q,              // Ensures a qubit is 0 (randomizes global phase if qubit was 1).
    Z_POW = 0x3 | OP_HAS_1Q,          // Single-qubit arbitrary angle phasing.
    DEBUG_PRINT_Q = 0x7 | OP_HAS_1Q,  // Asks simulator to print debugging information about a qubit.

    HMR = 0x0 | OP_HAS_1Q | OP_HAS_1B,  // Hadamard+measure+reset a qubit.

    CZ = 0x0 | OP_HAS_2Q,    // Negate global phase when two qubits are both 1.
    CX = 0x1 | OP_HAS_2Q,    // Bit flip second qubit if first qubit is 1.
    SWAP = 0x2 | OP_HAS_2Q,  // Exchanges the values of two qubits.

    CCZ = 0x0 | OP_HAS_3Q,  // Negate global phase when three qubits are all 1.
    CCX = 0x1 | OP_HAS_3Q,  // Bit flip third qubit if first two qubits are 1.

    NEG_IF = 0x0 | OP_HAS_COND,  // Flip global phase if given condition bit is 1.
    PUSH_CONDITION =
        0x1 | OP_HAS_COND,  // Unconditionally push a condition onto the condition stack. Operations other than
                            // PUSH_CONDITION/POP_CONDITION only occur when the condition stack is satisfied.
    DEBUG_PRINT_EMPTY_IF =
        0x7 | OP_HAS_COND,  // Asks simulator to print unspecified debugging information if given condition bit is 1.

    BIT_STORE0_IF = 0x0 | OP_HAS_1B | OP_HAS_COND,  // Stores 0 into a bit if given condition bit is 1.
    BIT_STORE1_IF = 0x1 | OP_HAS_1B | OP_HAS_COND,  // Stores 1 into a bit if given condition bit is 1.
    BIT_INVERT_IF = 0x2 | OP_HAS_1B | OP_HAS_COND,  // Inverts a bit if given condition bit is 1.
    DEBUG_PRINT_C_IF =
        0x7 | OP_HAS_1B |
        OP_HAS_COND,  // Asks simulator to print debugging information about a bit if given condition bit is 1.

    Z_IF = 0x0 | OP_HAS_1Q |
           OP_HAS_COND,  // Phase flips a qubit (negates global phase if qubit is 1) if given condition bit is 1.
    X_IF = 0x1 | OP_HAS_1Q | OP_HAS_COND,  // Bit flips a qubit (applies a NOT gate to it) if given condition bit is 1.
    R_IF = 0x2 | OP_HAS_1Q |
           OP_HAS_COND,  // Ensures a qubit is 0 (randomizes global phase if qubit was 1) if given condition bit is 1.
    DEBUG_PRINT_Q_IF =
        0x7 | OP_HAS_1Q |
        OP_HAS_COND,  // Asks simulator to print debugging information about a qubit if given condition bit is 1.

    HMR_IF = 0x0 | OP_HAS_1Q | OP_HAS_1B | OP_HAS_COND,  // Hadamard+measure+reset a qubit if given condition bit is 1.

    CZ_IF =
        0x0 | OP_HAS_2Q | OP_HAS_COND,  // Negate global phase when two qubits are both 1 if given condition bit is 1.
    CX_IF = 0x1 | OP_HAS_2Q | OP_HAS_COND,    // Bit flip second qubit if first qubit is 1 if given condition bit is 1.
    SWAP_IF = 0x2 | OP_HAS_2Q | OP_HAS_COND,  // Exchanges the values of two qubits if given condition bit is 1.

    CCZ_IF =
        0x0 | OP_HAS_3Q | OP_HAS_COND,  // Negate global phase when three qubits are all 1 if given condition bit is 1.
    CCX_IF =
        0x1 | OP_HAS_3Q | OP_HAS_COND,  // Bit flip third qubit if first two qubits are 1 if given condition bit is 1.
    Z_POW_IF = 0x3 | OP_HAS_1Q | OP_HAS_COND,  // Conditional single-qubit arbitrary angle phasing.
};
std::ostream &operator<<(std::ostream &out, const OpType &rhs);

inline OpType operator|(OpType t, uint8_t other) {
    return (OpType)((uint8_t)t | other);
}

inline uint8_t operator&(OpType t, uint8_t other) {
    return (uint8_t)t & other;
}

extern std::array<std::string_view, 256> OP_TYPE_NAME_TABLE;
extern std::array<OpType, 36> OP_TYPE_TABLE;

}  // namespace kickmix

#endif
