#include "kickmix/circuit/op_type.h"

#include <array>
#include <string_view>

using namespace kickmix;

static consteval std::array<std::string_view, 256> make_op_type_name_table() {
    std::array<std::string_view, 256> table;
    table[(uint8_t)OpType::NEG] = "NEG";
    table[(uint8_t)OpType::BIT_INVERT] = "BIT_INVERT";
    table[(uint8_t)OpType::BIT_STORE0] = "BIT_STORE0";
    table[(uint8_t)OpType::BIT_STORE1] = "BIT_STORE1";
    table[(uint8_t)OpType::X] = "X";
    table[(uint8_t)OpType::Z] = "Z";
    table[(uint8_t)OpType::R] = "R";
    table[(uint8_t)OpType::HMR] = "HMR";
    table[(uint8_t)OpType::CX] = "CX";
    table[(uint8_t)OpType::CZ] = "CZ";
    table[(uint8_t)OpType::SWAP] = "SWAP";
    table[(uint8_t)OpType::CCX] = "CCX";
    table[(uint8_t)OpType::CCZ] = "CCZ";
    table[(uint8_t)OpType::DEBUG_PRINT_EMPTY] = "DEBUG_PRINT_EMPTY";
    table[(uint8_t)OpType::DEBUG_PRINT_Q] = "DEBUG_PRINT_Q";
    table[(uint8_t)OpType::DEBUG_PRINT_C] = "DEBUG_PRINT_C";
    table[(uint8_t)OpType::POP_CONDITION] = "POP_CONDITION";
    table[(uint8_t)OpType::PUSH_CONDITION] = "PUSH_CONDITION";
    table[(uint8_t)OpType::NEG_IF] = "NEG_IF";
    table[(uint8_t)OpType::BIT_INVERT_IF] = "BIT_INVERT_IF";
    table[(uint8_t)OpType::BIT_STORE0_IF] = "BIT_STORE0_IF";
    table[(uint8_t)OpType::BIT_STORE1_IF] = "BIT_STORE1_IF";
    table[(uint8_t)OpType::X_IF] = "X_IF";
    table[(uint8_t)OpType::Z_IF] = "Z_IF";
    table[(uint8_t)OpType::R_IF] = "R_IF";
    table[(uint8_t)OpType::HMR_IF] = "HMR_IF";
    table[(uint8_t)OpType::CX_IF] = "CX_IF";
    table[(uint8_t)OpType::CZ_IF] = "CZ_IF";
    table[(uint8_t)OpType::SWAP_IF] = "SWAP_IF";
    table[(uint8_t)OpType::CCX_IF] = "CCX_IF";
    table[(uint8_t)OpType::CCZ_IF] = "CCZ_IF";
    table[(uint8_t)OpType::DEBUG_PRINT_EMPTY_IF] = "DEBUG_PRINT_EMPTY_IF";
    table[(uint8_t)OpType::DEBUG_PRINT_Q_IF] = "DEBUG_PRINT_Q_IF";
    table[(uint8_t)OpType::DEBUG_PRINT_C_IF] = "DEBUG_PRINT_C_IF";
    table[(uint8_t)OpType::Z_POW] = "Z_POW";
    table[(uint8_t)OpType::Z_POW_IF] = "Z_POW_IF";
    return table;
}
std::array<std::string_view, 256> kickmix::OP_TYPE_NAME_TABLE = make_op_type_name_table();

std::array<OpType, 36> kickmix::OP_TYPE_TABLE{
    OpType::NEG,
    OpType::BIT_INVERT,
    OpType::BIT_STORE0,
    OpType::BIT_STORE1,
    OpType::X,
    OpType::Z,
    OpType::R,
    OpType::HMR,
    OpType::CX,
    OpType::CZ,
    OpType::SWAP,
    OpType::CCX,
    OpType::CCZ,
    OpType::Z_POW,
    OpType::DEBUG_PRINT_EMPTY,
    OpType::DEBUG_PRINT_Q,
    OpType::DEBUG_PRINT_C,
    OpType::POP_CONDITION,
    OpType::PUSH_CONDITION,
    OpType::NEG_IF,
    OpType::BIT_INVERT_IF,
    OpType::BIT_STORE0_IF,
    OpType::BIT_STORE1_IF,
    OpType::X_IF,
    OpType::Z_IF,
    OpType::R_IF,
    OpType::HMR_IF,
    OpType::CX_IF,
    OpType::CZ_IF,
    OpType::SWAP_IF,
    OpType::CCX_IF,
    OpType::CCZ_IF,
    OpType::Z_POW_IF,
    OpType::DEBUG_PRINT_EMPTY_IF,
    OpType::DEBUG_PRINT_Q_IF,
    OpType::DEBUG_PRINT_C_IF,
};

std::ostream &kickmix::operator<<(std::ostream &out, const OpType &rhs) {
    std::string_view name = OP_TYPE_NAME_TABLE[(uint8_t)rhs];
    out << name;
    if (name == "") {
        out << "unknown_op_type[" << (uint16_t)rhs << "]";
    }
    return out;
}
