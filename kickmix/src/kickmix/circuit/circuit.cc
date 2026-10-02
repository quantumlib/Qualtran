#include "kickmix/circuit/circuit.h"

#include <cstring>
#include <iostream>
#include <sstream>

#include "circuit_util.h"
#include "kickmix/build/circuit_builder.h"
#include "kickmix/mem/util.h"
#include "kickmix/simd/simd.h"
#include "kickmix/util/binary_file_tools.h"

using namespace kickmix;

constexpr std::array<uint8_t, 16> KMB_MAGIC_BYTES{
    0xd7, 0x50, 0xc7, 0xd5, 0xc3, 0x29, 0xd3, 0x26, 0xe3, 0xcc, 0x9f, 0x68, 0x34, 0xf2, 0xb8, 0xbf};

void Circuit::clear() {
    num_ops = 0;
    num_qqq_ops = 0;
    num_qq_ops = 0;
    num_q_ops = 0;
    num_c_ops = 0;
    num_b_ops = 0;
    num_angle_ops = 0;
    if (op_types != nullptr) {
        free(op_types);
        op_types = nullptr;
    }
    if (qqq0 != nullptr) {
        free(qqq0);
        qqq0 = nullptr;
    }
    if (qqq1 != nullptr) {
        free(qqq1);
        qqq1 = nullptr;
    }
    if (qqq2 != nullptr) {
        free(qqq2);
        qqq2 = nullptr;
    }
    if (qq0 != nullptr) {
        free(qq0);
        qq0 = nullptr;
    }
    if (qq1 != nullptr) {
        free(qq1);
        qq1 = nullptr;
    }
    if (q0 != nullptr) {
        free(q0);
        q0 = nullptr;
    }
    if (bc != nullptr) {
        free(bc);
        bc = nullptr;
    }
    if (b0 != nullptr) {
        free(b0);
        b0 = nullptr;
    }
    if (angles != nullptr) {
        free(angles);
        angles = nullptr;
    }
}

Circuit::~Circuit() {
    clear();
}

Circuit::Circuit(std::string_view kmx_text) {
    MutableCircuit result;
    result.append_from_kmx_text(kmx_text);
    op_types = nullptr;
    qqq0 = nullptr;
    qqq1 = nullptr;
    qqq2 = nullptr;
    qq0 = nullptr;
    qq1 = nullptr;
    q0 = nullptr;
    bc = nullptr;
    b0 = nullptr;
    angles = nullptr;
    num_ops = 0;
    num_qqq_ops = 0;
    num_qq_ops = 0;
    num_q_ops = 0;
    num_c_ops = 0;
    num_b_ops = 0;
    num_bits = 0;
    num_qubits = 0;
    num_angle_ops = 0;
    *this = result.to_validated_circuit();
}

Circuit Circuit::from_kmx_file(FILE *file) {
    MutableCircuit result;
    result.append_from_kmx_file(file);
    return result.to_validated_circuit();
}

Circuit Circuit::from_kmx_or_kmb_file(FILE *file) {
    int i = getc(file);
    if (i == EOF) {
        return Circuit();
    } else if (i == KMB_MAGIC_BYTES[0]) {
        // Read rest of magic bytes.
        for (size_t k = 1; k < KMB_MAGIC_BYTES.size(); k++) {
            if (getc(file) != KMB_MAGIC_BYTES[k]) {
                throw std::invalid_argument(
                    "File deviated from the magic bytes identifying a kickmix binary format file.");
            }
        }
        return from_kmb_file(file, true);
    } else {
        // Read rest of line.
        std::string line;
        while (true) {
            if (i == EOF || i == '\n') {
                break;
            }
            line.push_back((char)i);
            i = getc(file);
        }

        MutableCircuit result;
        result.append_from_kmx_line(line);
        result.append_from_kmx_file(file);
        return result.to_validated_circuit();
    }
}

bool Circuit::operator==(const Circuit &rhs) const {
    if (register_data != rhs.register_data) {
        return false;
    }
    if (num_ops != rhs.num_ops) {
        return false;
    }
    if (num_qqq_ops != rhs.num_qqq_ops) {
        return false;
    }
    if (num_qq_ops != rhs.num_qq_ops) {
        return false;
    }
    if (num_q_ops != rhs.num_q_ops) {
        return false;
    }
    if (num_c_ops != rhs.num_c_ops) {
        return false;
    }
    if (num_b_ops != rhs.num_b_ops) {
        return false;
    }
    if (num_qubits != rhs.num_qubits) {
        return false;
    }
    if (num_bits != rhs.num_bits) {
        return false;
    }
    if (num_angle_ops != rhs.num_angle_ops) {
        return false;
    }
    if (num_ops > 0 && memcmp(op_types, rhs.op_types, sizeof(OpType) * num_ops)) {
        return false;
    }
    if (num_qqq_ops > 0 && memcmp(qqq0, rhs.qqq0, sizeof(uint32_t) * num_qqq_ops)) {
        return false;
    }
    if (num_qqq_ops > 0 && memcmp(qqq1, rhs.qqq1, sizeof(uint32_t) * num_qqq_ops)) {
        return false;
    }
    if (num_qqq_ops > 0 && memcmp(qqq2, rhs.qqq2, sizeof(uint32_t) * num_qqq_ops)) {
        return false;
    }
    if (num_qq_ops > 0 && memcmp(qq0, rhs.qq0, sizeof(uint32_t) * num_qq_ops)) {
        return false;
    }
    if (num_qq_ops > 0 && memcmp(qq1, rhs.qq1, sizeof(uint32_t) * num_qq_ops)) {
        return false;
    }
    if (num_q_ops > 0 && memcmp(q0, rhs.q0, sizeof(uint32_t) * num_q_ops)) {
        return false;
    }
    if (num_c_ops > 0 && memcmp(bc, rhs.bc, sizeof(uint32_t) * num_c_ops)) {
        return false;
    }
    if (num_b_ops > 0 && memcmp(b0, rhs.b0, sizeof(uint32_t) * num_b_ops)) {
        return false;
    }
    if (num_angle_ops > 0 && memcmp(angles, rhs.angles, sizeof(FixedPrecisionAngle128) * num_angle_ops)) {
        return false;
    }
    return true;
}

Circuit::Circuit(Circuit &&other) noexcept {
    register_data = std::move(other.register_data);
    op_types = other.op_types;
    other.op_types = nullptr;
    qqq0 = other.qqq0;
    other.qqq0 = nullptr;
    qqq1 = other.qqq1;
    other.qqq1 = nullptr;
    qqq2 = other.qqq2;
    other.qqq2 = nullptr;
    qq0 = other.qq0;
    other.qq0 = nullptr;
    qq1 = other.qq1;
    other.qq1 = nullptr;
    q0 = other.q0;
    other.q0 = nullptr;
    bc = other.bc;
    other.bc = nullptr;
    b0 = other.b0;
    other.b0 = nullptr;
    angles = other.angles;
    other.angles = nullptr;
    num_ops = other.num_ops;
    other.num_ops = 0;
    num_qqq_ops = other.num_qqq_ops;
    other.num_qqq_ops = 0;
    num_qq_ops = other.num_qq_ops;
    other.num_qq_ops = 0;
    num_q_ops = other.num_q_ops;
    other.num_q_ops = 0;
    num_c_ops = other.num_c_ops;
    other.num_c_ops = 0;
    num_b_ops = other.num_b_ops;
    other.num_b_ops = 0;
    num_bits = other.num_bits;
    other.num_bits = 0;
    num_qubits = other.num_qubits;
    other.num_qubits = 0;
    num_angle_ops = other.num_angle_ops;
    other.num_angle_ops = 0;
}

template <typename T>
static T *aligned_alloc_32(size_t count) {
    if (count == 0) {
        return nullptr;
    }
    size_t bytes = count * sizeof(T);
    bytes += size_t{31};
    bytes &= ~size_t{31};
    return (T *)std::aligned_alloc(32, bytes);
}

Circuit::Circuit(const Circuit &other) {
    register_data = other.register_data;
    num_ops = other.num_ops;
    num_qqq_ops = other.num_qqq_ops;
    num_qq_ops = other.num_qq_ops;
    num_q_ops = other.num_q_ops;
    num_c_ops = other.num_c_ops;
    num_b_ops = other.num_b_ops;
    num_bits = other.num_bits;
    num_qubits = other.num_qubits;
    num_angle_ops = other.num_angle_ops;

    op_types = aligned_alloc_32<OpType>(num_ops);
    qqq0 = aligned_alloc_32<uint32_t>(num_qqq_ops);
    qqq1 = aligned_alloc_32<uint32_t>(num_qqq_ops);
    qqq2 = aligned_alloc_32<uint32_t>(num_qqq_ops);
    qq0 = aligned_alloc_32<uint32_t>(num_qq_ops);
    qq1 = aligned_alloc_32<uint32_t>(num_qq_ops);
    q0 = aligned_alloc_32<uint32_t>(num_q_ops);
    bc = aligned_alloc_32<uint32_t>(num_c_ops);
    b0 = aligned_alloc_32<uint32_t>(num_b_ops);
    angles = aligned_alloc_32<FixedPrecisionAngle128>(num_angle_ops);

    memcpy(op_types, other.op_types, num_ops);
    memcpy(qqq0, other.qqq0, num_qqq_ops * sizeof(uint32_t));
    memcpy(qqq1, other.qqq1, num_qqq_ops * sizeof(uint32_t));
    memcpy(qqq2, other.qqq2, num_qqq_ops * sizeof(uint32_t));
    memcpy(qq0, other.qq0, num_qq_ops * sizeof(uint32_t));
    memcpy(qq1, other.qq1, num_qq_ops * sizeof(uint32_t));
    memcpy(q0, other.q0, num_q_ops * sizeof(uint32_t));
    memcpy(bc, other.bc, num_c_ops * sizeof(uint32_t));
    memcpy(b0, other.b0, num_b_ops * sizeof(uint32_t));
    memcpy(angles, other.angles, num_angle_ops * sizeof(FixedPrecisionAngle128));
}

Circuit &Circuit::operator=(const Circuit &other) {
    if (this == &other) {
        return *this;
    }
    clear();

    register_data = other.register_data;
    num_ops = other.num_ops;
    num_qqq_ops = other.num_qqq_ops;
    num_qq_ops = other.num_qq_ops;
    num_q_ops = other.num_q_ops;
    num_c_ops = other.num_c_ops;
    num_b_ops = other.num_b_ops;
    num_bits = other.num_bits;
    num_qubits = other.num_qubits;
    num_angle_ops = other.num_angle_ops;

    op_types = aligned_alloc_32<OpType>(num_ops);
    qqq0 = aligned_alloc_32<uint32_t>(num_qqq_ops);
    qqq1 = aligned_alloc_32<uint32_t>(num_qqq_ops);
    qqq2 = aligned_alloc_32<uint32_t>(num_qqq_ops);
    qq0 = aligned_alloc_32<uint32_t>(num_qq_ops);
    qq1 = aligned_alloc_32<uint32_t>(num_qq_ops);
    q0 = aligned_alloc_32<uint32_t>(num_q_ops);
    bc = aligned_alloc_32<uint32_t>(num_c_ops);
    b0 = aligned_alloc_32<uint32_t>(num_b_ops);
    angles = aligned_alloc_32<FixedPrecisionAngle128>(num_angle_ops);

    memcpy(op_types, other.op_types, num_ops);
    memcpy(qqq0, other.qqq0, num_qqq_ops * sizeof(uint32_t));
    memcpy(qqq1, other.qqq1, num_qqq_ops * sizeof(uint32_t));
    memcpy(qqq2, other.qqq2, num_qqq_ops * sizeof(uint32_t));
    memcpy(qq0, other.qq0, num_qq_ops * sizeof(uint32_t));
    memcpy(qq1, other.qq1, num_qq_ops * sizeof(uint32_t));
    memcpy(q0, other.q0, num_q_ops * sizeof(uint32_t));
    memcpy(bc, other.bc, num_c_ops * sizeof(uint32_t));
    memcpy(b0, other.b0, num_b_ops * sizeof(uint32_t));
    memcpy(angles, other.angles, num_angle_ops * sizeof(FixedPrecisionAngle128));

    return *this;
}

Circuit &Circuit::operator=(Circuit &&other) noexcept {
    if (this == &other) {
        return *this;
    }
    clear();
    register_data = std::move(other.register_data);
    op_types = other.op_types;
    other.op_types = nullptr;
    qqq0 = other.qqq0;
    other.qqq0 = nullptr;
    qqq1 = other.qqq1;
    other.qqq1 = nullptr;
    qqq2 = other.qqq2;
    other.qqq2 = nullptr;
    qq0 = other.qq0;
    other.qq0 = nullptr;
    qq1 = other.qq1;
    other.qq1 = nullptr;
    q0 = other.q0;
    other.q0 = nullptr;
    bc = other.bc;
    other.bc = nullptr;
    b0 = other.b0;
    other.b0 = nullptr;
    angles = other.angles;
    other.angles = nullptr;
    num_ops = other.num_ops;
    other.num_ops = 0;
    num_qqq_ops = other.num_qqq_ops;
    other.num_qqq_ops = 0;
    num_qq_ops = other.num_qq_ops;
    other.num_qq_ops = 0;
    num_q_ops = other.num_q_ops;
    other.num_q_ops = 0;
    num_c_ops = other.num_c_ops;
    other.num_c_ops = 0;
    num_b_ops = other.num_b_ops;
    other.num_b_ops = 0;
    num_bits = other.num_bits;
    other.num_bits = 0;
    num_qubits = other.num_qubits;
    other.num_qubits = 0;
    num_angle_ops = other.num_angle_ops;
    other.num_angle_ops = 0;
    return *this;
}

std::string Circuit::describe_qubit_control_collision() const {
    std::string result =
        "Circuit is invalid due to a qubit collision.\nOne or more operations used the same qubit twice (e.g. control "
        "same qubit as target):";
    size_t num_results = 0;
    iter_ops([&](const Op &op) {
        bool has_collision = false;
        if ((op.kind & OP_HAS_3Q) == OP_HAS_3Q) {
            auto q0 = op.q_control2;
            auto q1 = op.q_control1;
            auto q2 = op.q_target;
            if (q0 == q1 || q0 == q2 || q1 == q2) {
                has_collision = true;
            }
        } else if ((op.kind & OP_HAS_3Q) == OP_HAS_2Q) {
            auto q0 = op.q_target;
            auto q1 = op.q_control1;
            if (q0 == q1) {
                has_collision = true;
            }
        }
        if (has_collision) {
            num_results += 1;
            if (num_results < 5) {
                result.append("\n    ");
                result.append(op.kmx_str());
            }
        }
    });
    if (num_results > 5) {
        std::stringstream ss;
        ss << "\n    (... " << (num_results - 5) << " more ...)";
        result.append(ss.str());
    }
    return result;
}

bool Circuit::compute_has_qubit_control_collision() const {
    const b256 *block_qqq0 = (const b256 *)qqq0;
    const b256 *block_qqq1 = (const b256 *)qqq1;
    const b256 *block_qqq2 = (const b256 *)qqq2;
    b256 acc{};

    size_t k = 0;
    bool has_collision = false;
    for (; k + 8 < num_qqq_ops; k += 8) {
        auto a = *block_qqq0++;
        auto b = *block_qqq1++;
        auto c = *block_qqq2++;
        acc |= a.u32_eq(b) | a.u32_eq(c) | b.u32_eq(c);
    }
    while (k < num_qqq_ops) {
        has_collision |= (qqq0[k] == qqq1[k]) | (qqq0[k] == qqq2[k]) | (qqq1[k] == qqq2[k]);
        k++;
    }

    const b256 *block_qq0 = (const b256 *)qq0;
    const b256 *block_qq1 = (const b256 *)qq1;
    k = 0;
    for (; k + 8 < num_qq_ops; k += 8) {
        auto a = *block_qq0++;
        auto b = *block_qq1++;
        acc |= a.u32_eq(b);
    }
    while (k < num_qq_ops) {
        has_collision |= qq0[k] == qq1[k];
        k++;
    }
    has_collision |= acc.non_zero();
    return has_collision;
}

std::string Circuit::str() const {
    std::stringstream ss;
    write_kmx_to(ss);
    return ss.str();
}
std::ostream &kickmix::operator<<(std::ostream &out, const Circuit &circuit) {
    circuit.write_kmx_to(out);
    return out;
}

static char escape_char_for(char c) {
    switch (c) {
        case '\n':
            return 'n';
        case '\r':
            return 'r';
        case '#':
            return 'P';
        case '\"':
            return 'Q';
        case '\\':
            return 'B';
        default:
            return 0;
    }
}

void Circuit::write_kmx_to(FILE *file) const {
    for (size_t k = 0; k < register_data.size(); k++) {
        const auto &reg = register_data[k];
        for (size_t k2 = 0; k2 < reg.contents.size(); k2++) {
            const auto &qb = reg.contents[k2];
            if (qb.is_qubit()) {
                fprintf(file, "APPEND_TO_REGISTER q%u r%u\n", (unsigned)qb.untagged_id(), (unsigned)k);
            } else {
                fprintf(file, "APPEND_TO_REGISTER b%u r%u\n", (unsigned)qb.untagged_id(), (unsigned)k);
            }
        }
        if (!reg.name.empty()) {
            fprintf(file, "REGISTER r%u \"", (unsigned)k);
            for (char c : reg.name) {
                char e = escape_char_for(c);
                if (e) {
                    putc('\\', file);
                    putc(e, file);
                } else {
                    putc(c, file);
                }
            }
            putc('"', file);
            putc('\n', file);
        } else if (reg.contents.empty()) {
            fprintf(file, "REGISTER r%u\n", (unsigned)k);
        }
    }

    uint32_t *cur_qqq0 = qqq0;
    uint32_t *cur_qqq1 = qqq1;
    uint32_t *cur_qqq2 = qqq2;
    uint32_t *cur_qq0 = qq0;
    uint32_t *cur_qq1 = qq1;
    uint32_t *cur_q0 = q0;
    uint32_t *cur_bit_cond = bc;
    uint32_t *cur_bit_targ = b0;
    FixedPrecisionAngle128 *cur_angle = angles;
    for (size_t k = 0; k < num_ops; k++) {
        switch (op_types[k]) {
            case OpType::NEG:
                fprintf(file, "NEG\n");
                break;
            case OpType::NEG_IF:
                fprintf(file, "NEG if b%u\n", (unsigned)*cur_bit_cond++);
                break;
            case OpType::POP_CONDITION:
                fprintf(file, "POP_CONDITION\n");
                break;
            case OpType::DEBUG_PRINT_EMPTY:
                fprintf(file, "DEBUG_PRINT\n");
                break;
            case OpType::BIT_STORE0:
                fprintf(file, "BIT_STORE0 b%u\n", (unsigned)*cur_bit_targ++);
                break;
            case OpType::BIT_STORE1:
                fprintf(file, "BIT_STORE1 b%u\n", (unsigned)*cur_bit_targ++);
                break;
            case OpType::BIT_INVERT:
                fprintf(file, "BIT_INVERT b%u\n", (unsigned)*cur_bit_targ++);
                break;
            case OpType::DEBUG_PRINT_C:
                fprintf(file, "DEBUG_PRINT b%u\n", (unsigned)*cur_bit_targ++);
                break;
            case OpType::Z:
                fprintf(file, "Z q%u\n", (unsigned)*cur_q0++);
                break;
            case OpType::X:
                fprintf(file, "X q%u\n", (unsigned)*cur_q0++);
                break;
            case OpType::R:
                fprintf(file, "R q%u\n", (unsigned)*cur_q0++);
                break;
            case OpType::DEBUG_PRINT_Q:
                fprintf(file, "DEBUG_PRINT q%u\n", (unsigned)*cur_q0++);
                break;
            case OpType::HMR:
                fprintf(file, "HMR q%u b%u\n", (unsigned)*cur_q0++, (unsigned)*cur_bit_targ++);
                break;
            case OpType::CZ:
                fprintf(file, "CZ q%u q%u\n", (unsigned)*cur_qq0++, (unsigned)*cur_qq1++);
                break;
            case OpType::CX:
                fprintf(file, "CX q%u q%u\n", (unsigned)*cur_qq0++, (unsigned)*cur_qq1++);
                break;
            case OpType::SWAP:
                fprintf(file, "SWAP q%u q%u\n", (unsigned)*cur_qq0++, (unsigned)*cur_qq1++);
                break;
            case OpType::CCZ:
                fprintf(file, "CCZ q%u q%u q%u\n", (unsigned)*cur_qqq0++, (unsigned)*cur_qqq1++, (unsigned)*cur_qqq2++);
                break;
            case OpType::Z_POW:
                fprintf(file, "Z_POW q%u ", (unsigned)*cur_q0++);
                (*cur_angle++).write_decimal_half_turns_to(file);
                putc('\n', file);
                break;
            case OpType::CCX:
                fprintf(file, "CCX q%u q%u q%u\n", (unsigned)*cur_qqq0++, (unsigned)*cur_qqq1++, (unsigned)*cur_qqq2++);
                break;
            case OpType::PUSH_CONDITION:
                fprintf(file, "PUSH_CONDITION if b%u\n", (unsigned)*cur_bit_cond++);
                break;
            case OpType::DEBUG_PRINT_EMPTY_IF:
                fprintf(file, "DEBUG_PRINT if b%u\n", (unsigned)*cur_bit_cond++);
                break;
            case OpType::BIT_STORE0_IF:
                fprintf(file, "BIT_STORE0 b%u if b%u\n", (unsigned)*cur_bit_targ++, (unsigned)*cur_bit_cond++);
                break;
            case OpType::BIT_STORE1_IF:
                fprintf(file, "BIT_STORE1 b%u if b%u\n", (unsigned)*cur_bit_targ++, (unsigned)*cur_bit_cond++);
                break;
            case OpType::BIT_INVERT_IF:
                fprintf(file, "BIT_INVERT b%u if b%u\n", (unsigned)*cur_bit_targ++, (unsigned)*cur_bit_cond++);
                break;
            case OpType::DEBUG_PRINT_C_IF:
                fprintf(file, "DEBUG_PRINT b%u if b%u\n", (unsigned)*cur_bit_targ++, (unsigned)*cur_bit_cond++);
                break;
            case OpType::Z_IF:
                fprintf(file, "Z q%u if b%u\n", (unsigned)*cur_q0++, (unsigned)*cur_bit_cond++);
                break;
            case OpType::X_IF:
                fprintf(file, "X q%u if b%u\n", (unsigned)*cur_q0++, (unsigned)*cur_bit_cond++);
                break;
            case OpType::R_IF:
                fprintf(file, "R q%u if b%u\n", (unsigned)*cur_q0++, (unsigned)*cur_bit_cond++);
                break;
            case OpType::DEBUG_PRINT_Q_IF:
                fprintf(file, "DEBUG_PRINT q%u if b%u\n", (unsigned)*cur_q0++, (unsigned)*cur_bit_cond++);
                break;
            case OpType::HMR_IF:
                fprintf(
                    file,
                    "HMR q%u b%u if b%u\n",
                    (unsigned)*cur_q0++,
                    (unsigned)*cur_bit_targ++,
                    (unsigned)*cur_bit_cond++);
                break;
            case OpType::CZ_IF:
                fprintf(
                    file, "CZ q%u q%u if b%u\n", (unsigned)*cur_qq0++, (unsigned)*cur_qq1++, (unsigned)*cur_bit_cond++);
                break;
            case OpType::CX_IF:
                fprintf(
                    file, "CX q%u q%u if b%u\n", (unsigned)*cur_qq0++, (unsigned)*cur_qq1++, (unsigned)*cur_bit_cond++);
                break;
            case OpType::SWAP_IF:
                fprintf(
                    file,
                    "SWAP q%u q%u if b%u\n",
                    (unsigned)*cur_qq0++,
                    (unsigned)*cur_qq1++,
                    (unsigned)*cur_bit_cond++);
                break;
            case OpType::CCZ_IF:
                fprintf(
                    file,
                    "CCZ q%u q%u q%u if b%u\n",
                    (unsigned)*cur_qqq0++,
                    (unsigned)*cur_qqq1++,
                    (unsigned)*cur_qqq2++,
                    (unsigned)*cur_bit_cond++);
                break;
            case OpType::CCX_IF:
                fprintf(
                    file,
                    "CCX q%u q%u q%u if b%u\n",
                    (unsigned)*cur_qqq0++,
                    (unsigned)*cur_qqq1++,
                    (unsigned)*cur_qqq2++,
                    (unsigned)*cur_bit_cond++);
                break;
            case OpType::Z_POW_IF:
                fprintf(file, "Z_POW q%u ", (unsigned)*cur_q0++);
                (*cur_angle++).write_decimal_half_turns_to(file);
                fprintf(file, " if b%u\n", (unsigned)*cur_bit_cond++);
                break;
            default: {
                std::stringstream ss2;
                ss2 << "Unknown op type: " << op_types[k];
                throw std::invalid_argument(ss2.str());
            }
        }
    }
    if (ferror(file)) {
        throw std::invalid_argument("Error while writing file.");
    }
}

size_t Circuit::compute_max_condition_depth() const {
    size_t cur_depth = 0;
    size_t max_depth = 0;
    for (size_t k = 0; k < num_ops; k++) {
        if (op_types[k] == OpType::PUSH_CONDITION) {
            cur_depth++;
            max_depth = std::max(max_depth, cur_depth);
        } else if (op_types[k] == OpType::POP_CONDITION) {
            cur_depth -= cur_depth > 0;
        }
    }
    return max_depth;
}

size_t Circuit::reaction_depth() const {
    std::vector<Op> ops;
    iter_ops([&](Op op) {
        ops.push_back(op);
    });
    return compute_reaction_depth(num_qubits, num_bits, ops);
}

size_t Circuit::max_magic() const {
    return num_qqq_ops;
}

size_t Circuit::max_t() const {
    size_t total = 0;
    for (size_t k = 0; k < num_angle_ops; k++) {
        total += angles[k].is_multiple_of_45_degrees() && !angles[k].is_multiple_of_90_degrees();
    }
    return total;
}

size_t Circuit::max_rotations() const {
    size_t total = 0;
    for (size_t k = 0; k < num_angle_ops; k++) {
        total += !angles[k].is_multiple_of_45_degrees();
    }
    return total;
}

void Circuit::write_kmx_to(std::ostream &out) const {
    bool has_output = false;
    for (size_t k = 0; k < register_data.size(); k++) {
        const auto &reg = register_data[k];
        for (size_t k2 = 0; k2 < reg.contents.size(); k2++) {
            if (has_output) {
                out << '\n';
            }
            const auto &qb = reg.contents[k2];
            out << "APPEND_TO_REGISTER ";
            if (qb.is_qubit()) {
                out << (QubitId)qb;
            } else {
                out << (BitId)qb;
            }
            out << " r" << k;
            has_output = true;
        }
        if (reg.contents.empty() || !reg.name.empty()) {
            if (has_output) {
                out << '\n';
            }
            out << "REGISTER r" << k;
            if (!reg.name.empty()) {
                out << " \"";
                for (char c : reg.name) {
                    char e = escape_char_for(c);
                    if (e) {
                        out << '\\';
                        out << e;
                    } else {
                        out << c;
                    }
                }
                out << '"';
            }
            has_output = true;
        }
    }

    uint32_t *cur_qqq0 = qqq0;
    uint32_t *cur_qqq1 = qqq1;
    uint32_t *cur_qqq2 = qqq2;
    uint32_t *cur_qq0 = qq0;
    uint32_t *cur_qq1 = qq1;
    uint32_t *cur_q0 = q0;
    uint32_t *cur_bit_cond = bc;
    uint32_t *cur_bit_targ = b0;
    FixedPrecisionAngle128 *cur_angle = angles;
    for (size_t k = 0; k < num_ops; k++) {
        if (has_output) {
            out << '\n';
        }
        has_output = true;
        switch (op_types[k]) {
            case OpType::NEG:
                out << "NEG";
                break;
            case OpType::NEG_IF:
                out << "NEG if b" << *cur_bit_cond++;
                break;
            case OpType::POP_CONDITION:
                out << "POP_CONDITION";
                break;
            case OpType::DEBUG_PRINT_EMPTY:
                out << "DEBUG_PRINT";
                break;
            case OpType::BIT_STORE0:
                out << "BIT_STORE0 b" << *cur_bit_targ++;
                break;
            case OpType::BIT_STORE1:
                out << "BIT_STORE1 b" << *cur_bit_targ++;
                break;
            case OpType::BIT_INVERT:
                out << "BIT_INVERT b" << *cur_bit_targ++;
                break;
            case OpType::DEBUG_PRINT_C:
                out << "DEBUG_PRINT b" << *cur_bit_targ++;
                break;
            case OpType::Z:
                out << "Z q" << *cur_q0++;
                break;
            case OpType::X:
                out << "X q" << *cur_q0++;
                break;
            case OpType::R:
                out << "R q" << *cur_q0++;
                break;
            case OpType::DEBUG_PRINT_Q:
                out << "DEBUG_PRINT q" << *cur_q0++;
                break;
            case OpType::HMR:
                out << "HMR q" << *cur_q0++ << " b" << *cur_bit_targ++;
                break;
            case OpType::CZ:
                out << "CZ q" << *cur_qq0++ << " q" << *cur_qq1++;
                break;
            case OpType::CX:
                out << "CX q" << *cur_qq0++ << " q" << *cur_qq1++;
                break;
            case OpType::SWAP:
                out << "SWAP q" << *cur_qq0++ << " q" << *cur_qq1++;
                break;
            case OpType::CCZ:
                out << "CCZ q" << *cur_qqq0++ << " q" << *cur_qqq1++ << " q" << *cur_qqq2++;
                break;
            case OpType::CCX:
                out << "CCX q" << *cur_qqq0++ << " q" << *cur_qqq1++ << " q" << *cur_qqq2++;
                break;
            case OpType::PUSH_CONDITION:
                out << "PUSH_CONDITION if b" << *cur_bit_cond++;
                break;
            case OpType::DEBUG_PRINT_EMPTY_IF:
                out << "DEBUG_PRINT if b" << *cur_bit_cond++;
                break;
            case OpType::BIT_STORE0_IF:
                out << "BIT_STORE0 b" << *cur_bit_targ++ << " if b" << *cur_bit_cond++;
                break;
            case OpType::BIT_STORE1_IF:
                out << "BIT_STORE1 b" << *cur_bit_targ++ << " if b" << *cur_bit_cond++;
                break;
            case OpType::BIT_INVERT_IF:
                out << "BIT_INVERT b" << *cur_bit_targ++ << " if b" << *cur_bit_cond++;
                break;
            case OpType::DEBUG_PRINT_C_IF:
                out << "DEBUG_PRINT b" << *cur_bit_targ++ << " if b" << *cur_bit_cond++;
                break;
            case OpType::Z_IF:
                out << "Z q" << *cur_q0++ << " if b" << *cur_bit_cond++;
                break;
            case OpType::Z_POW:
                out << "Z_POW q" << *cur_q0++ << " " << (*cur_angle++).to_decimal_half_turns();
                break;
            case OpType::Z_POW_IF:
                out << "Z_POW q" << *cur_q0++ << " " << (*cur_angle++).to_decimal_half_turns() << " if b"
                    << *cur_bit_cond++;
                break;
            case OpType::X_IF:
                out << "X q" << *cur_q0++ << " if b" << *cur_bit_cond++;
                break;
            case OpType::R_IF:
                out << "R q" << *cur_q0++ << " if b" << *cur_bit_cond++;
                break;
            case OpType::DEBUG_PRINT_Q_IF:
                out << "DEBUG_PRINT q" << *cur_q0++ << " if b" << *cur_bit_cond++;
                break;
            case OpType::HMR_IF:
                out << "HMR q" << *cur_q0++ << " b" << *cur_bit_targ++ << " if b" << *cur_bit_cond++;
                break;
            case OpType::CZ_IF:
                out << "CZ q" << *cur_qq0++ << " q" << *cur_qq1++ << " if b" << *cur_bit_cond++;
                break;
            case OpType::CX_IF:
                out << "CX q" << *cur_qq0++ << " q" << *cur_qq1++ << " if b" << *cur_bit_cond++;
                break;
            case OpType::SWAP_IF:
                out << "SWAP q" << *cur_qq0++ << " q" << *cur_qq1++ << " if b" << *cur_bit_cond++;
                break;
            case OpType::CCZ_IF:
                out << "CCZ q" << *cur_qqq0++ << " q" << *cur_qqq1++ << " q" << *cur_qqq2++ << " if b"
                    << *cur_bit_cond++;
                break;
            case OpType::CCX_IF:
                out << "CCX q" << *cur_qqq0++ << " q" << *cur_qqq1++ << " q" << *cur_qqq2++ << " if b"
                    << *cur_bit_cond++;
                break;
            default: {
                std::stringstream ss2;
                ss2 << "Circuit::write_kmx_to: Unknown op type: " << op_types[k];
                throw std::invalid_argument(ss2.str());
            }
        }
    }
}

size_t Circuit::compute_num_bits() const {
    uint32_t max_bit = 0;
    bool saw = false;
    saw |= num_c_ops > 0;
    max_bit = std::max(max_bit, max_u32(bc, num_c_ops));
    saw |= num_b_ops > 0;
    max_bit = std::max(max_bit, max_u32(b0, num_b_ops));
    for (const auto &r : register_data) {
        for (const auto &e : r.contents) {
            if (e.is_bit()) {
                saw = true;
                max_bit = std::max(max_bit, e.untagged_id());
            }
        }
    }
    return max_bit + saw;
}

size_t Circuit::compute_num_qubits() const {
    uint32_t max_qubit = 0;
    bool saw = num_qqq_ops > 0 || num_qq_ops > 0 || num_q_ops > 0;
    max_qubit = std::max(max_qubit, max_u32(qqq0, num_qqq_ops));
    max_qubit = std::max(max_qubit, max_u32(qqq1, num_qqq_ops));
    max_qubit = std::max(max_qubit, max_u32(qqq2, num_qqq_ops));
    max_qubit = std::max(max_qubit, max_u32(qq0, num_qq_ops));
    max_qubit = std::max(max_qubit, max_u32(qq1, num_qq_ops));
    max_qubit = std::max(max_qubit, max_u32(q0, num_q_ops));
    for (const auto &r : register_data) {
        for (const auto &e : r.contents) {
            if (e.is_qubit()) {
                max_qubit = std::max(max_qubit, e.untagged_id());
                saw = true;
            }
        }
    }
    return max_qubit + saw;
}

size_t Circuit::compute_num_qqq() const {
    size_t total = 0;
    for (size_t k = 0; k < num_ops; k++) {
        total += ((uint8_t)op_types[k] & kickmix::OP_HAS_3Q) == kickmix::OP_HAS_3Q;
    }
    return total;
}

size_t Circuit::compute_num_qq() const {
    size_t total = 0;
    for (size_t k = 0; k < num_ops; k++) {
        total += ((uint8_t)op_types[k] & kickmix::OP_HAS_3Q) == kickmix::OP_HAS_2Q;
    }
    return total;
}

size_t Circuit::compute_num_q() const {
    size_t total = 0;
    for (size_t k = 0; k < num_ops; k++) {
        total += ((uint8_t)op_types[k] & kickmix::OP_HAS_3Q) == kickmix::OP_HAS_1Q;
    }
    return total;
}

size_t Circuit::compute_num_bit_cond() const {
    size_t total = 0;
    for (size_t k = 0; k < num_ops; k++) {
        total += ((uint8_t)op_types[k] & kickmix::OP_HAS_COND) == kickmix::OP_HAS_COND;
    }
    return total;
}

size_t Circuit::compute_num_bit_targ() const {
    size_t total = 0;
    for (size_t k = 0; k < num_ops; k++) {
        total += ((uint8_t)op_types[k] & kickmix::OP_HAS_1B) == kickmix::OP_HAS_1B;
    }
    return total;
}

size_t Circuit::compute_num_angles() const {
    size_t total = 0;
    for (size_t k = 0; k < num_ops; k++) {
        total += (op_types[k] == OpType::Z_POW) || (op_types[k] == OpType::Z_POW_IF);
    }
    return total;
}

void Circuit::validate_and_compute_stats() {
    if ((intptr_t)qqq0 & 31) {
        throw std::invalid_argument("qqq0 isn't aligned to a 32 byte boundary.");
    }
    if ((intptr_t)qqq1 & 31) {
        throw std::invalid_argument("qqq1 isn't aligned to a 32 byte boundary.");
    }
    if ((intptr_t)qqq2 & 31) {
        throw std::invalid_argument("qqq2 isn't aligned to a 32 byte boundary.");
    }
    if ((intptr_t)qq0 & 31) {
        throw std::invalid_argument("qq0 isn't aligned to a 32 byte boundary.");
    }
    if ((intptr_t)qq1 & 31) {
        throw std::invalid_argument("qq1 isn't aligned to a 32 byte boundary.");
    }
    if ((intptr_t)q0 & 31) {
        throw std::invalid_argument("q0 isn't aligned to a 32 byte boundary.");
    }
    if ((intptr_t)bc & 31) {
        throw std::invalid_argument("bc isn't aligned to a 32 byte boundary.");
    }
    if ((intptr_t)b0 & 31) {
        throw std::invalid_argument("b0 isn't aligned to a 32 byte boundary.");
    }
    if ((intptr_t)op_types & 31) {
        throw std::invalid_argument("op_types isn't aligned to a 32 byte boundary.");
    }
    if ((intptr_t)angles & 31) {
        throw std::invalid_argument("angles isn't aligned to a 32 byte boundary.");
    }
    if (compute_has_qubit_control_collision()) {
        throw std::invalid_argument(describe_qubit_control_collision());
    }

    num_qubits = compute_num_qubits();
    num_bits = compute_num_bits();
    if (num_qubits > MAX_NUM_QUBITS) {
        std::stringstream ss;
        ss << "num_qubits=" << num_qubits << " > MAX_NUM_QUBITS=" << MAX_NUM_QUBITS;
        throw std::invalid_argument(ss.str());
    }
    if (num_bits > MAX_NUM_BITS) {
        std::stringstream ss;
        ss << "num_bits=" << num_bits << " > MAX_NUM_BITS=" << MAX_NUM_BITS;
        throw std::invalid_argument(ss.str());
    }
    if (num_q_ops != compute_num_q()) {
        std::stringstream ss;
        ss << "num_q_ops=" << num_q_ops << " != compute_num_q()=" << compute_num_q();
        throw std::invalid_argument(ss.str());
    }
    if (num_qq_ops != compute_num_qq()) {
        std::stringstream ss;
        ss << "num_qq_ops=" << num_qq_ops << " != compute_num_qq()=" << compute_num_qq();
        throw std::invalid_argument(ss.str());
    }
    if (num_qqq_ops != compute_num_qqq()) {
        std::stringstream ss;
        ss << "num_qqq_ops=" << num_qqq_ops << " != compute_num_qqq()=" << compute_num_qqq();
        throw std::invalid_argument(ss.str());
    }
    if (num_c_ops != compute_num_bit_cond()) {
        std::stringstream ss;
        ss << "num_c_ops=" << num_c_ops << " != compute_num_bit_cond()=" << compute_num_bit_cond();
        throw std::invalid_argument(ss.str());
    }
    if (num_b_ops != compute_num_bit_targ()) {
        std::stringstream ss;
        ss << "num_b_ops=" << num_b_ops << " != compute_num_bit_targ()=" << compute_num_bit_targ();
        throw std::invalid_argument(ss.str());
    }
    if (num_angle_ops != compute_num_angles()) {
        std::stringstream ss;
        ss << "num_angle_ops=" << num_angle_ops << " != compute_num_angles()=" << compute_num_angles();
        throw std::invalid_argument(ss.str());
    }
}

enum class KmbPacketType : uint64_t {
    REGISTER_TARGETS = 0,
    OPERATION_TYPES = 1,
    QUBIT0_DATA_FOR_3_TARGET_QUBIT_GATES = 2,
    QUBIT1_DATA_FOR_3_TARGET_QUBIT_GATES = 3,
    QUBIT2_DATA_FOR_3_TARGET_QUBIT_GATES = 4,
    QUBIT0_DATA_FOR_2_TARGET_QUBIT_GATES = 5,
    QUBIT1_DATA_FOR_2_TARGET_QUBIT_GATES = 6,
    QUBIT0_DATA_FOR_1_TARGET_QUBIT_GATES = 7,
    BIT0_DATA_FOR_1_TARGET_BIT_GATES = 8,
    BIT0_DATA_FOR_1_CONDITION_BIT_GATES = 9,
    ANGLE_DATA_FOR_ROTATION_GATES = 10,
};

template <typename T>
static void write_T_blob_packet(FILE *file, KmbPacketType type, const T *data, size_t count) {
    write_u64_be(file, (uint64_t)type);
    write_u64_be(file, count * sizeof(T));
    fwrite_else_throw(data, count * sizeof(T), file);
}

void Circuit::write_kmb_to(FILE *file) const {
    constexpr uint32_t version = 2;

    // Header
    fwrite_else_throw(KMB_MAGIC_BYTES.data(), KMB_MAGIC_BYTES.size(), file);
    write_u32_be(file, version);
    write_u32_be(file, num_qubits);
    write_u32_be(file, num_bits);
    write_u32_be(file, register_data.size());
    write_u64_be(file, num_ops);
    write_u64_be(file, num_qqq_ops);
    write_u64_be(file, num_qq_ops);
    write_u64_be(file, num_q_ops);
    write_u64_be(file, num_b_ops);
    write_u64_be(file, num_c_ops);
    write_u64_be(file, num_angle_ops);

    // Register data.
    for (const auto &reg : register_data) {
        write_u64_be(file, (uint32_t)KmbPacketType::REGISTER_TARGETS);
        write_u64_be(file, reg.name.size() + reg.contents.size() * sizeof(uint32_t) + 8);
        if (reg.name.size() > UINT32_MAX || reg.contents.size() > UINT32_MAX) {
            throw std::invalid_argument("Too much register data.");
        }
        write_u32_be(file, (uint32_t)reg.name.size());
        write_u32_be(file, (uint32_t)reg.contents.size());
        fwrite_else_throw(reg.name.data(), reg.name.size(), file);
        for (auto e : reg.contents) {
            write_u32_be(file, e.untagged_id() | (e.is_qubit() ? 0x80000000 : 0));
        }
    }

    // Operation type data.
    write_u64_be(file, (uint32_t)KmbPacketType::OPERATION_TYPES);
    write_u64_be(file, num_ops);
    if (num_ops) {
        fwrite_else_throw(op_types, num_ops, file);
    }

    // Operation target data.
    write_T_blob_packet<uint32_t>(file, KmbPacketType::QUBIT0_DATA_FOR_3_TARGET_QUBIT_GATES, qqq0, num_qqq_ops);
    write_T_blob_packet<uint32_t>(file, KmbPacketType::QUBIT1_DATA_FOR_3_TARGET_QUBIT_GATES, qqq1, num_qqq_ops);
    write_T_blob_packet<uint32_t>(file, KmbPacketType::QUBIT2_DATA_FOR_3_TARGET_QUBIT_GATES, qqq2, num_qqq_ops);
    write_T_blob_packet<uint32_t>(file, KmbPacketType::QUBIT0_DATA_FOR_2_TARGET_QUBIT_GATES, qq0, num_qq_ops);
    write_T_blob_packet<uint32_t>(file, KmbPacketType::QUBIT1_DATA_FOR_2_TARGET_QUBIT_GATES, qq1, num_qq_ops);
    write_T_blob_packet<uint32_t>(file, KmbPacketType::QUBIT0_DATA_FOR_1_TARGET_QUBIT_GATES, q0, num_q_ops);
    write_T_blob_packet<uint32_t>(file, KmbPacketType::BIT0_DATA_FOR_1_TARGET_BIT_GATES, b0, num_b_ops);
    write_T_blob_packet<uint32_t>(file, KmbPacketType::BIT0_DATA_FOR_1_CONDITION_BIT_GATES, bc, num_c_ops);
    write_T_blob_packet<FixedPrecisionAngle128>(
        file, KmbPacketType::ANGLE_DATA_FOR_ROTATION_GATES, angles, num_angle_ops);
}

static uint32_t read_u32_be(FILE *file) {
    uint32_t result{};
    fread_else_throw(&result, sizeof(uint32_t), file);
    if constexpr (std::endian::native == std::endian::big) {
        result = __builtin_bswap32(result);
    }
    return result;
}

static uint64_t read_u64_be(FILE *file) {
    uint64_t result{};
    fread_else_throw(&result, sizeof(uint64_t), file);
    if constexpr (std::endian::native == std::endian::big) {
        result = __builtin_bswap64(result);
    }
    return result;
}

static void read_u64_be_expect(FILE *file, uint64_t expected, const char *message) {
    uint64_t actual = read_u64_be(file);
    if (actual != expected) {
        throw std::invalid_argument(message);
    }
}

template <typename T>
static void alloc_and_read_T_blob_packet(FILE *file, KmbPacketType type, T **data_out, size_t count) {
    read_u64_be_expect(file, (uint64_t)type, "Payload packets in wrong order.");
    read_u64_be_expect(file, count * sizeof(T), "Size mismatch: actual payload size vs size specified in header.");
    if (count > 0) {
        *data_out = aligned_alloc_32<T>(count);
        fread_else_throw(*data_out, count * sizeof(T), file);
    }
}

Circuit Circuit::from_kmb_file(FILE *file, bool skip_magic) {
    Circuit result{};

    // Header
    if (!skip_magic) {
        for (auto e : KMB_MAGIC_BYTES) {
            if (getc(file) != e) {
                throw std::invalid_argument(
                    "File didn't start with the magic bytes identifying a kickmix binary format file.");
            }
        }
    }
    if (read_u32_be(file) != 2) {
        throw std::invalid_argument("File specifies an unsupported kickmix binary format file version.");
    }
    uint32_t num_qubits = read_u32_be(file);
    uint32_t num_bits = read_u32_be(file);
    uint32_t num_registers = read_u32_be(file);
    result.num_ops = read_u64_be(file);
    result.num_qqq_ops = read_u64_be(file);
    result.num_qq_ops = read_u64_be(file);
    result.num_q_ops = read_u64_be(file);
    result.num_b_ops = read_u64_be(file);
    result.num_c_ops = read_u64_be(file);
    result.num_angle_ops = read_u64_be(file);

    // Register data.
    for (size_t k = 0; k < num_registers; k++) {
        read_u64_be_expect(file, (uint64_t)KmbPacketType::REGISTER_TARGETS, "Expected register target data.");
        uint64_t register_payload_size = read_u64_be(file);
        size_t name_len = read_u32_be(file);
        size_t reg_len = read_u32_be(file);
        if (reg_len * 4 + name_len + 8 != register_payload_size) {
            throw std::invalid_argument("Register data had wrong payload size.");
        }

        result.register_data.push_back({});
        auto &reg = result.register_data.back();

        reg.name.resize(name_len);
        if (name_len) {
            fread_else_throw(reg.name.data(), name_len, file);
        }

        for (size_t k2 = 0; k2 < reg_len; k2++) {
            uint32_t id = read_u32_be(file);
            if (id & uint32_t{0x80000000ull}) {
                reg.contents.push_back(QubitId{id & ~uint32_t{0x80000000ull}});
            } else {
                reg.contents.push_back(BitId{id});
            }
        }
    }

    // Operation type data.
    read_u64_be_expect(file, (uint64_t)KmbPacketType::OPERATION_TYPES, "Expected operation type data.");
    read_u64_be_expect(
        file, result.num_ops, "Operation type payload size was inconsistent with amount specified in header.");
    if (result.num_ops) {
        result.op_types = aligned_alloc_32<OpType>(result.num_ops);
        fread_else_throw((void *)result.op_types, result.num_ops, file);
    }

    // Operation target data.
    alloc_and_read_T_blob_packet<uint32_t>(
        file, KmbPacketType::QUBIT0_DATA_FOR_3_TARGET_QUBIT_GATES, &result.qqq0, result.num_qqq_ops);
    alloc_and_read_T_blob_packet<uint32_t>(
        file, KmbPacketType::QUBIT1_DATA_FOR_3_TARGET_QUBIT_GATES, &result.qqq1, result.num_qqq_ops);
    alloc_and_read_T_blob_packet<uint32_t>(
        file, KmbPacketType::QUBIT2_DATA_FOR_3_TARGET_QUBIT_GATES, &result.qqq2, result.num_qqq_ops);
    alloc_and_read_T_blob_packet<uint32_t>(
        file, KmbPacketType::QUBIT0_DATA_FOR_2_TARGET_QUBIT_GATES, &result.qq0, result.num_qq_ops);
    alloc_and_read_T_blob_packet<uint32_t>(
        file, KmbPacketType::QUBIT1_DATA_FOR_2_TARGET_QUBIT_GATES, &result.qq1, result.num_qq_ops);
    alloc_and_read_T_blob_packet<uint32_t>(
        file, KmbPacketType::QUBIT0_DATA_FOR_1_TARGET_QUBIT_GATES, &result.q0, result.num_q_ops);
    alloc_and_read_T_blob_packet<uint32_t>(
        file, KmbPacketType::BIT0_DATA_FOR_1_TARGET_BIT_GATES, &result.b0, result.num_b_ops);
    alloc_and_read_T_blob_packet<uint32_t>(
        file, KmbPacketType::BIT0_DATA_FOR_1_CONDITION_BIT_GATES, &result.bc, result.num_c_ops);
    alloc_and_read_T_blob_packet<FixedPrecisionAngle128>(
        file, KmbPacketType::ANGLE_DATA_FOR_ROTATION_GATES, &result.angles, result.num_angle_ops);

    if (fgetc(file) != EOF) {
        throw std::invalid_argument("Failed to read kmb data from file: data leftover.");
    }
    result.validate_and_compute_stats();
    if (result.num_bits != num_bits) {
        throw std::invalid_argument("num_bits from header wasn't one more than maximum used bit id (else 0).");
    }
    if (result.num_qubits != num_qubits) {
        throw std::invalid_argument("num_qubits from header wasn't one more than maximum used qubit id (else 0).");
    }
    return result;
}

std::string Op::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

std::string Op::kmx_str() const {
    std::stringstream ss;
    OpType printed_kind = kind;
    if ((kind & OP_HAS_COND) && kind != OpType::PUSH_CONDITION) {
        printed_kind = (OpType)((uint8_t)kind & ~(uint8_t)OP_HAS_COND);
    }
    if (printed_kind == OpType::DEBUG_PRINT_Q || printed_kind == OpType::DEBUG_PRINT_C ||
        printed_kind == OpType::DEBUG_PRINT_EMPTY) {
        ss << "DEBUG_PRINT";
    } else {
        ss << printed_kind;
    }
    if (q_control2.is_qubit()) {
        ss << " " << q_control2.qubit();
    }
    if (q_control1.is_qubit()) {
        ss << " " << q_control1.qubit();
    }
    if (q_target.is_qubit()) {
        ss << " " << q_target.qubit();
    }
    if (c_target.is_bit()) {
        ss << " " << c_target.bit();
    }
    if (kind == OpType::Z_POW || kind == OpType::Z_POW_IF) {
        ss << " " << angle.to_decimal_half_turns();
    }
    if (c_condition.is_bit()) {
        ss << " if " << c_condition.bit();
    }
    return ss.str();
}

std::ostream &kickmix::operator<<(std::ostream &out, const Op &op) {
    out << "Op{\n";
    out << "    .kind=" << op.kind << ",\n";
    if (op.q_control2.is_qubit()) {
        out << "    .q_control2=" << op.q_control2 << ",\n";
    }
    if (op.q_control1.is_qubit()) {
        out << "    .q_control1=" << op.q_control1 << ",\n";
    }
    if (op.q_target.is_qubit()) {
        out << "    .q_target=" << op.q_target << ",\n";
    }
    if (op.c_target.is_bit()) {
        out << "    .c_target=" << op.c_target << ",\n";
    }
    if (op.kind == OpType::Z_POW || op.kind == OpType::Z_POW_IF || op.angle) {
        out << "    .angle=" << op.angle << ",\n";
    }
    if (op.c_condition.is_bit()) {
        out << "    .c_condition=" << op.c_condition << ",\n";
    }
    out << "}";
    return out;
}
