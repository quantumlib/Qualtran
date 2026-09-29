#include "mutable_circuit.h"

#include <iostream>
#include <sstream>

#include "kickmix/id/register_id.h"

using namespace kickmix;

struct OpData {
    std::string_view name;
    OpType op_type;
    bool allows_no_condition;
    bool allows_condition;
    bool exists;
};

static constexpr uint8_t name_hash(std::string_view name) {
    uint8_t result = name.size();
    if (name.size() > 0) {
        result += name.back() * 133;
        result ^= name[0] * 22;
    }
    return result & 63;
}

static std::array<OpData, 64> compute_op_data() {
    std::array<OpData, 64> result{};
    for (size_t k = 0; k < 64; k++) {
        result[k].name = "";
        result[k].op_type = (OpType)0xFF;
        result[k].exists = false;
    }
    auto define = [&](const char *name, OpType op_type, bool allows_no_condition, bool allows_condition) {
        auto r = name_hash(name);
        if (result[r].exists) {
            std::stringstream ss;
            ss << "Hash collision: ";
            ss << name;
            ss << " vs ";
            ss << result[r].name;
            throw std::invalid_argument(ss.str());
        }
        result[r].exists = true;
        result[r].name = name;
        result[r].op_type = op_type;
        result[r].allows_no_condition = allows_no_condition;
        result[r].allows_condition = allows_condition;
    };
    define("X", OpType::X, true, true);
    define("CX", OpType::CX, true, true);
    define("CCX", OpType::CCX, true, true);
    define("Z", OpType::Z, true, true);
    define("CZ", OpType::CZ, true, true);
    define("CCZ", OpType::CCZ, true, true);
    define("Z_POW", OpType::Z_POW, true, true);
    define("NEG", OpType::NEG, true, true);
    define("SWAP", OpType::SWAP, true, true);
    define("R", OpType::R, true, true);
    define("HMR", OpType::HMR, true, true);
    define("BIT_INVERT", OpType::BIT_INVERT, true, true);
    define("BIT_STORE0", OpType::BIT_STORE0, true, true);
    define("BIT_STORE1", OpType::BIT_STORE1, true, true);
    define("PUSH_CONDITION", OpType::PUSH_CONDITION, false, true);
    define("POP_CONDITION", OpType::POP_CONDITION, true, false);
    define("REGISTER", (OpType)0xFF, true, false);
    define("APPEND_TO_REGISTER", (OpType)0xFF, true, false);
    define("DEBUG_PRINT", (OpType)0xFF, true, true);
    return result;
}

static std::array<OpData, 64> OP_DATA = compute_op_data();

static uint32_t try_parse_uint32_t_below_max(std::string_view text, std::string_view *error, uint32_t max) {
    if (text.empty()) {
        *error = "missing_number";
        return 0;
    }
    uint64_t result = 0;
    for (char digit : text) {
        if (!(digit >= '0' && digit <= '9')) {
            *error = "bad_digit";
            return 0;
        }
        result *= 10;
        result += digit - '0';
        if (result >= max) {
            *error = "number_too_large";
            return 0;
        }
    }
    return static_cast<uint32_t>(result);
}

static std::string_view read_word(const char *&c, const char *end) {
    while (c != end && (*c == ' ' || *c == '\t')) {
        c++;
    }
    const char *start = c;
    while (c != end && *c != ' ' && *c != '\t' && *c != '\r' && *c != '\n' && *c != '#') {
        c++;
    }
    return std::string_view{start, c};
}

static FixedPrecisionAngle128 read_angle(const char *&c, const char *end) {
    std::string_view word = read_word(c, end);
    return FixedPrecisionAngle128::from_rounded_decimal_half_turns(word);
}

static QubitId read_qubit_id(const char *&c, const char *end, std::string_view *error) {
    std::string_view word = read_word(c, end);
    if (!word.starts_with("q")) {
        *error = "expected_a_qubit_id";
        return QubitId(0);
    }
    return QubitId(try_parse_uint32_t_below_max(word.substr(1), error, MAX_NUM_QUBITS));
}

static uint32_t read_register_id(const char *&c, const char *end, std::string_view *error) {
    std::string_view word = read_word(c, end);
    if (!word.starts_with("r")) {
        *error = "expected_a_register_id";
        return 0;
    }
    return try_parse_uint32_t_below_max(word.substr(1), error, UINT32_MAX);
}

static std::string read_register_name(const char *&c, const char *end, std::string_view *error) {
    while (c != end && (*c == ' ' || *c == '\t')) {
        c++;
    }
    if (c == end) {
        return "";
    }
    if (*c != '"') {
        throw std::invalid_argument("Expected register name to be unspecified or start with \".");
    }
    c++;

    std::string name;
    while (c != end && *c != '"') {
        if (*c == '\\') {
            c++;
            if (c == end) {
                *error =
                    "Unknown escape sequence. Known sequences are \\n=newline, \\r=linefeed, \\P=#, \\Q=\", \\B=\\.";
                return "";
            } else if (*c == 'P') {
                name.push_back('#');
            } else if (*c == 'B') {
                name.push_back('\\');
            } else if (*c == 'Q') {
                name.push_back('"');
            } else if (*c == 'n') {
                name.push_back('\n');
            } else if (*c == 'r') {
                name.push_back('\r');
            } else {
                *error =
                    "Unknown escape sequence. Known sequences are \\n=newline, \\r=linefeed, \\Q=\", \\B=\\, \\P=#";
                return "";
            }
        } else {
            name.push_back(*c);
        }
        c++;
    }
    if (c == end) {
        *error = "Quoted text ended before a second '\"' was found (did you forget to escape '#' as '\\P')?).";
        return "";
    }
    c++;

    return name;
}

static BitId read_bit_id(const char *&c, const char *end, std::string_view *error) {
    std::string_view word = read_word(c, end);
    if (!word.starts_with("b")) {
        *error = "expected_a_bit_id";
        return BitId(0);
    }
    return BitId(try_parse_uint32_t_below_max(word.substr(1), error, MAX_NUM_BITS));
}

static BitIdOrFalse try_read_cond(const char *&c, const char *end, std::string_view *error) {
    std::string_view word = read_word(c, end);
    if (word == "if") {
        return read_bit_id(c, end, error);
    } else if (word.empty()) {
        return {};
    } else {
        *error = "unexpected_term";
        return BitId(0);
    }
}

static void parse_register_append_line(
    MutableCircuit &builder, const char *c, const char *end, std::string_view *error) {
    auto w = read_word(c, end);
    RegisterId r = {read_register_id(c, end, error)};

    if (w.starts_with("q")) {
        QubitId q(try_parse_uint32_t_below_max(w.substr(1), error, MAX_NUM_QUBITS));
        if (error->empty()) {
            while (builder.register_data.size() <= r.id) {
                builder.register_data.push_back({});
            }
            builder.register_data[r.id].contents.push_back(q);
        }
    } else if (w.starts_with("b")) {
        BitId b(try_parse_uint32_t_below_max(w.substr(1), error, MAX_NUM_BITS));
        if (error->empty()) {
            while (builder.register_data.size() <= r.id) {
                builder.register_data.push_back({});
            }
            builder.register_data[r.id].contents.push_back(b);
        }
    } else {
        *error = "expected_a_bit_or_qubit_id";
    }
}
static void parse_debug_print_line(MutableCircuit &builder, const char *c, const char *end, std::string_view *error) {
    std::vector<std::string_view> words;
    while (true) {
        words.push_back(read_word(c, end));
        if (words.back().empty()) {
            words.pop_back();
            break;
        }
    }
    BitIdOrFalse cond = {};
    if (words.size() >= 2 && words[words.size() - 2] == "if" && words[words.size() - 1].starts_with("b")) {
        cond = BitId(try_parse_uint32_t_below_max(words.back().substr(1), error, MAX_NUM_BITS));
        words.pop_back();
        words.pop_back();
    }
    if (!error->empty()) {
        return;
    }
    if (words.empty()) {
        if (cond.is_bit()) {
            builder.op_types.push_back(OpType::DEBUG_PRINT_EMPTY_IF);
            builder.bc.push_back(cond.tagged_id);
        } else {
            builder.op_types.push_back(OpType::DEBUG_PRINT_EMPTY);
        }
        return;
    }

    for (auto w : words) {
        if (w.starts_with("q")) {
            QubitId q(try_parse_uint32_t_below_max(w.substr(1), error, MAX_NUM_QUBITS));
            if (error->empty()) {
                builder.q0.push_back(q.tagged_id);
                if (cond.is_bit()) {
                    builder.op_types.push_back(OpType::DEBUG_PRINT_Q_IF);
                    builder.bc.push_back(cond.tagged_id);
                } else {
                    builder.op_types.push_back(OpType::DEBUG_PRINT_Q);
                }
            }
        } else if (w.starts_with("b")) {
            BitId b(try_parse_uint32_t_below_max(w.substr(1), error, MAX_NUM_BITS));
            if (error->empty()) {
                builder.b0.push_back(b.tagged_id);
                if (cond.is_bit()) {
                    builder.op_types.push_back(OpType::DEBUG_PRINT_C_IF);
                    builder.bc.push_back(cond.tagged_id);
                } else {
                    builder.op_types.push_back(OpType::DEBUG_PRINT_C);
                }
            }
        } else if (w.starts_with("r")) {
            RegisterId r = {try_parse_uint32_t_below_max(w.substr(1), error, UINT32_MAX)};
            if (error->empty()) {
                if (r.id < builder.register_data.size()) {
                    for (const auto &e : builder.register_data[r.id].contents) {
                        if (e.is_qubit()) {
                            builder.q0.push_back(e.tagged_id);
                            if (cond.is_bit()) {
                                builder.op_types.push_back(OpType::DEBUG_PRINT_Q_IF);
                                builder.bc.push_back(cond.tagged_id);
                            } else {
                                builder.op_types.push_back(OpType::DEBUG_PRINT_Q);
                            }
                        } else {
                            builder.b0.push_back(e.tagged_id);
                            if (cond.is_bit()) {
                                builder.op_types.push_back(OpType::DEBUG_PRINT_C_IF);
                                builder.bc.push_back(cond.tagged_id);
                            } else {
                                builder.op_types.push_back(OpType::DEBUG_PRINT_C);
                            }
                        }
                    }
                }
            }
        } else {
            *error = "expected_a_bit_or_qubit_or_reg_id";
        }
        if (!error->empty()) {
            return;
        }
    }
}

void MutableCircuit::append_from_kmx_line(std::string_view kmx_line) {
    const char *end = kmx_line.end();
    const char *c = kmx_line.data();

    std::string_view op_name = read_word(c, end);
    if (op_name.empty()) {
        return;  // Empty line.
    }

    uint8_t hash = name_hash(op_name);
    const auto &data = OP_DATA[hash];
    if (!data.exists || op_name != data.name) {
        throw std::invalid_argument(
            "Failed to parse kmx line (unknown operation name): '" + std::string(kmx_line) + "'");
    }

    std::string_view error{};

    // Handle special cases.
    if (data.op_type == (OpType)0xFF) {
        if (hash == name_hash("APPEND_TO_REGISTER")) {
            parse_register_append_line(*this, c, end, &error);
        } else if (hash == name_hash("REGISTER")) {
            RegisterId r = {read_register_id(c, end, &error)};
            std::string name = read_register_name(c, end, &error);
            if (error.empty()) {
                while (register_data.size() <= r.id) {
                    register_data.push_back({});
                }
                register_data[r.id].name = name;
            }
        } else if (hash == name_hash("DEBUG_PRINT")) {
            parse_debug_print_line(*this, c, end, &error);
        } else {
            error = "unknown_operation";
        }
        if (!error.empty()) {
            throw std::invalid_argument(
                "Failed to parse kmx line (" + std::string(error) + "): '" + std::string(kmx_line) + "'");
        }
        return;
    }
    bool has_angle = data.op_type == OpType::Z_POW || data.op_type == OpType::Z_POW_IF;

    Op op{
        .kind = data.op_type,
        .angle = {},
        .q_control2 = {},
        .q_control1 = {},
        .q_target = {},
        .c_target = {},
        .c_condition = {},
    };
    switch (op.kind & OP_HAS_3Q) {
        case OP_HAS_3Q:
            op.q_control2 = read_qubit_id(c, end, &error);
        case OP_HAS_2Q:
            op.q_control1 = read_qubit_id(c, end, &error);
        case OP_HAS_1Q:
            op.q_target = read_qubit_id(c, end, &error);
        default:
            break;
    }
    if (op.kind & OP_HAS_1B) {
        op.c_target = read_bit_id(c, end, &error);
    }
    if (has_angle) {
        op.angle = read_angle(c, end);
    }
    op.c_condition = try_read_cond(c, end, &error);
    if (!op.c_condition.is_bit() && !data.allows_no_condition) {
        error = "missing_condition";
    }
    if (op.c_condition.is_bit() && !data.allows_condition) {
        error = "unexpected_condition";
    }
    if (!error.empty()) {
        throw std::invalid_argument(
            "Failed to parse kmx line (" + std::string(error) + "): '" + std::string(kmx_line) + "'");
    }
    if (op.c_condition.is_bit()) {
        op.kind = op.kind | OP_HAS_COND;
    }
    switch (op.kind & OP_HAS_3Q) {
        case OP_HAS_3Q:
            qqq0.push_back(op.q_control2.tagged_id);
            qqq1.push_back(op.q_control1.tagged_id);
            qqq2.push_back(op.q_target.tagged_id);
            break;
        case OP_HAS_2Q:
            qq0.push_back(op.q_control1.tagged_id);
            qq1.push_back(op.q_target.tagged_id);
            break;
        case OP_HAS_1Q:
            q0.push_back(op.q_target.tagged_id);
            break;
        default:
            break;
    }
    if (op.kind & OP_HAS_1B) {
        b0.push_back(op.c_target.tagged_id);
    }
    if (has_angle) {
        angles.push_back(op.angle);
    }
    if (op.kind & OP_HAS_COND) {
        bc.push_back(op.c_condition.tagged_id);
    }
    op_types.push_back(op.kind);
}

void MutableCircuit::append_from_kmx_file(FILE *file) {
    std::string buffer;
    while (true) {
        int c = getc_unlocked(file);
        if (c == EOF || c == '\n' || c == '\r') {
            append_from_kmx_line(buffer);
            if (c == EOF) {
                return;
            }
            buffer.clear();
        } else {
            buffer.push_back(c);
        }
    }
}

void MutableCircuit::append_from_kmx_text(std::string_view kmx_text) {
    const char *end = kmx_text.end();
    const char *c = kmx_text.data();
    while (c < end) {
        // Scan for end of line (or beginning of comment).
        const char *line_start = c;
        while (c < end && *c != '\r' && *c != '\n' && *c != '#') {
            c++;
        }

        // Parse the instruction that's present (if any).
        append_from_kmx_line(std::string_view{line_start, c});

        // Move past comment (if present).
        while (c < end && *c != '\r' && *c != '\n') {
            c++;
        }
        // Move to start of next line.
        while (c < end && (*c == '\r' || *c == '\n')) {
            c++;
        }
    }
}

template <typename T>
static T *aligned_alloc_32(size_t count) {
    size_t bytes = count * sizeof(T);
    bytes += size_t{31};
    bytes &= ~size_t{31};
    return (T *)std::aligned_alloc(32, bytes);
}

template <typename T>
static uint32_t max_u32_buf_untagged(const MonotonicArena<T, 32> &buf) {
    uint32_t result = 0;
    buf.iter_spans([&](std::span<T> items) {
        uint32_t t = 0;
        for (auto e : items) {
            t = std::max(t, e & UNTAGGED_MASK);
        }
        result = std::max(result, t);
    });
    return result;
}

size_t MutableCircuit::compute_cur_num_qubits() const {
    uint32_t max_qubit = 0;
    bool has_qubit = false;
    for (const auto &e : register_data) {
        for (const auto &q : e.contents) {
            if (q.is_qubit()) {
                max_qubit = std::max(max_qubit, q.untagged_id());
                has_qubit = true;
            }
        }
    }
    max_qubit = std::max(max_qubit, max_u32_buf_untagged(qqq0));
    max_qubit = std::max(max_qubit, max_u32_buf_untagged(qqq1));
    max_qubit = std::max(max_qubit, max_u32_buf_untagged(qqq2));
    max_qubit = std::max(max_qubit, max_u32_buf_untagged(qq0));
    max_qubit = std::max(max_qubit, max_u32_buf_untagged(qq1));
    max_qubit = std::max(max_qubit, max_u32_buf_untagged(q0));
    size_t result = max_qubit;
    result += has_qubit || qqq0.size() > 0 || qq0.size() > 0 || q0.size() > 0;
    if (result > MAX_NUM_QUBITS) {
        std::stringstream ss;
        ss << "num_qubits=" << result << " > MAX_NUM_QUBITS=" << MAX_NUM_QUBITS;
        throw std::invalid_argument(ss.str());
    }
    return result;
}

Circuit MutableCircuit::to_validated_circuit() const {
    if (qq0.size() != qq1.size()) {
        throw std::invalid_argument("data_qq0.size() != data_qq1.size()");
    }
    if (qqq0.size() != qqq1.size()) {
        throw std::invalid_argument("data_qqq0.size() != data_qqq1.size()");
    }
    if (qqq0.size() != qqq2.size()) {
        throw std::invalid_argument("data_qqq0.size() != data_qqq2.size()");
    }
    Circuit result;
    result.register_data = register_data;

    result.num_ops = op_types.size();
    result.num_qqq_ops = qqq0.size();
    result.num_qq_ops = qq0.size();
    result.num_q_ops = q0.size();
    result.num_c_ops = bc.size();
    result.num_b_ops = b0.size();
    result.num_angle_ops = angles.size();

    // Copy qubit and bit data into contiguous memory blocks.
    result.op_types = aligned_alloc_32<kickmix::OpType>(op_types.size());
    result.qqq0 = aligned_alloc_32<uint32_t>(qqq0.size());
    result.qqq1 = aligned_alloc_32<uint32_t>(qqq1.size());
    result.qqq2 = aligned_alloc_32<uint32_t>(qqq2.size());
    result.qq0 = aligned_alloc_32<uint32_t>(qq0.size());
    result.qq1 = aligned_alloc_32<uint32_t>(qq1.size());
    result.q0 = aligned_alloc_32<uint32_t>(q0.size());
    result.bc = aligned_alloc_32<uint32_t>(bc.size());
    result.b0 = aligned_alloc_32<uint32_t>(b0.size());
    result.angles = aligned_alloc_32<FixedPrecisionAngle128>(angles.size());
    op_types.memcpy_into(result.op_types);
    qqq0.memcpy_into(result.qqq0);
    qqq1.memcpy_into(result.qqq1);
    qqq2.memcpy_into(result.qqq2);
    qq0.memcpy_into(result.qq0);
    qq1.memcpy_into(result.qq1);
    q0.memcpy_into(result.q0);
    bc.memcpy_into(result.bc);
    b0.memcpy_into(result.b0);
    angles.memcpy_into(result.angles);

    // Delete tags.
    for (size_t k = 0; k < result.num_qqq_ops; k++) {
        result.qqq0[k] &= UNTAGGED_MASK;
        result.qqq1[k] &= UNTAGGED_MASK;
        result.qqq2[k] &= UNTAGGED_MASK;
    }
    for (size_t k = 0; k < result.num_qq_ops; k++) {
        result.qq0[k] &= UNTAGGED_MASK;
        result.qq1[k] &= UNTAGGED_MASK;
    }
    for (size_t k = 0; k < result.num_q_ops; k++) {
        result.q0[k] &= UNTAGGED_MASK;
    }
    for (size_t k = 0; k < result.num_c_ops; k++) {
        result.bc[k] &= UNTAGGED_MASK;
    }
    for (size_t k = 0; k < result.num_b_ops; k++) {
        result.b0[k] &= UNTAGGED_MASK;
    }

    result.num_qubits = 0;
    result.num_bits = 0;
    result.validate_and_compute_stats();
    return result;
}

void MutableCircuit::clear() {
    register_data.clear();
    op_types.clear();
    qqq0.clear();
    qqq1.clear();
    qqq2.clear();
    qq0.clear();
    qq1.clear();
    q0.clear();
    bc.clear();
    b0.clear();
    angles.clear();
}

void MutableCircuit::append(const Circuit &other) {
    op_types.push_back_many({other.op_types, other.num_ops});
    qqq0.push_back_many({other.qqq0, other.num_qqq_ops});
    qqq1.push_back_many({other.qqq1, other.num_qqq_ops});
    qqq2.push_back_many({other.qqq2, other.num_qqq_ops});
    qq0.push_back_many({other.qq0, other.num_qq_ops});
    qq1.push_back_many({other.qq1, other.num_qq_ops});
    q0.push_back_many({other.q0, other.num_q_ops});
    bc.push_back_many({other.bc, other.num_c_ops});
    b0.push_back_many({other.b0, other.num_b_ops});
    angles.push_back_many({other.angles, other.num_angle_ops});
}

void MutableCircuit::append(const MutableCircuit &other) {
    op_types.push_back_many(other.op_types);
    qqq0.push_back_many(other.qqq0);
    qqq1.push_back_many(other.qqq1);
    qqq2.push_back_many(other.qqq2);
    qq0.push_back_many(other.qq0);
    qq1.push_back_many(other.qq1);
    q0.push_back_many(other.q0);
    bc.push_back_many(other.bc);
    b0.push_back_many(other.b0);
    angles.push_back_many(other.angles);
}
void MutableCircuit::append_reversed(const MutableCircuit &other) {
    op_types.push_back_many_reversed(other.op_types);
    qqq0.push_back_many_reversed(other.qqq0);
    qqq1.push_back_many_reversed(other.qqq1);
    qqq2.push_back_many_reversed(other.qqq2);
    qq0.push_back_many_reversed(other.qq0);
    qq1.push_back_many_reversed(other.qq1);
    q0.push_back_many_reversed(other.q0);
    bc.push_back_many_reversed(other.bc);
    b0.push_back_many_reversed(other.b0);
    angles.push_back_many_reversed(other.angles);
}
