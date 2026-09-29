#include <iostream>
#include <set>
#include <span>
#include <sstream>

#include "kickmix/circuit/circuit.h"

using namespace kickmix;

static void push_back_u32(std::string &out, uint32_t val) {
    if (!val) {
        out.push_back('0');
        return;
    }

    size_t start = out.size();
    while (val) {
        out.push_back("0123456789"[val % 10]);
        val /= 10;
    }
    size_t end = out.size() - 1;
    while (start < end) {
        std::swap(out[start], out[end]);
        start++;
        end--;
    }
}

std::string Circuit::text_diagram() const {
    std::stringstream ss;
    write_text_diagram_to(ss);
    return ss.str();
}

static std::vector<std::string> generate_text_diagram_lines(const Circuit &circuit, bool use_unicode) {
    constexpr char UNCERTAIN_FILL_MARKER = 1;
    std::vector<char> filler_chars;
    std::vector<std::string> result_lines;
    std::vector<bool> used_ys;
    for (size_t y = 0; y < 2 * circuit.num_qubits + 2; y++) {
        char filler = y > 0 && y <= circuit.num_qubits * 2 && y % 2 == 0 ? UNCERTAIN_FILL_MARKER : ' ';
        filler_chars.push_back(filler);
    }
    result_lines.resize(filler_chars.size());
    used_ys.resize(filler_chars.size());

    constexpr auto q2y = [](QubitOrFalse q) {
        return q.untagged_id() * 2 + 2;
    };

    // Qubit labels.
    {
        size_t max_col_len = 0;
        for (uint32_t q = 0; q < circuit.num_qubits; q++) {
            auto &line = result_lines[q2y(QubitId{q})];
            line.push_back('q');
            push_back_u32(line, q);
            line.push_back(':');
            line.push_back(' ');
            line.push_back(UNCERTAIN_FILL_MARKER);
            max_col_len = std::max(max_col_len, line.size());
        }
        for (auto &line : result_lines) {
            if (line.size() < max_col_len) {
                line.insert(0, max_col_len - line.size(), ' ');
            }
        }
    }

    uint32_t meta_col_next_y = UINT32_MAX;

    std::span<const Op> merge_queue_span;
    QubitOrFalse merge_queue_common_control = {};
    std::set<BitId> read_bits_in_col;
    std::set<BitId> written_bits_in_col;

    const auto flush_col = [&]() {
        read_bits_in_col.clear();
        written_bits_in_col.clear();

        size_t w = 0;
        for (const auto &line : result_lines) {
            w = std::max(w, line.size());
        }

        for (size_t y = 0; y < result_lines.size(); y++) {
            char filler = filler_chars[y];
            auto &line = result_lines[y];
            if (line.size() < w) {
                line.insert(line.size(), w - line.size(), filler);
            }
            line.push_back(filler);
        }

        used_ys.clear();
        used_ys.resize(result_lines.size());
    };

    const auto reserve_and_render_vertical_line = [&](uint32_t min_y, uint32_t max_y, BitIdOrFalse condition) {
        if (min_y < 2) {
            throw std::invalid_argument("min_y < 2");
        }
        bool need_new_col = false;
        if (condition.is_bit()) {
            if (used_ys[min_y - 1] || used_ys[min_y - 2]) {
                need_new_col = true;
            }
        }
        for (size_t y = min_y; y <= max_y; y++) {
            if (used_ys[y]) {
                need_new_col = true;
                break;
            }
        }
        if (need_new_col) {
            flush_col();
        }
        for (size_t y = min_y; y <= max_y; y++) {
            used_ys[y] = true;
        }
        for (size_t y = min_y; y <= max_y; y++) {
            result_lines[y].push_back('|');
        }
        if (condition.is_bit()) {
            used_ys[min_y - 1] = true;
            used_ys[min_y - 2] = true;
            auto &line = result_lines[min_y - 1];
            line.append("if(");
            line.push_back('b');
            push_back_u32(line, condition.untagged_id());
            line.push_back(')');
        }
    };

    std::set<QubitId> used_qs_for_cx_cz_merge;
    const auto flush_merged_queue = [&]() {
        if (merge_queue_span.empty()) {
            return;
        }
        used_qs_for_cx_cz_merge.clear();

        uint32_t min_y = q2y(merge_queue_common_control);
        uint32_t max_y = min_y;
        for (auto op : merge_queue_span) {
            auto y = q2y(op.q_target);
            min_y = std::min(y, min_y);
            max_y = std::max(y, max_y);
        }
        reserve_and_render_vertical_line(min_y, max_y, {});

        result_lines[q2y(merge_queue_common_control)].back() = '@';
        for (const auto &op : merge_queue_span) {
            auto &line = result_lines[q2y(op.q_target)];
            switch (op.kind) {
                case OpType::CX:
                case OpType::CX_IF:
                    filler_chars[q2y(op.q_target)] = '-';
                    line.back() = 'X';
                    break;
                case OpType::CZ:
                case OpType::CZ_IF:
                    line.back() = 'Z';
                    break;
                default:
                    throw std::invalid_argument("Unexpected operation in common control merge queue: " + op.str());
            }
            if (op.c_condition.is_bit()) {
                line.push_back('*');
                line.push_back('*');
                line.push_back('b');
                push_back_u32(line, op.c_condition.untagged_id());
            }
        }
        merge_queue_span = {};
        merge_queue_common_control = {};
    };

    const auto draw_non_qubit_operation = [&](const Op &op) {
        flush_merged_queue();
        if (meta_col_next_y > circuit.num_qubits * 2 + 1) {
            flush_col();
            meta_col_next_y = 1;
        }
        auto &line = result_lines[meta_col_next_y];
        switch (op.kind) {
            case OpType::PUSH_CONDITION:
                line.append("push_cond");
                break;
            case OpType::POP_CONDITION:
                line.append("pop_cond");
                break;
            case OpType::NEG:
            case OpType::NEG_IF:
                line.append("neg");
                break;
            case OpType::BIT_INVERT:
            case OpType::BIT_INVERT_IF:
                line.push_back('b');
                push_back_u32(line, op.c_target.untagged_id());
                line.append("^=1");
                break;
            case OpType::BIT_STORE0:
            case OpType::BIT_STORE0_IF:
                line.push_back('b');
                push_back_u32(line, op.c_target.untagged_id());
                line.append("=0");
                break;
            case OpType::BIT_STORE1:
            case OpType::BIT_STORE1_IF:
                line.push_back('b');
                push_back_u32(line, op.c_target.untagged_id());
                line.append("=1");
                break;
            default:
                line.append("UNEXPECTED_NON_QUBIT_OPERATION");
        }
        if (op.c_condition.is_bit()) {
            line.append(" if b");
            push_back_u32(line, op.c_condition.untagged_id());
        }
        meta_col_next_y += 2;
    };

    const auto draw_cx_cz = [&](const Op &op) {
        if (op.q_control1 != merge_queue_common_control || used_qs_for_cx_cz_merge.contains(op.q_target.qubit())) {
            flush_merged_queue();
        }
        if (merge_queue_span.empty()) {
            merge_queue_span = {&op, 1};
            merge_queue_common_control = op.q_control1;
        } else {
            merge_queue_span = {&merge_queue_span[0], merge_queue_span.size() + 1};
        }
        used_qs_for_cx_cz_merge.insert(op.q_target.qubit());
    };

    for (size_t k = 0; k < circuit.register_data.size(); k++) {
        const auto &reg = circuit.register_data[k];
        for (size_t k2 = 0; k2 < reg.contents.size(); k2++) {
            const auto &qb = reg.contents[k2];
            // Avoid editing and reading a bit in the same column.
            if (qb.is_bit() && read_bits_in_col.contains((BitId)qb)) {
                flush_col();
            }

            if (qb.is_bit()) {
                flush_merged_queue();
                if (meta_col_next_y > circuit.num_qubits * 2 + 1) {
                    flush_col();
                    meta_col_next_y = 1;
                }
                auto &line = result_lines[meta_col_next_y];
                if (reg.name.empty()) {
                    line.append("reg");
                    push_back_u32(line, k);
                } else {
                    line.append(reg.name);
                }
                line.push_back('[');
                push_back_u32(line, k2);
                line.push_back(']');
                line.push_back('=');
                line.push_back('b');
                push_back_u32(line, qb.untagged_id());
                meta_col_next_y += 2;
                continue;
            }
            if (meta_col_next_y != UINT32_MAX) {
                meta_col_next_y = UINT32_MAX;
                flush_col();
            }
            flush_merged_queue();

            uint32_t y = q2y((QubitId)qb);
            reserve_and_render_vertical_line(y, y, {});
            auto &line = result_lines[y];
            line.pop_back();
            if (reg.name.empty()) {
                line.append("reg");
                push_back_u32(line, k);
            } else {
                line.append(reg.name);
            }
            line.push_back('[');
            push_back_u32(line, k2);
            line.push_back(']');
        }
    }

    std::vector<Op> ops;
    circuit.iter_ops([&](const Op &op) {
        ops.push_back(op);
    });
    for (const auto &op : ops) {
        // Avoid editing and reading a bit in the same column.
        if ((op.c_condition.is_bit() && written_bits_in_col.contains(op.c_condition.bit())) ||
            (op.c_target.is_bit() && read_bits_in_col.contains(op.c_target.bit()))) {
            flush_col();
        }

        if (!op.q_target.is_qubit()) {
            draw_non_qubit_operation(op);
            continue;
        }
        if (meta_col_next_y != UINT32_MAX) {
            meta_col_next_y = UINT32_MAX;
            flush_col();
        }
        if (op.kind == OpType::CX || op.kind == OpType::CZ || op.kind == OpType::CX_IF || op.kind == OpType::CZ_IF) {
            draw_cx_cz(op);
            continue;
        }
        flush_merged_queue();

        uint32_t min_y = q2y(op.q_target);
        uint32_t max_y = q2y(op.q_target);
        if (op.q_control1.is_qubit()) {
            min_y = std::min(min_y, q2y(op.q_control1));
            max_y = std::max(max_y, q2y(op.q_control1));
        }
        if (op.q_control2.is_qubit()) {
            min_y = std::min(min_y, q2y(op.q_control2));
            max_y = std::max(max_y, q2y(op.q_control2));
        }

        switch (op.kind) {
            case OpType::X:
            case OpType::Z:
            case OpType::CCX:
            case OpType::CCZ:
            case OpType::X_IF:
            case OpType::Z_IF:
            case OpType::CCX_IF:
            case OpType::CCZ_IF:
                reserve_and_render_vertical_line(min_y, max_y, {});
                break;
            default:
                reserve_and_render_vertical_line(min_y, max_y, op.c_condition);
        }

        if (op.c_target.is_bit()) {
            written_bits_in_col.insert(op.c_target.bit());
            read_bits_in_col.insert(op.c_target.bit());
        }
        if (op.c_condition.is_bit()) {
            read_bits_in_col.insert(op.c_condition.bit());
        }

        auto &line = result_lines[q2y(op.q_target)];
        switch (op.kind) {
            case OpType::R:
            case OpType::R_IF:
                if (filler_chars[q2y(op.q_target)] == UNCERTAIN_FILL_MARKER) {
                    for (auto &e : line) {
                        if (e == UNCERTAIN_FILL_MARKER) {
                            e = ' ';
                        }
                    }
                }
                line.pop_back();
                filler_chars[q2y(op.q_target)] = '-';
                line.append("|0>");
                break;
            case OpType::HMR:
            case OpType::HMR_IF:
                line.pop_back();
                filler_chars[q2y(op.q_target)] = ' ';
                line.append("HMR=b");
                push_back_u32(line, op.c_target.untagged_id());
                break;
            case OpType::X:
            case OpType::X_IF:
                filler_chars[q2y(op.q_target)] = '-';
                line.back() = 'X';
                break;
            case OpType::Z:
            case OpType::Z_IF:
                line.back() = 'Z';
                break;
            case OpType::Z_POW:
            case OpType::Z_POW_IF:
                line.pop_back();
                line.append("Z^");
                line.append(op.angle.to_decimal_half_turns());
                break;
            case OpType::CCX:
            case OpType::CCX_IF:
                filler_chars[q2y(op.q_target)] = '-';
                result_lines[q2y(op.q_control2)].back() = '@';
                result_lines[q2y(op.q_control1)].back() = '@';
                line.back() = 'X';
                break;
            case OpType::CCZ:
            case OpType::CCZ_IF:
                result_lines[q2y(op.q_control2)].back() = '@';
                result_lines[q2y(op.q_control1)].back() = '@';
                line.back() = 'Z';
                break;
            case OpType::SWAP:
            case OpType::SWAP_IF:
                filler_chars[q2y(op.q_target)] = '-';
                filler_chars[q2y(op.q_control1)] = '-';
                result_lines[q2y(op.q_control1)].pop_back();
                result_lines[q2y(op.q_control1)].append("SWAP");
                line.pop_back();
                line.append("SWAP");
                break;
            case OpType::DEBUG_PRINT_EMPTY:
            case OpType::DEBUG_PRINT_Q:
            case OpType::DEBUG_PRINT_C:
            case OpType::DEBUG_PRINT_EMPTY_IF:
            case OpType::DEBUG_PRINT_Q_IF:
            case OpType::DEBUG_PRINT_C_IF:
                break;
            default:
                throw std::invalid_argument("Circuit.write_text_diagram: Unhandled operation: " + op.str());
        }
        switch (op.kind) {
            case OpType::X:
            case OpType::Z:
            case OpType::CCX:
            case OpType::CCZ:
            case OpType::X_IF:
            case OpType::Z_IF:
            case OpType::CCX_IF:
            case OpType::CCZ_IF:
                if (op.c_condition.is_bit()) {
                    line.append("**b");
                    push_back_u32(line, op.c_condition.untagged_id());
                }
                break;
            default:
                break;
        }
    }

    flush_merged_queue();
    flush_col();

    // Overwrite unused space with qubit wire.
    for (auto &line : result_lines) {
        for (auto &e : line) {
            if (e == UNCERTAIN_FILL_MARKER) {
                e = '-';
            }
        }
    }

    // Trim.
    for (auto &line : result_lines) {
        while (line.ends_with(' ')) {
            line.pop_back();
        }
    }
    while (!result_lines.empty() && result_lines.back().empty()) {
        result_lines.pop_back();
    }

    return result_lines;
}
void Circuit::write_text_diagram_to(FILE *out, bool use_unicode) const {
    auto result_lines = generate_text_diagram_lines(*this, use_unicode);

    // Final conversion into characters.
    bool started = false;
    bool was_line = false;
    for (const auto &line : result_lines) {
        if (line.empty() && !started) {
            continue;
        }
        if (!started) {
            if (line.empty()) {
                continue;
            }
            started = true;
        } else {
            fputc('\n', out);
        }
        if (use_unicode) {
            for (char c : line) {
                switch (c) {
                    case '=':
                        fputs("═", out);
                        break;
                    case '-':
                        fputs("─", out);
                        break;
                    case '|':
                        fputs(was_line ? "┼" : "│", out);
                        break;
                    case '@':
                        fputs("●", out);
                        break;
                    case '>':
                        fputs("⟩", out);
                        break;
                    default:
                        fputc(c, out);
                }
                was_line = c == '-';
            }
        } else {
            fwrite(line.data(), 1, line.size(), out);
        }
    }
}
void Circuit::write_text_diagram_to(std::ostream &out_stream, bool use_unicode) const {
    auto result_lines = generate_text_diagram_lines(*this, use_unicode);

    // Final conversion into characters.
    bool started = false;
    bool was_line = false;
    for (const auto &line : result_lines) {
        if (line.empty() && !started) {
            continue;
        }
        if (!started) {
            if (line.empty()) {
                continue;
            }
            started = true;
        } else {
            out_stream << '\n';
        }
        if (use_unicode) {
            for (char c : line) {
                switch (c) {
                    case '=':
                        out_stream << "═";
                        break;
                    case '-':
                        out_stream << "─";
                        break;
                    case '|':
                        out_stream << (was_line ? "┼" : "│");
                        break;
                    case '@':
                        out_stream << "●";
                        break;
                    case '>':
                        out_stream << "⟩";
                        break;
                    default:
                        out_stream << c;
                }
                was_line = c == '-';
            }
        } else {
            out_stream << line;
        }
    }
}
