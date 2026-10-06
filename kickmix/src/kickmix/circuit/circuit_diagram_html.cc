#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>
#include <span>
#include <sstream>

#include "circuit.h"
#include "circuit_util.h"
#include "kickmix/circuit/circuit.h"
#include "kickmix/id/register_id.h"

using namespace kickmix;

std::string Circuit::html_diagram() const {
    std::stringstream ss;
    write_svg_or_html_diagram_to(ss, true);
    return ss.str();
}

std::string Circuit::svg_diagram() const {
    std::stringstream ss;
    write_svg_or_html_diagram_to(ss, false);
    return ss.str();
}

std::string escape_text_for_html(std::string_view src) {
    // From https://stackoverflow.com/a/9907752
    std::stringstream dst;
    for (char ch : src) {
        switch (ch) {
            case '&':
                dst << "&amp;";
                break;
            case '\'':
                dst << "&apos;";
                break;
            case '"':
                dst << "&quot;";
                break;
            case '<':
                dst << "&lt;";
                break;
            case '>':
                dst << "&gt;";
                break;
            default:
                dst << ch;
                break;
        }
    }
    return dst.str();
}

void write_svg_or_html_diagram_to_helper(
    size_t reaction_depth, size_t touched_qubits, const Circuit &circuit, std::ostream &out_stream, bool html) {
    std::vector<Op> operations;
    circuit.iter_ops([&](const Op &op) {
        operations.push_back(op);
    });

    std::stringstream buf_stream;
    if (html) {
        buf_stream << R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Circuit Viewer</title>
</head>

<div>
Controls: mouse wheel to zoom, click and drag to pan.)HTML";
    }

    {
        size_t num_classical_ops = 0;
        size_t num_stabilizer_ops = 0;
        size_t num_ccp = 0;
        size_t num_t = 0;
        size_t num_rz_theta = 0;
        for (const auto &op : operations) {
            switch (op.kind) {
                case OpType::NEG_IF:
                case OpType::NEG:
                case OpType::DEBUG_PRINT_EMPTY_IF:
                case OpType::DEBUG_PRINT_Q_IF:
                case OpType::DEBUG_PRINT_C_IF:
                case OpType::DEBUG_PRINT_EMPTY:
                case OpType::DEBUG_PRINT_Q:
                case OpType::DEBUG_PRINT_C:
                    break;
                case OpType::PUSH_CONDITION:
                case OpType::POP_CONDITION:
                case OpType::BIT_INVERT:
                case OpType::BIT_STORE0:
                case OpType::BIT_STORE1:
                case OpType::X:
                case OpType::Z:
                case OpType::SWAP:
                case OpType::BIT_INVERT_IF:
                case OpType::BIT_STORE0_IF:
                case OpType::BIT_STORE1_IF:
                case OpType::X_IF:
                case OpType::Z_IF:
                case OpType::SWAP_IF:
                    num_classical_ops += 1;
                    break;
                case OpType::CX_IF:
                case OpType::CZ_IF:
                case OpType::R_IF:
                case OpType::HMR_IF:
                case OpType::CX:
                case OpType::CZ:
                case OpType::R:
                case OpType::HMR:
                    num_stabilizer_ops += 1;
                    break;
                case OpType::CCX:
                case OpType::CCZ:
                case OpType::CCX_IF:
                case OpType::CCZ_IF:
                    num_ccp += 1;
                    break;
                case OpType::Z_POW:
                case OpType::Z_POW_IF: {
                    if (!op.angle.is_multiple_of_45_degrees()) {
                        num_rz_theta++;
                    } else if (!op.angle.is_multiple_of_90_degrees()) {
                        num_t++;
                    } else if (!op.angle.is_multiple_of_180_degrees()) {
                        num_stabilizer_ops++;
                    } else {
                        num_classical_ops++;
                    }
                    break;
                }
                default: {
                    std::stringstream ss;
                    ss << "Unhandled diagram circuit stat operation: ";
                    ss << op.kind;
                    throw std::invalid_argument(ss.str());
                }
            }
        }
        if (html) {
            buf_stream << " (Circuit stats:";
            buf_stream << " bit+x+z+swap=" << num_classical_ops;
            buf_stream << ", cx+cz+r+hmr=" << num_stabilizer_ops;
            buf_stream << ", ccx+ccz=<strong>" << num_ccp << "</strong>";
            if (num_t) {
                buf_stream << ", t=<strong>" << num_t << "</strong>";
            }
            if (num_rz_theta) {
                buf_stream << ", rz_theta=<strong>" << num_rz_theta << "</strong>";
            }
            buf_stream << ", reaction depth=<strong>" << reaction_depth << "</strong>";
            buf_stream << ", touched qubits=<strong>" << touched_qubits << "</strong>";
            buf_stream << ")";
        }
    }

    if (html) {
        buf_stream << R"HTML(</div>
<svg id="viewportSvg" style="width: 100%; height: 95vh; border: 1px solid black" viewBox="0 0 2000 800" preserveAspectRatio="none">
</svg>

<script>
    class BoundedObject {
        constructor(start_x, content, far_content) {
            this.start_x = start_x;
            this.content = content;
            this.far_content = far_content;
            this.element = undefined;
            this.far_element = undefined;
        }
        ensureElementExists() {
            if (this.element === undefined) {
                this.element = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                this.element.setHTMLUnsafe(this.content);
            }
        }
        ensureFarElementExists() {
            if (this.far_element === undefined) {
                this.far_element = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                this.far_element.setHTMLUnsafe(this.far_content);
            }
        }
    }
    class WireObject {
        constructor(content) {
            this.content = content;
            this.element = undefined;
        }
        ensureElementExists() {
            if (this.element === undefined) {
                this.element = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                this.element.setHTMLUnsafe(this.content);
            }
        }
    }
    let objects = [
)HTML";
    }
    auto q2y = [](QubitOrFalse q) -> double {
        if (q.is_qubit()) {
            return (double)q.untagged_id() * 16.0 + 16.0;
        }
        return -1.0;
    };

    double x = 0.0;
    std::stringstream svg;
    std::stringstream far_svg;
    svg << std::setprecision(std::numeric_limits<double>::max_digits10);
    far_svg << std::setprecision(std::numeric_limits<double>::max_digits10);

    std::vector<bool> active(circuit.num_qubits, true);
    for (size_t k = operations.size(); k--;) {
        const auto &op = operations[k];
        if (op.kind == OpType::R || op.kind == OpType::R_IF) {
            active[op.q_target.untagged_id()] = false;
        } else if (op.kind == OpType::HMR || op.kind == OpType::HMR_IF) {
            active[op.q_target.untagged_id()] = true;
        }
    }

    std::vector<std::vector<int>> wire_transitions(circuit.num_qubits);
    for (uint32_t q = 0; q < circuit.num_qubits; q++) {
        if (active[q]) {
            wire_transitions[q].push_back(0);
        }
    }

    std::vector<double> last_used(circuit.num_qubits, 0.0);

    // Draw qubit labels.
    if (html) {
        buf_stream << "new BoundedObject(\n    " << x << ",\n    `";
        for (uint32_t q = 0; q < circuit.num_qubits; q++) {
            buf_stream << "<text";
            buf_stream << " x=\"" << x << "\"";
            buf_stream << " y=\"" << q2y(QubitId{q}) << "\"";
            buf_stream << " text-anchor=\"end\"";
            buf_stream << " dominant-baseline=\"middle\"";
            buf_stream << " font-size=\"6\"";
            buf_stream << " font-family=\"monospace\"";
            buf_stream << " fill=\"black\"";
            buf_stream << " >" << QubitId{q} << ":</text>";
        }
        buf_stream << "`,\n    ``,\n),\n";
    }

    uint32_t text_pos = UINT32_MAX;
    const double MIN_COL_WIDTH = 16;
    double cur_col_width = 32;

    for (size_t k = 0; k < circuit.register_data.size(); k++) {
        const auto &reg = circuit.register_data[k];
        for (size_t k2 = 0; k2 < reg.contents.size(); k2++) {
            const auto &qb = reg.contents[k2];
            bool is_classical_command = qb.is_bit();

            uint32_t min_q = UINT32_MAX;
            uint32_t max_q = 0;
            if (qb.is_qubit()) {
                min_q = std::min(qb.untagged_id(), min_q);
                max_q = std::max(qb.untagged_id(), max_q);
            }

            // Track x position of column of operations.
            bool need_new_col = false;
            need_new_col |= qb.is_qubit() && text_pos != UINT32_MAX;
            if (is_classical_command) {
                if (text_pos >= circuit.num_qubits) {
                    need_new_col = true;
                    text_pos = 0;
                } else {
                    text_pos++;
                }
            }
            if (need_new_col) {
                if (!is_classical_command) {
                    text_pos = UINT32_MAX;
                }
                x += cur_col_width;
                cur_col_width = MIN_COL_WIDTH;
            }
            if (min_q <= max_q) {
                for (uint32_t q = min_q; q <= max_q; q++) {
                    if (last_used[q] >= x) {
                        x += cur_col_width;
                        cur_col_width = MIN_COL_WIDTH;
                        break;
                    }
                }
                for (uint32_t q = min_q; q <= max_q; q++) {
                    last_used[q] = x;
                }
            }

            // Draw classical command.
            if (is_classical_command) {
                double y = q2y(QubitId{text_pos}) - 8;
                svg << "<text";
                svg << " x=\"" << x << "\"";
                svg << " y=\"" << y << "\"";
                svg << " text-anchor=\"middle\"";
                svg << " dominant-baseline=\"middle\"";
                svg << " font-size=\"6\"";
                svg << " font-family=\"monospace\"";
                svg << " fill=\"red\"";
                svg << " >";
                far_svg << "<rect";
                far_svg << " x=\"" << x - 8 << "\"";
                far_svg << " y=\"" << y - 4 << "\"";
                far_svg << " width=\"16\"";
                far_svg << " height=\"4\"";
                far_svg << " stroke=\"red\"";
                far_svg << " fill=\"none\"";
                far_svg << " />";
                cur_col_width = std::max(cur_col_width, 48.0);

                if (reg.name.empty()) {
                    svg << RegisterId{(uint32_t)k};
                } else {
                    svg << escape_text_for_html(reg.name);
                }
                svg << "[" << k2 << "]==" << qb.untagged_id();
                svg << "</text>";
            }

            // Draw target shape.
            if (qb.is_qubit()) {
                double y = q2y((QubitId)qb);

                svg << "<rect";
                svg << " x=\"" << x - 12 << "\"";
                svg << " y=\"" << y - 4 << "\"";
                svg << " width=\"24\"";
                svg << " height=\"8\"";
                svg << " stroke=\"black\"";
                svg << " fill=\"lightgray\"";
                svg << " />";
                far_svg << "<rect";
                far_svg << " x=\"" << x - 12 << "\"";
                far_svg << " y=\"" << y - 4 << "\"";
                far_svg << " width=\"24\"";
                far_svg << " height=\"8\"";
                far_svg << " stroke=\"black\"";
                far_svg << " fill=\"lightgray\"";
                far_svg << " />";

                svg << "<text";
                svg << " x=\"" << x << "\"";
                svg << " y=\"" << y << "\"";
                svg << " text-anchor=\"middle\"";
                svg << " dominant-baseline=\"middle\"";
                svg << " font-size=\"6\"";
                svg << " font-family=\"monospace\"";
                svg << " fill=\"black\"";
                svg << " >";
                svg << RegisterId{(uint32_t)k} << "[" << k2 << "]";
                svg << "</text>";
            }

            if (html) {
                buf_stream << "new BoundedObject(\n    " << x << ",\n    `" << svg.str() << "`,\n    `" << far_svg.str()
                           << "`,\n),\n";
            } else {
                buf_stream << svg.str();
            }
            svg.str("");
            far_svg.str("");
        }
    }

    bool saw_hmr = false;
    size_t group_len = 1;
    for (size_t op_index = 0; op_index < operations.size(); op_index += group_len) {
        group_len = 1;
        const Op &op = operations[op_index];
        if (op.kind == OpType::CX) {
            while (op_index + group_len < operations.size() && operations[op_index + group_len].kind == OpType::CX &&
                   operations[op_index + group_len].q_control1 == op.q_control1) {
                group_len++;
            }
            if (group_len > 1) {
                std::set<QubitId> targets;
                for (size_t k = 0; k < group_len; k++) {
                    if (!targets.insert(operations[op_index + k].q_target.qubit()).second) {
                        group_len = k;
                        break;
                    }
                }
            }
        }
        std::span<const Op> op_group = operations;
        op_group = op_group.subspan(op_index, group_len);

        bool is_classical_command;
        switch (op.kind) {
            case OpType::NEG:
            case OpType::BIT_INVERT:
            case OpType::BIT_STORE0:
            case OpType::BIT_STORE1:
            case OpType::NEG_IF:
            case OpType::BIT_INVERT_IF:
            case OpType::BIT_STORE0_IF:
            case OpType::BIT_STORE1_IF:
            case OpType::PUSH_CONDITION:
            case OpType::POP_CONDITION:
                is_classical_command = true;
                break;
            default:
                is_classical_command = false;
        }

        uint32_t min_q = UINT32_MAX;
        uint32_t max_q = 0;
        double min_y = INFINITY;
        double max_y = -INFINITY;
        for (const auto &sub_op : op_group) {
            std::array<QubitOrFalse, 3> qs{sub_op.q_target, sub_op.q_control1, sub_op.q_control2};
            for (auto q : qs) {
                if (q.is_qubit()) {
                    min_q = std::min(q.untagged_id(), min_q);
                    max_q = std::max(q.untagged_id(), max_q);
                    min_y = std::min(min_y, q2y(q));
                    max_y = std::max(max_y, q2y(q));
                }
            }
        }

        std::array<QubitOrFalse, 3> qs{op.q_target, op.q_control1, op.q_control2};

        // Track x position of column of operations.
        bool need_new_col = false;
        need_new_col |= (op.q_target.is_qubit()) && (text_pos != UINT32_MAX);
        need_new_col |= saw_hmr && op.c_condition.is_bit();
        if (is_classical_command) {
            if (text_pos >= circuit.num_qubits) {
                need_new_col = true;
                text_pos = 0;
            } else {
                text_pos++;
            }
        }
        if (need_new_col) {
            if (!is_classical_command) {
                text_pos = UINT32_MAX;
            }
            x += cur_col_width;
            saw_hmr = false;
            cur_col_width = MIN_COL_WIDTH;
        }
        if (min_q <= max_q) {
            for (uint32_t q = min_q; q <= max_q; q++) {
                if (last_used[q] >= x) {
                    x += cur_col_width;
                    cur_col_width = MIN_COL_WIDTH;
                    break;
                }
            }
            for (uint32_t q = min_q; q <= max_q; q++) {
                last_used[q] = x;
            }
        }

        if ((op.kind == OpType::R || op.kind == OpType::R_IF) && !active[op.q_target.untagged_id()]) {
            wire_transitions[op.q_target.untagged_id()].push_back(x);
            active[op.q_target.untagged_id()] = true;
        } else if ((op.kind == OpType::HMR || op.kind == OpType::HMR_IF) && active[op.q_target.untagged_id()]) {
            wire_transitions[op.q_target.untagged_id()].push_back(x);
            active[op.q_target.untagged_id()] = false;
        }

        // Draw center stem connecting the controls to the target.
        if (max_y > min_y) {
            svg << "<path";
            svg << " d=\"M" << x << "," << min_y << " L" << x << "," << max_y << "\"";
            svg << " stroke=\"black\"";
            svg << " fill=\"none\"";
            svg << " />";

            far_svg << "<path";
            far_svg << " d=\"M" << x << "," << min_y << " L" << x << "," << max_y << "\"";
            far_svg << " stroke=\"black\"";
            far_svg << " fill=\"none\"";
            far_svg << " vector-effect=\"non-scaling-stroke\"";
            far_svg << " />";
        }

        // Draw classical condition.
        if (op.q_target.is_qubit() and op.c_condition.is_bit()) {
            svg << "<path";
            svg << " d=\"M" << x - 1 << "," << min_y - 1 << " l0,-5.5 m2,0 l0,5.5\"";
            svg << " stroke=\"black\"";
            svg << " fill=\"none\"";
            svg << " />";

            svg << "<text";
            svg << " x=\"" << x << "\"";
            svg << " y=\"" << min_y - 8.5 << "\"";
            svg << " text-anchor=\"middle\"";
            svg << " dominant-baseline=\"middle\"";
            svg << " font-size=\"5\"";
            svg << " font-family=\"monospace\"";
            svg << " fill=\"black\"";
            svg << " >" << op.c_condition << "</text>";
        }

        // Draw black control bulbs.
        if (op.kind != OpType::SWAP && op.kind != OpType::SWAP_IF) {
            for (size_t k = 1; k < qs.size(); k++) {
                if (qs[k].is_qubit()) {
                    svg << "<circle";
                    svg << " cx=\"" << x << "\"";
                    svg << " cy=\"" << q2y(qs[k]) << "\"";
                    svg << " r=\"4\"";
                    svg << " stroke=\"black\"";
                    svg << " fill=\"black\"";
                    svg << " />";

                    far_svg << "<circle";
                    far_svg << " cx=\"" << x << "\"";
                    far_svg << " cy=\"" << q2y(qs[k]) << "\"";
                    far_svg << " r=\"4\"";
                    far_svg << " stroke=\"black\"";
                    far_svg << " fill=\"black\"";
                    far_svg << " />";
                }
            }
        } else {
            if (op.q_control1.is_qubit()) {
                double y = q2y(op.q_control1);
                svg << "<path";
                svg << " d=\"M" << x - 4 << "," << y - 4 << " l8,8 m-8,0 l8,-8\"";
                svg << " stroke=\"black\"";
                svg << " fill=\"none\"";
                svg << " />";
            }
        }

        // Draw classical command.
        if (is_classical_command) {
            double y = q2y(QubitId{text_pos}) - 8;
            svg << "<text";
            svg << " x=\"" << x << "\"";
            svg << " y=\"" << y << "\"";
            svg << " text-anchor=\"middle\"";
            svg << " dominant-baseline=\"middle\"";
            svg << " font-size=\"6\"";
            svg << " font-family=\"monospace\"";
            svg << " fill=\"red\"";
            svg << " >";
            far_svg << "<rect";
            far_svg << " x=\"" << x - 8 << "\"";
            far_svg << " y=\"" << y - 4 << "\"";
            far_svg << " width=\"16\"";
            far_svg << " height=\"4\"";
            far_svg << " stroke=\"red\"";
            far_svg << " fill=\"none\"";
            far_svg << " />";
            cur_col_width = std::max(cur_col_width, 48.0);

            switch (op.kind) {
                case OpType::NEG:
                case OpType::NEG_IF:
                    svg << "neg";
                    if (op.c_condition.is_bit()) {
                        svg << " if " << op.c_condition;
                    }
                    break;
                case OpType::BIT_INVERT:
                case OpType::BIT_INVERT_IF:
                    if (op.c_condition.is_bit()) {
                        svg << op.c_target << "^=" << op.c_condition;
                    } else {
                        svg << op.c_target << "^=1";
                    }
                    break;
                case OpType::BIT_STORE0:
                case OpType::BIT_STORE0_IF:
                    if (op.c_condition.is_bit()) {
                        svg << op.c_target << "&=!" << op.c_condition;
                    } else {
                        svg << op.c_target << "=0";
                    }
                    break;
                case OpType::BIT_STORE1:
                case OpType::BIT_STORE1_IF:
                    if (op.c_condition.is_bit()) {
                        svg << op.c_target << "|=" << op.c_condition;
                    } else {
                        svg << op.c_target << "=1";
                    }
                    break;
                case OpType::PUSH_CONDITION:
                    svg << "push_cond " << op.c_condition;
                    break;
                case OpType::POP_CONDITION:
                    svg << "pop_cond";
                    break;
                default: {
                    std::stringstream ss;
                    ss << "Unhandled classical op in html diagram: ";
                    ss << op.kind;
                    throw std::invalid_argument(ss.str());
                }
            }
            svg << "</text>";
        }

        // Draw target shape.
        for (const auto &sub_op : op_group) {
            switch (sub_op.kind) {
                case OpType::CCX_IF:
                case OpType::CX_IF:
                case OpType::X_IF:
                case OpType::CCX:
                case OpType::CX:
                case OpType::X: {
                    // Draw white circle with black outline and cross.
                    double y = q2y(sub_op.q_target);
                    svg << "<circle";
                    svg << " cx=\"" << x << "\"";
                    svg << " cy=\"" << y << "\"";
                    svg << " r=\"4\"";
                    svg << " stroke=\"black\"";
                    svg << " fill=\"white\"";
                    svg << " />";
                    far_svg << "<circle";
                    far_svg << " cx=\"" << x << "\"";
                    far_svg << " cy=\"" << y << "\"";
                    far_svg << " r=\"4\"";
                    far_svg << " stroke=\"black\"";
                    far_svg << " fill=\"white\"";
                    far_svg << " />";
                    svg << "<path";
                    svg << " d=\"M" << x << "," << y - 4 << " l0,8 m-4,-4 l8,0\"";
                    svg << " stroke=\"black\"";
                    svg << " fill=\"none\"";
                    svg << " />";
                    break;
                }
                case OpType::CCZ_IF:
                case OpType::CZ_IF:
                case OpType::Z_IF:
                case OpType::CCZ:
                case OpType::CZ:
                case OpType::Z: {
                    // Draw black control bulb.
                    double y = q2y(sub_op.q_target);
                    svg << "<circle";
                    svg << " cx=\"" << x << "\"";
                    svg << " cy=\"" << y << "\"";
                    svg << " r=\"4\"";
                    svg << " stroke=\"black\"";
                    svg << " fill=\"black\"";
                    svg << " />";

                    far_svg << "<circle";
                    far_svg << " cx=\"" << x << "\"";
                    far_svg << " cy=\"" << y << "\"";
                    far_svg << " r=\"4\"";
                    far_svg << " stroke=\"black\"";
                    far_svg << " fill=\"black\"";
                    far_svg << " />";
                    break;
                }
                case OpType::SWAP_IF:
                case OpType::SWAP: {
                    // Draw diagonal swap cross.
                    double y = q2y(sub_op.q_target);
                    svg << "<path";
                    svg << " d=\"M" << x - 4 << "," << y - 4 << " l8,8 m-8,0 l8,-8\"";
                    svg << " stroke=\"black\"";
                    svg << " fill=\"none\"";
                    svg << " />";
                    break;
                }
                case OpType::Z_POW_IF:
                case OpType::Z_POW: {
                    // Draw box labeled with 'Z^pow'.
                    double y = q2y(sub_op.q_target);
                    svg << "<rect";
                    svg << " x=\"" << x - 4 << "\"";
                    svg << " y=\"" << y - 4 << "\"";
                    svg << " width=\"120\"";
                    svg << " height=\"8\"";
                    svg << " stroke=\"black\"";
                    svg << " fill=\"white\"";
                    svg << " />";
                    far_svg << "<rect";
                    far_svg << " x=\"" << x - 4 << "\"";
                    far_svg << " y=\"" << y - 4 << "\"";
                    far_svg << " width=\"120\"";
                    far_svg << " height=\"8\"";
                    far_svg << " stroke=\"black\"";
                    far_svg << " fill=\"white\"";
                    far_svg << " />";
                    svg << "<text";
                    svg << " x=\"" << (x - 4 + 60) << "\"";
                    svg << " y=\"" << y << "\"";
                    svg << " text-anchor=\"middle\"";
                    svg << " dominant-baseline=\"middle\"";
                    svg << " font-size=\"6\"";
                    svg << " font-family=\"monospace\"";
                    svg << " fill=\"black\"";
                    svg << " >Z^" << sub_op.angle.to_decimal_half_turns() << "</text>";
                    saw_hmr = true;

                    cur_col_width = std::max(cur_col_width, 128.0);

                    break;
                }
                case OpType::HMR_IF:
                case OpType::HMR: {
                    // Draw black box labeled with 'MX'.
                    double y = q2y(sub_op.q_target);
                    svg << "<rect";
                    svg << " x=\"" << x - 4 << "\"";
                    svg << " y=\"" << y - 4 << "\"";
                    svg << " width=\"8\"";
                    svg << " height=\"8\"";
                    svg << " stroke=\"black\"";
                    svg << " fill=\"black\"";
                    svg << " />";
                    far_svg << "<rect";
                    far_svg << " x=\"" << x - 4 << "\"";
                    far_svg << " y=\"" << y - 4 << "\"";
                    far_svg << " width=\"8\"";
                    far_svg << " height=\"8\"";
                    far_svg << " stroke=\"black\"";
                    far_svg << " fill=\"black\"";
                    far_svg << " />";
                    svg << "<text";
                    svg << " x=\"" << x << "\"";
                    svg << " y=\"" << y << "\"";
                    svg << " text-anchor=\"middle\"";
                    svg << " dominant-baseline=\"middle\"";
                    svg << " font-size=\"6\"";
                    svg << " font-family=\"monospace\"";
                    svg << " fill=\"white\"";
                    svg << " >MX</text>";

                    // Draw classical output.
                    svg << "<path";
                    svg << " d=\"M" << x + 4 << "," << y - 1 << " l3,0 m0,2 l-3,0\"";
                    svg << " stroke=\"black\"";
                    svg << " fill=\"none\"";
                    svg << " />";
                    svg << "<text";
                    svg << " x=\"" << x + 7 << "\"";
                    svg << " y=\"" << y << "\"";
                    svg << " text-anchor=\"left\"";
                    svg << " dominant-baseline=\"middle\"";
                    svg << " font-size=\"6\"";
                    svg << " font-family=\"monospace\"";
                    svg << " fill=\"black\"";
                    svg << " >" << sub_op.c_target << "</text>";
                    saw_hmr = true;

                    cur_col_width = std::max(cur_col_width, 32.0);

                    break;
                }
                case OpType::R_IF:
                case OpType::R: {
                    double y = q2y(sub_op.q_target);
                    svg << "<text";
                    svg << " x=\"" << x << "\"";
                    svg << " y=\"" << y << "\"";
                    svg << " text-anchor=\"end\"";
                    svg << " dominant-baseline=\"middle\"";
                    svg << " font-size=\"6\"";
                    svg << " font-family=\"monospace\"";
                    svg << " fill=\"black\"";
                    svg << " >|0⟩</text>";
                    break;
                }
                case OpType::DEBUG_PRINT_Q:
                case OpType::DEBUG_PRINT_C:
                case OpType::DEBUG_PRINT_EMPTY:
                case OpType::DEBUG_PRINT_Q_IF:
                case OpType::DEBUG_PRINT_C_IF:
                case OpType::DEBUG_PRINT_EMPTY_IF:
                case OpType::NEG_IF:
                case OpType::BIT_INVERT_IF:
                case OpType::BIT_STORE0_IF:
                case OpType::BIT_STORE1_IF:
                case OpType::NEG:
                case OpType::BIT_INVERT:
                case OpType::BIT_STORE0:
                case OpType::BIT_STORE1:
                case OpType::PUSH_CONDITION:
                case OpType::POP_CONDITION:
                    break;
                default: {
                    std::stringstream ss;
                    ss << "Unhandled operation for html diagram: ";
                    ss << sub_op.kind;
                    throw std::invalid_argument(ss.str());
                }
            }
        }

        if (html) {
            buf_stream << "new BoundedObject(\n    " << x << ",\n    `" << svg.str() << "`,\n    `" << far_svg.str()
                       << "`,\n),\n";
        } else {
            buf_stream << svg.str();
        }
        svg.str("");
        far_svg.str("");
    }
    x += cur_col_width;
    for (uint32_t q = 0; q < circuit.num_qubits; q++) {
        if (wire_transitions[q].size() % 2 == 1) {
            wire_transitions[q].push_back(x);
        }
    }

    if (html) {
        buf_stream << "];\n";
        buf_stream << "let wire_transitions = [\n";
        for (uint32_t q = 0; q < circuit.num_qubits; q++) {
            buf_stream << "    [";
            for (size_t k = 0; k < wire_transitions[q].size(); k++) {
                buf_stream << wire_transitions[q][k];
                buf_stream << ",";
            }
            buf_stream << "],\n";
        }
        buf_stream << "];\n";
    }
    if (html) {
        buf_stream << "let wire_objects = [\n";
        out_stream << buf_stream.str();
    } else {
        out_stream << "<svg viewBox=\"0 0 " << x << " " << q2y(QubitId{(uint32_t)circuit.num_qubits})
                   << "\" xmlns=\"http://www.w3.org/2000/svg\">";
    }
    for (uint32_t q = 0; q < circuit.num_qubits; q++) {
        double y = q2y(QubitId{q});
        if (html) {
            out_stream << "    [";
        }
        for (size_t k = 0; k < wire_transitions[q].size(); k += 2) {
            if (html) {
                out_stream << "new WireObject(`";
            }
            out_stream << "<path";
            out_stream << " d=\"M" << wire_transitions[q][k] << "," << y << " L" << wire_transitions[q][k + 1] << ","
                       << y << "\"";
            out_stream << " stroke=\"black\"";
            out_stream << " fill=\"none\"";
            out_stream << " />";
            if (html) {
                out_stream << "`),";
            }
        }
        if (html) {
            out_stream << "],\n";
        }
    }
    if (!html) {
        out_stream << buf_stream.str();
    }
    if (html) {
        out_stream << "];\n";
        out_stream << "const min_x = 0;\n";
        out_stream << "const max_x = " << x << ";\n";
        out_stream << "const min_y = 0;\n";
        out_stream << "const max_y = " << q2y(QubitId{(uint32_t)circuit.num_qubits}) << ";\n";

        out_stream << R"HTML(

    const target_objects = 2000;
    const target_wire_transitions = 20;
    const view_pad = 40;
    const max_view_width = (max_x - min_x) * 3;
    const max_view_height = max_y - min_y;
    let viewBox = { x: min_x, y: min_y, width: max_x - min_x, height: max_y - min_y };

    document.addEventListener('DOMContentLoaded', () => {
        let old_shown_start_i = 0;
        let old_shown_end_i = -1;
        let old_shown_stride = 1;
        let old_shown_far = false;
        let old_shown_wire_start_i = [];
        let old_shown_wire_end_i = [];

        const svg = document.getElementById('viewportSvg');
        const zoomStep = 1.002;

        let dragPrevPoint = undefined;

        const updateViewBox = () => {
            viewBox.width = Math.min(viewBox.width, max_view_width);
            viewBox.height = Math.min(viewBox.height, max_view_height);

            let w2h = svg.clientHeight / svg.clientWidth;
            if (w2h * max_view_width > max_view_height) {
                viewBox.height = viewBox.width * w2h;
            }
            viewBox.width = Math.min(viewBox.width, max_view_width);
            viewBox.height = Math.min(viewBox.height, max_view_height);

            viewBox.x = Math.max(viewBox.x, min_x - viewBox.width * 0.5);
            viewBox.x += viewBox.width;
            viewBox.x = Math.min(viewBox.x, max_x + viewBox.width * 0.5);
            viewBox.x -= viewBox.width;

            viewBox.y = Math.max(viewBox.y, min_y);
            viewBox.y += viewBox.height;
            viewBox.y = Math.min(viewBox.y, max_y);
            viewBox.y -= viewBox.height;

            let hash = `x=${viewBox.x},y=${viewBox.y},w=${viewBox.width},h=${viewBox.height}`;
            try {
                let new_url = new URL(window.location);
                new_url.hash = hash;
                history.replaceState(null, '', new_url);
            } catch {
                window.location.hash = hash;
            }
            checkAndRemoveWires();
            checkAndRemoveRects();
        };

        function find_wire_start_index(q) {
            let ts = wire_transitions[q];
            let start_i = 0;
            let end_i = ts.length - 1;
            while (start_i < end_i) {
                let mid_i = (start_i + end_i) >>1;
                if (ts[mid_i] < viewBox.x) {
                    start_i = mid_i + 1;
                } else {
                    end_i = mid_i;
                }
            }
            if (start_i > 0) {
                start_i -= 1;
            }
            if (start_i % 2 === 1) {
                start_i -= 1;
            }
            return start_i / 2;
        }

        function find_wire_end_index(q) {
            let ts = wire_transitions[q];
            let start_i = 0;
            let end_i = ts.length - 1;
            while (start_i < end_i) {
                let mid_i = (start_i + end_i) >> 1;
                if (ts[mid_i] < viewBox.x + viewBox.width) {
                    start_i = mid_i + 1;
                } else {
                    end_i = mid_i;
                }
            }
            if (end_i > 0 && ts[end_i] >= viewBox.x + viewBox.width) {
                end_i -= 1;
            }
            if (end_i % 2 == 1) {
                end_i += 1;
            }
            if (end_i == ts.length) {
                end_i -= 2;
            }
            return end_i / 2;
        }

        function find_first_shown_object_index() {
            let start_i = 0;
            let end_i = objects.length - 1;
            while (start_i < end_i) {
                let mid_i = (start_i + end_i) >>1;
                if (objects[mid_i].start_x < viewBox.x - view_pad) {
                    start_i = mid_i + 1;
                } else {
                    end_i = mid_i;
                }
            }
            if (start_i > 0) {
                start_i -= 1;
            }
            return start_i;
        }

        function find_last_shown_object_index() {
            let start_i = 0;
            let end_i = objects.length - 1;
            while (start_i < end_i) {
                let mid_i = (start_i + end_i) >> 1;
                if (objects[mid_i].start_x < viewBox.x + viewBox.width + view_pad) {
                    start_i = mid_i + 1;
                } else {
                    end_i = mid_i;
                }
            }
            if (end_i > 0 && objects[end_i].start_x >= viewBox.x + viewBox.width + view_pad) {
                end_i -= 1;
            }
            return end_i;
        }

        const checkAndRemoveWires = () => {
            while (old_shown_wire_start_i.length < wire_transitions.length) {
                old_shown_wire_start_i.push(-1);
                old_shown_wire_end_i.push(-1);
            }
            for (let q = 0; q < wire_transitions.length; q++) {
                let start_i = find_wire_start_index(q);
                let end_i = find_wire_end_index(q);
                if (end_i - start_i > target_wire_transitions) {
                    start_i = -1;
                    end_i = -1;
                }

                for (let i = start_i; i >= 0 && i <= end_i; i++) {
                    if (!(i >= old_shown_wire_start_i[q] && i <= old_shown_wire_end_i[q])) {
                        wire_objects[q][i].ensureElementExists();
                        svg.appendChild(wire_objects[q][i].element)
                    }
                }
                if (old_shown_wire_start_i[q] != start_i || old_shown_wire_end_i[q] != end_i) {
                    for (let i = old_shown_wire_start_i[q]; i >= 0 && i <= old_shown_wire_end_i[q]; i++) {
                        if (!(i >= start_i && i <= end_i)) {
                            wire_objects[q][i].element.remove();
                        }
                    }
                    old_shown_wire_start_i[q] = start_i;
                    old_shown_wire_end_i[q] = end_i;
                }
            }
        };
        const checkAndRemoveRects = () => {
            let start_i = find_first_shown_object_index();
            let end_i = find_last_shown_object_index();
            let stride = 1;
            while (end_i - start_i > target_objects * stride) {
                stride *= 2;
            }
            start_i = Math.floor(start_i / stride) * stride;
            let far = stride > 1 || viewBox.width / svg.clientWidth > 2.5 || viewBox.height / svg.clientHeight > 2.5;

            if (stride !== old_shown_stride || old_shown_far != far) {
                for (let i = old_shown_start_i; i <= old_shown_end_i; i += old_shown_stride) {
                    if (old_shown_far) {
                        objects[i].far_element.remove();
                    } else {
                        objects[i].element.remove();
                    }
                }
                old_shown_start_i = 0;
                old_shown_end_i = -1;
            }
            for (let i = start_i; i <= end_i; i += stride) {
                if (!(i >= old_shown_start_i && i <= old_shown_end_i)) {
                    if (far) {
                        objects[i].ensureFarElementExists();
                        svg.appendChild(objects[i].far_element)
                    } else {
                        objects[i].ensureElementExists();
                        svg.appendChild(objects[i].element)
                    }
                }
            }
            svg.setAttribute('viewBox', `${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`);
            for (let i = old_shown_start_i; i <= old_shown_end_i; i += old_shown_stride) {
                if (!(i >= start_i && i <= end_i)) {
                    if (old_shown_far) {
                        objects[i].far_element.remove();
                    } else {
                        objects[i].element.remove();
                    }
                }
            }
            old_shown_start_i = start_i;
            old_shown_end_i = end_i;
            old_shown_stride = stride;
            old_shown_far = far;
        };

        svg.addEventListener('mousedown', e => {
            if (e.button === 0) {
                dragPrevPoint = {x: e.clientX, y: e.clientY};
                svg.style.cursor = 'grabbing';
                e.preventDefault();
            }
        });

        window.addEventListener('mousemove', e => {
            if (dragPrevPoint === undefined || e.button !== 0) {
                return;
            }

            e.preventDefault();
            const dx = (dragPrevPoint.x - e.clientX) * (viewBox.width / svg.clientWidth);
            const dy = (dragPrevPoint.y - e.clientY) * (viewBox.height / svg.clientHeight);

            viewBox.x += dx;
            viewBox.y += dy;
            dragPrevPoint = {x: e.clientX, y: e.clientY};

            requestAnimationFrame(updateViewBox);
        });

        window.addEventListener('mouseup', () => {
            if (dragPrevPoint !== undefined) {
                dragPrevPoint = undefined;
                svg.style.cursor = 'grab';
            }
        });

        svg.addEventListener('wheel', e => {
            e.preventDefault();
            const mousePointToSvg = svg.createSVGPoint();
            mousePointToSvg.x = e.clientX;
            mousePointToSvg.y = e.clientY;

            const screenToSvgMatrix = svg.getScreenCTM().inverse();
            const svgPoint = mousePointToSvg.matrixTransform(screenToSvgMatrix);
            const desiredZoomFactor = Math.pow(zoomStep, e.deltaY);
            const newWidth = Math.min(viewBox.width * desiredZoomFactor, max_view_width)
            const newHeight = Math.min(viewBox.height * desiredZoomFactor, max_view_height)
            const zoomFactor = Math.max(newWidth / viewBox.width, newHeight / viewBox.height);
            viewBox.x = svgPoint.x - (svgPoint.x - viewBox.x) * zoomFactor;
            viewBox.y = svgPoint.y - (svgPoint.y - viewBox.y) * zoomFactor;
            viewBox.width *= zoomFactor;
            viewBox.height *= zoomFactor;

            requestAnimationFrame(updateViewBox);
        });

        svg.style.cursor = 'grab';

        let v = window.location.hash;
        if (v.startsWith('#')) {
            v = v.substring(1);
        }
        let pieces = v.split(',');
        for (let piece of pieces) {
            let terms = piece.split('=');
            if (terms.length === 2) {
                try {
                    if (terms[0] === 'x') {
                        viewBox.x = parseFloat(terms[1])
                    }
                    if (terms[0] === 'y') {
                        viewBox.y = parseFloat(terms[1])
                    }
                    if (terms[0] === 'w') {
                        viewBox.width = parseFloat(terms[1])
                    }
                    if (terms[0] === 'h') {
                        viewBox.height = parseFloat(terms[1])
                    }
                } catch (ex) {
                    console.error(ex);
                }
            }
        }
        requestAnimationFrame(updateViewBox);

        new ResizeObserver(() => requestAnimationFrame(updateViewBox)).observe(svg);
    });
</script>

</html>
)HTML";
    }

    if (!html) {
        out_stream << "</svg>";
    }
}

void Circuit::write_svg_or_html_diagram_to(std::ostream &out_stream, bool html) const {
    std::vector<Op> ops;
    iter_ops([&](Op op) {
        ops.push_back(op);
    });
    size_t reaction_depth = compute_reaction_depth(num_qubits, num_bits, ops);

    size_t touched_qubits = compute_num_touched_qubits((*this));

    write_svg_or_html_diagram_to_helper(reaction_depth, touched_qubits, *this, out_stream, html);
}
