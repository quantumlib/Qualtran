#ifndef KICKGEN_PYBIND_UTIL_H
#define KICKGEN_PYBIND_UTIL_H

#include <array>
#include <map>
#include <pybind11/pybind11.h>
#include <string_view>
#include <utility>
#include <vector>

#include "kickmix/circuit/op_type.h"
#include "kickmix/util/fixed_precision_angle_128.h"

namespace kickmix_py {

inline const std::array<std::pair<std::string_view, std::vector<kickmix::OpType>>, 17> OP_COUNT_GROUPS{{
    {"NEG", {kickmix::OpType::NEG, kickmix::OpType::NEG_IF}},
    {"BIT_INVERT", {kickmix::OpType::BIT_INVERT, kickmix::OpType::BIT_INVERT_IF}},
    {"BIT_STORE0", {kickmix::OpType::BIT_STORE0, kickmix::OpType::BIT_STORE0_IF}},
    {"BIT_STORE1", {kickmix::OpType::BIT_STORE1, kickmix::OpType::BIT_STORE1_IF}},
    {"X", {kickmix::OpType::X, kickmix::OpType::X_IF}},
    {"Z", {kickmix::OpType::Z, kickmix::OpType::Z_IF}},
    {"R", {kickmix::OpType::R, kickmix::OpType::R_IF}},
    {"HMR", {kickmix::OpType::HMR, kickmix::OpType::HMR_IF}},
    {"CX", {kickmix::OpType::CX, kickmix::OpType::CX_IF}},
    {"CZ", {kickmix::OpType::CZ, kickmix::OpType::CZ_IF}},
    {"SWAP", {kickmix::OpType::SWAP, kickmix::OpType::SWAP_IF}},
    {"CCX", {kickmix::OpType::CCX, kickmix::OpType::CCX_IF}},
    {"CCZ", {kickmix::OpType::CCZ, kickmix::OpType::CCZ_IF}},
    {"Z_POW", {kickmix::OpType::Z_POW, kickmix::OpType::Z_POW_IF}},
    {"DEBUG_PRINT",
     {kickmix::OpType::DEBUG_PRINT_EMPTY,
      kickmix::OpType::DEBUG_PRINT_Q,
      kickmix::OpType::DEBUG_PRINT_C,
      kickmix::OpType::DEBUG_PRINT_EMPTY_IF,
      kickmix::OpType::DEBUG_PRINT_Q_IF,
      kickmix::OpType::DEBUG_PRINT_C_IF}},
    {"POP_CONDITION", {kickmix::OpType::POP_CONDITION}},
    {"PUSH_CONDITION", {kickmix::OpType::PUSH_CONDITION}},
}};

inline pybind11::object fixed_precision_angle_to_py_fraction(const kickmix::FixedPrecisionAngle128 &angle) {
    uint64_t w0 = angle.words[0];
    uint64_t w1 = angle.words[1];
    return pybind11::module_::import("fractions")
        .attr("Fraction")(
            pybind11::cast(w0) | (pybind11::cast(w1) << pybind11::cast(64)), pybind11::cast(1) << pybind11::cast(127));
}

inline void populate_z_pow_counts(
    pybind11::dict &result,
    const pybind11::object &z_pow_key,
    const std::map<kickmix::FixedPrecisionAngle128, uint64_t> &angle_counts) {
    for (const auto &[angle, count] : angle_counts) {
        pybind11::object frac = fixed_precision_angle_to_py_fraction(angle);
        pybind11::object key_obj = z_pow_key(frac);
        if (!pybind11::isinstance<pybind11::str>(key_obj)) {
            throw pybind11::type_error("z_pow_key must return a str");
        }
        uint64_t prev = 0;
        if (result.contains(key_obj)) {
            prev = pybind11::cast<uint64_t>(result[key_obj]);
        }
        result[key_obj] = prev + count;
    }
}

consteval bool const_eval_string_starts_with(const char *string, const char *prefix) {
    while (*prefix) {
        if (*string++ != *prefix++) {
            return false;
        }
    }
    return true;
}

consteval bool const_eval_string_contains_substring(const char *string, const char *substring) {
    while (*string) {
        if (const_eval_string_starts_with(string++, substring)) {
            return true;
        }
    }
    return false;
}

consteval void const_eval_write_cleaned_doc_string_to(const char *doc, char *out) {
    // Determine indentation using first non-empty line.
    size_t indent = 0;
    while (*doc == ' ' || *doc == '\n') {
        if (*doc == '\n') {
            indent = 0;
        } else {
            indent++;
        }
        doc++;
    }

    // Copy lines with indent removed into output.
    while (*doc != '\0') {
        // Skip indentation.
        for (size_t j = 0; j < indent && *doc == ' '; j++) {
            doc++;
        }

        // Copy rest of line.
        const char *start_of_line = out;
        while (*doc != '\0') {
            *out++ = *doc;
            if (*doc == '\n') {
                doc++;
                break;
            } else {
                doc++;
            }
        }

        // Validate.
        if (const_eval_string_contains_substring(start_of_line, "\"\"\"")) {
            throw R"(Docstring used """ instead of '''.)";
        }
        if (out - start_of_line > 80 && !const_eval_string_starts_with(start_of_line, "@signature") &&
            !const_eval_string_starts_with(start_of_line, "@overload") &&
            !const_eval_string_contains_substring(start_of_line, "https://")) {
            throw "Docstring has a line longer than 80 characters.";
        }
    }
}

template <size_t N>
consteval std::array<char, N + 1> clean_doc_string(const char (&doc)[N]) {
    std::array<char, N + 1> buf{};
    const_eval_write_cleaned_doc_string_to(doc, buf.data());
    return buf;
}

}  // namespace kickmix_py

#endif
