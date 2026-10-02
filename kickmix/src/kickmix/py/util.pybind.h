#ifndef KICKGEN_PYBIND_UTIL_H
#define KICKGEN_PYBIND_UTIL_H

#include <pybind11/pybind11.h>

namespace kickmix_py {

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
