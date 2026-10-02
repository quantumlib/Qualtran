#include "kickmix/util/arg_parse.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <set>
#include <string>

const char *kickmix::require_find_argument(const char *name_c_str, int argc, const char **argv) {
    const char *result = find_argument(name_c_str, argc, argv);
    if (result == nullptr) {
        std::stringstream msg;
        msg << "\033[31mMissing command line argument: '" << name_c_str << "'";
        throw std::invalid_argument(msg.str());
    }
    return result;
}

const char *kickmix::find_argument(const char *name_c_str, int argc, const char **argv) {
    // Respect that the "--" argument terminates flags.
    size_t flag_count = 1;
    while (flag_count < static_cast<size_t>(argc) && strcmp(argv[flag_count], "--") != 0) {
        flag_count++;
    }

    // Search for the desired flag.
    size_t n = strlen(name_c_str);
    for (size_t i = 1; i < flag_count; i++) {
        // Check if argument starts with expected flag.
        const char *loc = strstr(argv[i], name_c_str);
        if (loc != argv[i] || (loc[n] != '\0' && loc[n] != '=')) {
            continue;
        }

        // If the flag is alone and followed by the end or another flag, no
        // argument was provided. Return the empty string to indicate this.
        if (loc[n] == '\0' &&
            (static_cast<int>(i) == argc - 1 || (argv[i + 1][0] == '-' && !isdigit(argv[i + 1][1])))) {
            return argv[i] + n;
        }

        // If the flag value is specified inline with '=', return a pointer to
        // the start of the value within the flag string.
        if (loc[n] == '=') {
            // Argument provided inline.
            return loc + n + 1;
        }

        // The argument value is specified by the next command line argument.
        return argv[i + 1];
    }

    // Not found.
    return nullptr;
}

void kickmix::check_for_unknown_arguments(
    const std::vector<const char *> &known_arguments,
    const std::vector<const char *> &known_but_deprecated_arguments,
    const char *program_name,
    const char *for_mode,
    int argc,
    const char **argv) {
    for (int i = 1; i < argc; i++) {
        if (for_mode != nullptr && i == 1 && strcmp(argv[i], for_mode) == 0) {
            continue;
        }
        // Respect that the "--" argument terminates flags.
        if (!strcmp(argv[i], "--")) {
            break;
        }

        // Check if there's a matching command line argument.
        int matched = 0;
        std::array<const std::vector<const char *> *, 2> both{&known_arguments, &known_but_deprecated_arguments};
        for (const auto &knowns : both) {
            for (const auto &known : *knowns) {
                const char *loc = strstr(argv[i], known);
                size_t n = strlen(known);
                if (loc == argv[i] && (loc[n] == '\0' || loc[n] == '=')) {
                    // Skip words that are values for a previous flag.
                    if (loc[n] == '\0' && i < argc - 1 && argv[i + 1][0] != '-') {
                        i++;
                    }
                    matched = 1;
                    break;
                }
            }
        }

        // Print error and exit if flag is not recognized.
        if (!matched) {
            std::stringstream msg;
            if (for_mode == nullptr) {
                msg << "Unrecognized command line argument " << argv[i] << ".\n";
                msg << "Recognized command line arguments:\n";
            } else {
                msg << "Unrecognized command line argument " << argv[i] << " for `";
                msg << program_name << " " << for_mode << "`.\n";
                msg << "Recognized command line arguments for `" << program_name << " " << for_mode << "`:\n";
            }
            std::set<std::string> known_sorted;
            for (const auto &v : known_arguments) {
                known_sorted.insert(v);
            }
            for (const auto &v : known_sorted) {
                msg << "    " << v << "\n";
            }
            throw std::invalid_argument(msg.str());
        }
    }
}

bool kickmix::find_bool_argument(const char *name_c_str, int argc, const char **argv) {
    const char *text = find_argument(name_c_str, argc, argv);
    if (text == nullptr) {
        return false;
    }
    if (text[0] == '\0') {
        return true;
    }
    std::stringstream msg;
    msg << "Got non-empty value '" << text << "' for boolean flag '" << name_c_str << "'.";
    throw std::invalid_argument(msg.str());
}

static bool parse_int64(std::string_view data, int64_t *out) {
    if (data.empty()) {
        return false;
    }
    bool negate = false;
    if (data.starts_with("-")) {
        negate = true;
        data = data.substr(1);
    } else if (data.starts_with("+")) {
        data = data.substr(1);
    }

    uint64_t accumulator = 0;
    for (char c : data) {
        if (c == '_' && accumulator > 0) {
            continue;
        }
        if (!(c >= '0' && c <= '9')) {
            return false;
        }
        uint64_t digit = c - '0';
        uint64_t next = accumulator * 10 + digit;
        if (accumulator != (next - digit) / 10) {
            return false;  // Overflow.
        }
        accumulator = next;
    }

    if (negate && accumulator == static_cast<uint64_t>(INT64_MAX) + uint64_t{1}) {
        *out = INT64_MIN;
        return true;
    }
    if (accumulator > INT64_MAX) {
        return false;
    }

    *out = static_cast<int64_t>(accumulator);
    if (negate) {
        *out *= -1;
    }
    return true;
}

int64_t kickmix::find_int64_argument(
    const char *name_c_str, int64_t default_value, int64_t min_value, int64_t max_value, int argc, const char **argv) {
    const char *text = find_argument(name_c_str, argc, argv);
    if (text == nullptr || text[0] == '\0') {
        if (default_value < min_value || default_value > max_value) {
            std::stringstream msg;
            msg << "Must specify a value for int flag '" << name_c_str << "'.";
            throw std::invalid_argument(msg.str());
        }
        return default_value;
    }

    // Attempt to parse.
    int64_t i;
    if (!parse_int64(text, &i)) {
        std::stringstream msg;
        msg << "Got non-int64 value '" << text << "' for int64 flag '" << name_c_str << "'.";
        throw std::invalid_argument(msg.str());
    }

    // In range?
    if (i < min_value || i > max_value) {
        std::stringstream msg;
        msg << "Integer value '" << text << "' for flag '" << name_c_str << "' doesn't satisfy " << min_value
            << " <= " << i << " <= " << max_value << ".";
        throw std::invalid_argument(msg.str());
    }

    return i;
}

float kickmix::find_float_argument(
    const char *name_c_str, float default_value, float min_value, float max_value, int argc, const char **argv) {
    const char *text = find_argument(name_c_str, argc, argv);
    if (text == nullptr) {
        if (default_value < min_value || default_value > max_value) {
            std::stringstream msg;
            msg << "Must specify a value for float flag '" << name_c_str << "'.";
            throw std::invalid_argument(msg.str());
        }
        return default_value;
    }

    // Attempt to parse.
    char *processed;
    float f = strtof(text, &processed);
    if (*processed != '\0') {
        std::stringstream msg;
        msg << "Got non-float value '" << text << "' for float flag '" << name_c_str << "'.";
        throw std::invalid_argument(msg.str());
    }

    // In range?
    if (f < min_value || f > max_value || std::isnan(f)) {
        std::stringstream msg;
        msg << "Float value '" << text << "' for flag '" << name_c_str << "' doesn't satisfy " << min_value
            << " <= " << f << " <= " << max_value << ".";
        throw std::invalid_argument(msg.str());
    }

    return f;
}

double kickmix::find_double_argument(
    const char *name_c_str, double default_value, double min_value, double max_value, int argc, const char **argv) {
    const char *text = find_argument(name_c_str, argc, argv);
    if (text == nullptr) {
        if (default_value < min_value || default_value > max_value) {
            std::stringstream msg;
            msg << "Must specify a value for float flag '" << name_c_str << "'.";
            throw std::invalid_argument(msg.str());
        }
        return default_value;
    }

    // Attempt to parse.
    char *processed;
    double d = strtod(text, &processed);
    if (*processed != '\0') {
        std::stringstream msg;
        msg << "Got non-double-precision-float value '" << text << "' for flag '" << name_c_str << "'.";
        throw std::invalid_argument(msg.str());
    }

    // In range?
    if (d < min_value || d > max_value || std::isnan(d)) {
        std::stringstream msg;
        msg << "Double precision float value '" << text << "' for flag '" << name_c_str << "' doesn't satisfy "
            << min_value << " <= " << d << " <= " << max_value << ".";
        throw std::invalid_argument(msg.str());
    }

    return d;
}

FILE *kickmix::find_open_file_argument(
    const char *name_c_str, FILE *default_file, const char *mode, int argc, const char **argv) {
    const char *path_c_str = find_argument(name_c_str, argc, argv);
    if (path_c_str == nullptr) {
        if (default_file == nullptr) {
            std::stringstream msg;
            msg << "Missing command line argument: '" << name_c_str << "'";
            throw std::invalid_argument(msg.str());
        }
        return default_file;
    }
    if (*path_c_str == '\0') {
        std::stringstream msg;
        msg << "Command line argument '" << name_c_str << "' can't be empty. It's supposed to be a file path.";
        throw std::invalid_argument(msg.str());
    }
    FILE *file = fopen(path_c_str, mode);
    if (file == nullptr) {
        std::stringstream msg;
        msg << "Failed to open '" << path_c_str << "'";
        throw std::invalid_argument(msg.str());
    }
    return file;
}
