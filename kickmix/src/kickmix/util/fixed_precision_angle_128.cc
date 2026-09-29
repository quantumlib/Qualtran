#include "kickmix/util/fixed_precision_angle_128.h"

#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace kickmix;

FixedPrecisionAngle128::operator bool() const {
    return words[0] != 0 || words[1] != 0;
}

FixedPrecisionAngle128 FixedPrecisionAngle128::from_half_turns_exact_double(double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("Angle cannot be NaN or infinite.");
    }
    value *= 0.5;
    bool neg = value < 0;
    double abs_val = std::abs(value);
    double frac = abs_val - std::floor(abs_val);
    double high_f = std::ldexp(frac, 64);
    double w1_f = std::floor(high_f);
    double low_f = std::ldexp(high_f - w1_f, 64);
    double w0_f = std::floor(low_f);
    if (low_f != w0_f) {
        throw std::invalid_argument("Double has bits below 2**-128 and so cannot be converted exactly.");
    }
    uint64_t w1 = static_cast<uint64_t>(w1_f);
    uint64_t w0 = static_cast<uint64_t>(w0_f);
    if (neg && (w0 != 0 || w1 != 0)) {
        w0 = ~w0 + 1;
        w1 = ~w1 + (w0 == 0 ? 1 : 0);
    }
    return {w0, w1};
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

FixedPrecisionAngle128 FixedPrecisionAngle128::from_rounded_decimal_half_turns(
    std::string_view decimal_to_round_to_nearest) {
    std::string_view s = decimal_to_round_to_nearest;
    bool neg = false;
    if (s.starts_with("-")) {
        neg = true;
        s.remove_prefix(1);
    } else if (s.starts_with("+")) {
        s.remove_prefix(1);
    }

    size_t dot = s.find('.');
    if (dot == std::string_view::npos || dot == 0 || dot + 1 >= s.size()) {
        std::stringstream ss;
        ss << "Expected a value matching /[-+]?[0-9]+\\.[0-9]+/ but got '" << decimal_to_round_to_nearest << "'.";
        throw std::invalid_argument(ss.str());
    }

    std::string_view int_part = s.substr(0, dot);
    for (char c : int_part) {
        if (c < '0' || c > '9') {
            throw std::invalid_argument("Invalid decimal angle: non-digit in integer part.");
        }
    }
    std::string_view frac_part = s.substr(dot + 1);
    for (char c : frac_part) {
        if (c < '0' || c > '9') {
            throw std::invalid_argument("Invalid decimal angle: non-digit in fractional part.");
        }
    }
    bool is_odd = (int_part.back() - '0') & 1;
    while (!frac_part.empty() && frac_part.back() == '0') {
        frac_part.remove_suffix(1);
    }
    if (frac_part.empty()) {
        if (is_odd) {
            return {0, uint64_t{1} << 63};
        } else {
            return {0, 0};
        }
    }

    // Every multiple of 2**-128 half-turns has an exact decimal expansion of at most 128 digits.
    // If there are non-zero digits beyond the 128th place, replacing everything from the
    // 129th place onwards with a single '5' keeps the value in the exact same (P/10**128, (P+1)/10**128)
    // interval and therefore rounds to the exact same nearest multiple of 2**-127 half-turns.
    bool has_sticky_digit = frac_part.size() > 128;
    size_t num_prefix_digits = has_sticky_digit ? 128 : frac_part.size();

    uint64_t acc[8]{};
    auto step_digit = [&](uint64_t d) {
        uint64_t rem = d;
        for (int i = 7; i >= 0; i--) {
            unsigned __int128 cur = (static_cast<unsigned __int128>(rem) << 64) | acc[i];
            acc[i] = static_cast<uint64_t>(cur / 10);
            rem = static_cast<uint64_t>(cur % 10);
        }
    };

    if (has_sticky_digit) {
        step_digit(5);
    }
    for (size_t k = num_prefix_digits; k--;) {
        step_digit(static_cast<uint64_t>(frac_part[k] - '0'));
    }

    uint64_t w1 = (acc[7] >> 1) | (static_cast<uint64_t>(is_odd) << 63);
    uint64_t w0 = (acc[7] << 63) | (acc[6] >> 1);
    bool half_bit = acc[6] & 1;
    if (half_bit) {
        bool sticky_low = acc[5] != 0 || acc[4] != 0 || acc[3] != 0 || acc[2] != 0 || acc[1] != 0 || acc[0] != 0;
        if (sticky_low || (w0 & 1)) {
            w0 += 1;
            if (w0 == 0) {
                w1 += 1;
            }
        }
    }

    FixedPrecisionAngle128 result{w0, w1};
    return neg ? -result : result;
}

template <typename TCallback>
inline void to_half_turns_decimal_helper(const FixedPrecisionAngle128 &angle, const TCallback &callback) {
    uint64_t w0 = angle.words[0];
    uint64_t w1 = angle.words[1];
    if (w1 & (uint64_t{1} << 63)) {
        callback('1');
    } else {
        callback('0');
    }
    callback('.');
    w1 <<= 1;
    if (w0 & (uint64_t{1} << 63)) {
        w1 |= 1;
    }
    w0 <<= 1;
    if (w0 == 0 && w1 == 0) {
        callback('0');
        return;
    }

    while (w0 != 0 || w1 != 0) {
        unsigned __int128 low_prod = static_cast<unsigned __int128>(w0) * 10;
        unsigned __int128 high_prod = static_cast<unsigned __int128>(w1) * 10 + (low_prod >> 64);
        uint64_t digit = static_cast<uint64_t>(high_prod >> 64);
        callback('0' + digit);
        w0 = static_cast<uint64_t>(low_prod);
        w1 = static_cast<uint64_t>(high_prod);
    }
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

std::string FixedPrecisionAngle128::to_decimal_half_turns() const {
    std::string result = "";
    to_half_turns_decimal_helper(*this, [&](char c) {
        result.push_back(c);
    });
    return result;
}

void FixedPrecisionAngle128::write_decimal_half_turns_to(FILE *f) const {
    to_half_turns_decimal_helper(*this, [&](char c) {
        putc(c, f);
    });
}

FixedPrecisionAngle128 FixedPrecisionAngle128::operator+(const FixedPrecisionAngle128 &other) const {
    FixedPrecisionAngle128 result = *this;
    result += other;
    return result;
}

FixedPrecisionAngle128 &FixedPrecisionAngle128::operator+=(const FixedPrecisionAngle128 &other) {
    uint64_t next_low = words[0] + other.words[0];
    uint64_t carry = next_low < words[0];
    words[0] = next_low;
    words[1] += other.words[1] + carry;
    return *this;
}

std::string FixedPrecisionAngle128::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

std::ostream &kickmix::operator<<(std::ostream &out, const FixedPrecisionAngle128 &val) {
    to_half_turns_decimal_helper(val, [&](char c) {
        out << c;
    });
    out << "π";
    return out;
}

FixedPrecisionAngle128 FixedPrecisionAngle128::operator-() const {
    uint64_t w0 = words[0];
    uint64_t w1 = words[1];

    // Bitwise complement replaces w with -w-1.
    w0 = ~w0;
    w1 = ~w1;

    // Increment to get back to -w.
    w0 += 1;
    if (w0 == 0) {
        w1 += 1;
    }

    return {w0, w1};
}

FixedPrecisionAngle128 FixedPrecisionAngle128::rotated180() const {
    return {words[0], words[1] ^ (uint64_t{1} << 63)};
}

bool FixedPrecisionAngle128::is_closer_to_half_turn_than_no_turn() const {
    auto r = words[1] >> 62;
    return r == 0b10 || r == 0b01;
}

bool FixedPrecisionAngle128::is_multiple_of_45_degrees() const {
    return words[0] == 0 && (words[1] & ~(uint64_t{7} << 61)) == 0;
}

bool FixedPrecisionAngle128::is_multiple_of_90_degrees() const {
    return words[0] == 0 && (words[1] & ~(uint64_t{3} << 62)) == 0;
}

bool FixedPrecisionAngle128::is_multiple_of_180_degrees() const {
    return words[0] == 0 && (words[1] & ~(uint64_t{1} << 63)) == 0;
}

FixedPrecisionAngle128 &FixedPrecisionAngle128::operator=(int half_turns) {
    words[0] = 0;
    words[1] = (uint64_t)(half_turns & 1) << 63;
    return *this;
}

FixedPrecisionAngle128 &FixedPrecisionAngle128::operator+=(int half_turns) {
    words[1] ^= (uint64_t)(half_turns & 1) << 63;
    return *this;
}
