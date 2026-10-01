#ifndef KICKMIX_UTIL_FIXED_PRECISION_ANGLE_H
#define KICKMIX_UTIL_FIXED_PRECISION_ANGLE_H

#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

namespace kickmix {

/// An angle, specified as a fraction of a turn, stored to a fixed precision of 2**-128.
struct FixedPrecisionAngle128 {
    /// The angle is equal to 2*pi*(words[0] / 2**128 + words[1] / 2**64).
    uint64_t words[2];

    bool operator==(const FixedPrecisionAngle128 &rhs) const = default;
    explicit operator bool() const;

    static FixedPrecisionAngle128 from_power_of_2_half_turns(int exponent);

    /// Returns the given double (mod 2) as a FixedPrecisionAngle128, or else throws.
    static FixedPrecisionAngle128 from_half_turns_exact_double(double value);

    /// Parses the given decimal text in order to produce a FixedPrecisionAngle128.
    ///
    /// Args:
    ///     decimal_half_turns: Text specifying the angle in half turns. This text must
    ///         match the regex /[-+]?[0-9]+.[0-9]+/. For example, it could be
    ///         "0.250000000000000000000000000000000000002938735877055718" or "-5.25"
    ///         or "+0.25". Scientific notation is not permitted.
    static FixedPrecisionAngle128 from_rounded_decimal_half_turns(std::string_view decimal_half_turns);

    /// Returns the angle's exact representation in decimal.
    ///
    /// Returns:
    ///     Text starting with "0." followed by a series of digits specifying the
    ///     angle exactly as a fraction of a turn. Trailing zeroes are not included
    ///     (except that 0 is represented as "0.0"). For example, 90° is represented
    ///     as "0.25".
    std::string to_decimal_half_turns() const;
    void write_decimal_half_turns_to(FILE *f) const;
    std::string str() const;

    /// Returns an angle equal to the sum of two given angles (modulo an entire turn).
    FixedPrecisionAngle128 operator+(const FixedPrecisionAngle128 &other) const;
    /// Inplace angle addition (modulo an entire turn).
    FixedPrecisionAngle128 &operator+=(const FixedPrecisionAngle128 &other);

    FixedPrecisionAngle128 operator-() const;

    FixedPrecisionAngle128 rotated180() const;
    bool is_closer_to_half_turn_than_no_turn() const;
    bool is_multiple_of_45_degrees() const;
    bool is_multiple_of_90_degrees() const;
    bool is_multiple_of_180_degrees() const;
    FixedPrecisionAngle128 &operator=(int half_turns);
    FixedPrecisionAngle128 &operator+=(int half_turns);
    FixedPrecisionAngle128 &operator*=(uint64_t factor);
    FixedPrecisionAngle128 operator*(uint64_t factor) const;
};
std::ostream &operator<<(std::ostream &out, const FixedPrecisionAngle128 &val);

}  // namespace kickmix

#endif
