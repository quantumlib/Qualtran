#ifndef _KICKMIX_BIG_INT_H
#define _KICKMIX_BIG_INT_H

#include <cstdint>
#include <optional>
#include <random>
#include <string>

namespace kickmix {

struct FixedWidthIntIter_End {};
struct FixedWidthIntIter {
    const uint64_t *words;
    size_t num_bits;
    size_t k;
    bool operator*() const {
        return (words[k / 64] >> (k & 63)) & 1;
    }
    FixedWidthIntIter &operator++() {
        k++;
        return *this;
    }
    bool operator==(FixedWidthIntIter_End) const {
        return k == num_bits;
    }
};

struct BitWordRef {
    uint64_t *word;
    uint8_t offset;
    BitWordRef &operator=(bool other) {
        if (other) {
            *word |= uint64_t{1} << offset;
        } else {
            *word &= ~(uint64_t{1} << offset);
        }
        return *this;
    }
    BitWordRef &operator^=(bool other) {
        *word ^= uint64_t{other} << offset;
        return *this;
    }
    operator bool() const {
        return ((*word >> offset) & 1) != 0;
    }
};

struct FixedWidthInt {
    /// The target size of the fixed width integer. Inplace operations
    /// will be truncated to this size.
    size_t num_bits;
    /// The allocated storage size of the fixed width integer. Some methods
    /// assume num_words == ceil(num_bits / 64.0), or at least rely on it
    /// for good performance.
    size_t num_words;
    /// The allocated storage containing the bits of the fixed width integer.
    /// When num_words == 0 this may be a null pointer.
    uint64_t *words;

    explicit FixedWidthInt(size_t num_bits);
    explicit FixedWidthInt(size_t num_bits, uint64_t value);
    explicit FixedWidthInt(size_t num_bits, uint32_t value);
    explicit FixedWidthInt(size_t num_bits, uint16_t value);
    explicit FixedWidthInt(size_t num_bits, uint8_t value);
    explicit FixedWidthInt(size_t num_bits, int64_t value);
    explicit FixedWidthInt(size_t num_bits, int32_t value);
    explicit FixedWidthInt(size_t num_bits, int16_t value);
    explicit FixedWidthInt(size_t num_bits, int8_t value);
    explicit FixedWidthInt(size_t num_bits, std::string_view text);
    explicit FixedWidthInt(std::string_view text);

    static FixedWidthInt from_str(std::string_view text, std::optional<size_t> num_bits = {});
    static FixedWidthInt from_str_binary(std::string_view text, std::optional<size_t> num_bits = {});
    static FixedWidthInt from_str_hex(std::string_view text, std::optional<size_t> num_bits = {});
    static FixedWidthInt from_str_dec(std::string_view text, std::optional<size_t> num_bits = {});

    FixedWidthInt();
    FixedWidthInt(const FixedWidthInt &value);
    FixedWidthInt(FixedWidthInt &&value) noexcept;
    ~FixedWidthInt();
    static FixedWidthInt random(size_t num_bits, std::mt19937_64 &rng);
    static FixedWidthInt random_mod(std::mt19937_64 &rng, const FixedWidthInt &modulus);
    void randomize(std::mt19937_64 &rng);
    void randomize_mod(std::mt19937_64 &rng, const FixedWidthInt &modulus);
    void randomize_coprime_mod(std::mt19937_64 &rng, const FixedWidthInt &modulus);
    explicit operator uint64_t() const;
    explicit operator bool() const;
    explicit operator double() const;
    /// Converts the integer to a double, initializes the exponent to a constant instead
    /// of to a value that depends on the size of the input number. The result is that the
    /// returned double always lies in the range [1.0, 2.0), except if the input is 0 in
    /// which case the returned double is 0.
    ///
    /// For non-zero values X, this method satisfies the following approximate invariant:
    ///
    ///     X.to_approx_mantissa() * pow(2, X.num_bits_in_use() - 1) ~= X
    double to_approx_mantissa() const;

    inline size_t size() const {
        return num_bits;
    }
    inline FixedWidthIntIter begin() const {
        return FixedWidthIntIter{words, num_bits, 0};
    }
    inline FixedWidthIntIter_End end() const {
        return {};
    }

    size_t num_bits_in_use() const;
    uint64_t read_bit_shifted_word(int64_t offset) const;
    uint64_t read_bit_shifted_word(size_t offset) const;

    FixedWidthInt &operator=(const FixedWidthInt &value);
    FixedWidthInt &operator=(FixedWidthInt &&value) noexcept;
    FixedWidthInt &operator=(uint64_t value);
    FixedWidthInt &operator=(int64_t value);

    /// Inplace multiplication by 5.
    ///
    /// Returns:
    ///     Whether the multiplication overflowed.
    bool inplace_times5();
    /// Inplace division by 5, rounding down.
    void inplace_div5();

    void resize_clear(size_t new_num_bits);
    void resize_shrink(size_t new_num_bits);
    void clear_to_zero();
    inline bool clip_hanging_bits() {
        if (!(num_bits & 63)) {
            return false;
        }
        uint64_t mask = ~(-static_cast<uint64_t>(1) << (num_bits & 63));
        auto old_word = words[num_words - 1];
        words[num_words - 1] = old_word & mask;
        return old_word != words[num_words - 1];
    }

    bool operator==(const FixedWidthInt &other) const;
    bool operator==(uint64_t other) const;
    bool operator==(int64_t other) const;

    void idouble_mod(const FixedWidthInt &modulus);
    void ihalve_mod(const FixedWidthInt &modulus);
    bool operator<=(const FixedWidthInt &other) const;
    bool operator<=(uint64_t other) const;
    bool operator<=(int64_t other) const;

    bool operator<(const FixedWidthInt &other) const;
    bool operator<(uint64_t other) const;
    bool operator<(int64_t other) const;

    bool operator[](size_t index) const;
    BitWordRef bit_ref(size_t index);
    BitWordRef back_ref();
    BitWordRef front_ref();
    void flip_bit(size_t index);
    void reset_bit(size_t index);
    void set_bit(size_t index, bool new_value = true);
    std::string str() const;
    std::string hex() const;
    std::string bin() const;

    bool non_zero() const;
    bool increment(size_t bit_offset = 0);
    bool decrement(size_t bit_offset = 0);
    void negate();
    void negate_mod(const FixedWidthInt &modulus);

    /// Overwrites contents with bits from the other int.
    ///
    /// The other value is truncated, or zero-padded, as needed.
    void write_from(const FixedWidthInt &other);

    FixedWidthInt &operator^=(int mask);
    FixedWidthInt &operator^=(const FixedWidthInt &offset);
    FixedWidthInt &operator&=(const FixedWidthInt &offset);
    FixedWidthInt &operator|=(const FixedWidthInt &offset);
    FixedWidthInt &operator+=(const FixedWidthInt &offset);
    FixedWidthInt &operator+=(uint64_t offset);
    FixedWidthInt &operator+=(int64_t offset);
    void iadd_mod(const FixedWidthInt &offset, const FixedWidthInt &modulus);
    void isub_mod(const FixedWidthInt &offset, const FixedWidthInt &modulus);
    FixedWidthInt &iadd_shifted(const FixedWidthInt &offset, size_t shift);
    FixedWidthInt &isub_shifted(const FixedWidthInt &offset, size_t shift, bool *borrow_out);
    bool iadd_carry(const FixedWidthInt &offset, bool carry_in = false);
    bool iadd_carry(uint64_t offset, bool carry_in = false);
    bool isub_borrow(const FixedWidthInt &offset);

    bool is_coprime_to(const FixedWidthInt &other) const;
    FixedWidthInt mult_inverse(const FixedWidthInt &modulus) const;
    FixedWidthInt &operator*=(const FixedWidthInt &value);
    FixedWidthInt &operator*=(uint64_t value);
    void iadd_product(uint64_t v1, const FixedWidthInt &v2);
    void iadd_product(const FixedWidthInt &v1, uint64_t v2);
    void iadd_product(const FixedWidthInt &v1, const FixedWidthInt &v2);
    void iadd_product_shifted(uint64_t v1, const FixedWidthInt &v2, size_t shift);
    void iadd_product_shifted(const FixedWidthInt &v1, uint64_t v2, size_t shift);
    void isub_product_shifted(const FixedWidthInt &v1, uint64_t v2, size_t shift);
    void isub_product_shifted(uint64_t v1, const FixedWidthInt &v2, size_t shift);
    void isub_product(uint64_t v1, const FixedWidthInt &v2);
    void isub_product(const FixedWidthInt &v1, uint64_t v2);
    FixedWidthInt operator*(const FixedWidthInt &value) const;
    FixedWidthInt operator+(const FixedWidthInt &value) const;
    FixedWidthInt operator-(const FixedWidthInt &value) const;
    FixedWidthInt operator-() const;

    FixedWidthInt &operator-=(const FixedWidthInt &offset);
    FixedWidthInt &operator-=(uint64_t offset);
    FixedWidthInt &operator-=(int64_t offset);

    FixedWidthInt &operator<<=(int64_t shift);
    FixedWidthInt &operator<<=(uint64_t shift);
    FixedWidthInt &operator>>=(int64_t shift);
    FixedWidthInt &operator>>=(uint64_t shift);

    inline FixedWidthInt &operator=(uint32_t value) {
        return *this = static_cast<uint64_t>(value);
    }
    inline FixedWidthInt &operator=(uint16_t value) {
        return *this = static_cast<uint64_t>(value);
    }
    inline FixedWidthInt &operator=(uint8_t value) {
        return *this = static_cast<uint64_t>(value);
    }
    inline FixedWidthInt &operator=(int32_t value) {
        return *this = static_cast<int64_t>(value);
    }
    inline FixedWidthInt &operator=(int16_t value) {
        return *this = static_cast<int64_t>(value);
    }
    inline FixedWidthInt &operator=(int8_t value) {
        return *this = static_cast<int64_t>(value);
    }

    inline FixedWidthInt &operator+=(uint32_t offset) {
        return *this += static_cast<uint64_t>(offset);
    }
    inline FixedWidthInt &operator+=(uint16_t offset) {
        return *this += static_cast<uint64_t>(offset);
    }
    inline FixedWidthInt &operator+=(uint8_t offset) {
        return *this += static_cast<uint64_t>(offset);
    }
    inline FixedWidthInt &operator+=(int32_t offset) {
        return *this += static_cast<int64_t>(offset);
    }
    inline FixedWidthInt &operator+=(int16_t offset) {
        return *this += static_cast<int64_t>(offset);
    }
    inline FixedWidthInt &operator+=(int8_t offset) {
        return *this += static_cast<int64_t>(offset);
    }

    inline FixedWidthInt &operator-=(uint32_t offset) {
        return *this -= static_cast<uint64_t>(offset);
    }
    inline FixedWidthInt &operator-=(uint16_t offset) {
        return *this -= static_cast<uint64_t>(offset);
    }
    inline FixedWidthInt &operator-=(uint8_t offset) {
        return *this -= static_cast<uint64_t>(offset);
    }
    inline FixedWidthInt &operator-=(int32_t offset) {
        return *this -= static_cast<int64_t>(offset);
    }
    inline FixedWidthInt &operator-=(int16_t offset) {
        return *this -= static_cast<int64_t>(offset);
    }
    inline FixedWidthInt &operator-=(int8_t offset) {
        return *this -= static_cast<int64_t>(offset);
    }

    inline bool operator==(uint32_t other) const {
        return *this == static_cast<uint64_t>(other);
    }
    inline bool operator==(uint16_t other) const {
        return *this == static_cast<uint64_t>(other);
    }
    inline bool operator==(uint8_t other) const {
        return *this == static_cast<uint64_t>(other);
    }
    inline bool operator==(int32_t other) const {
        return *this == static_cast<int64_t>(other);
    }
    inline bool operator==(int16_t other) const {
        return *this == static_cast<int64_t>(other);
    }
    inline bool operator==(int8_t other) const {
        return *this == static_cast<int64_t>(other);
    }

    inline bool operator<=(uint32_t other) const {
        return *this <= static_cast<uint64_t>(other);
    }
    inline bool operator<=(uint16_t other) const {
        return *this <= static_cast<uint64_t>(other);
    }
    inline bool operator<=(uint8_t other) const {
        return *this <= static_cast<uint64_t>(other);
    }
    inline bool operator<=(int32_t other) const {
        return *this <= static_cast<int64_t>(other);
    }
    inline bool operator<=(int16_t other) const {
        return *this <= static_cast<int64_t>(other);
    }
    inline bool operator<=(int8_t other) const {
        return *this <= static_cast<int64_t>(other);
    }

    inline bool operator>=(uint64_t other) const {
        return !(*this < other);
    }
    inline bool operator>=(int64_t other) const {
        return !(*this < other);
    }
    inline bool operator>=(const FixedWidthInt &other) const {
        return !(*this < other);
    }
    inline bool operator>(uint64_t other) const {
        return !(*this <= other);
    }
    inline bool operator>(int64_t other) const {
        return !(*this <= other);
    }
    inline bool operator>(const FixedWidthInt &other) const {
        return !(*this <= other);
    }

    inline bool operator>=(uint32_t other) const {
        return *this >= static_cast<uint64_t>(other);
    }
    inline bool operator>=(uint16_t other) const {
        return *this >= static_cast<uint64_t>(other);
    }
    inline bool operator>=(uint8_t other) const {
        return *this >= static_cast<uint64_t>(other);
    }
    inline bool operator>=(int32_t other) const {
        return *this >= static_cast<int64_t>(other);
    }
    inline bool operator>=(int16_t other) const {
        return *this >= static_cast<int64_t>(other);
    }
    inline bool operator>=(int8_t other) const {
        return *this >= static_cast<int64_t>(other);
    }

    inline bool operator<(uint32_t other) const {
        return *this < static_cast<uint64_t>(other);
    }
    inline bool operator<(uint16_t other) const {
        return *this < static_cast<uint64_t>(other);
    }
    inline bool operator<(uint8_t other) const {
        return *this < static_cast<uint64_t>(other);
    }
    inline bool operator<(int32_t other) const {
        return *this < static_cast<int64_t>(other);
    }
    inline bool operator<(int16_t other) const {
        return *this < static_cast<int64_t>(other);
    }
    inline bool operator<(int8_t other) const {
        return *this < static_cast<int64_t>(other);
    }

    inline FixedWidthInt &operator<<=(uint32_t other) {
        return *this <<= static_cast<uint64_t>(other);
    }
    inline FixedWidthInt &operator<<=(uint16_t other) {
        return *this <<= static_cast<uint64_t>(other);
    }
    inline FixedWidthInt &operator<<=(uint8_t other) {
        return *this <<= static_cast<uint64_t>(other);
    }
    inline FixedWidthInt &operator<<=(int32_t other) {
        return *this <<= static_cast<int64_t>(other);
    }
    inline FixedWidthInt &operator<<=(int16_t other) {
        return *this <<= static_cast<int64_t>(other);
    }
    inline FixedWidthInt &operator<<=(int8_t other) {
        return *this <<= static_cast<int64_t>(other);
    }

    inline FixedWidthInt &operator>>=(uint32_t other) {
        return *this >>= static_cast<uint64_t>(other);
    }
    inline FixedWidthInt &operator>>=(uint16_t other) {
        return *this >>= static_cast<uint64_t>(other);
    }
    inline FixedWidthInt &operator>>=(uint8_t other) {
        return *this >>= static_cast<uint64_t>(other);
    }
    inline FixedWidthInt &operator>>=(int32_t other) {
        return *this >>= static_cast<int64_t>(other);
    }
    void isigned_right_shift(uint64_t shift);
    inline FixedWidthInt &operator>>=(int16_t other) {
        return *this >>= static_cast<int64_t>(other);
    }
    inline FixedWidthInt &operator>>=(int8_t other) {
        return *this >>= static_cast<int64_t>(other);
    }

    inline FixedWidthInt operator<<(int64_t shift) const {
        FixedWidthInt result(num_bits + (shift > 0 ? shift : 0));
        result ^= *this;
        result <<= shift;
        return result;
    }
    inline FixedWidthInt operator<<(uint64_t shift) const {
        FixedWidthInt result(num_bits + shift);
        result ^= *this;
        result <<= shift;
        return result;
    }
    inline FixedWidthInt operator>>(int64_t shift) const {
        FixedWidthInt result = *this;
        result >>= shift;
        return result;
    }
    inline FixedWidthInt operator>>(uint64_t shift) const {
        FixedWidthInt result = *this;
        result >>= shift;
        return result;
    }
    inline FixedWidthInt operator<<(int32_t shift) const {
        return *this << static_cast<int64_t>(shift);
    }
    inline FixedWidthInt operator<<(uint32_t shift) const {
        return *this << static_cast<uint64_t>(shift);
    }
    inline FixedWidthInt operator>>(int32_t shift) const {
        FixedWidthInt result = *this;
        result >>= shift;
        return result;
    }
    inline FixedWidthInt operator>>(uint32_t shift) const {
        FixedWidthInt result = *this;
        result >>= shift;
        return result;
    }
    FixedWidthInt operator%(const FixedWidthInt &modulus) const;
    FixedWidthInt &operator%=(const FixedWidthInt &modulus);

    std::string decimal() const;
};
FixedWidthInt operator*(const FixedWidthInt &lhs, size_t rhs);
FixedWidthInt operator*(size_t lhs, const FixedWidthInt &rhs);
FixedWidthInt operator-(const FixedWidthInt &lhs, size_t rhs);
FixedWidthInt operator-(size_t lhs, const FixedWidthInt &rhs);
FixedWidthInt operator+(const FixedWidthInt &lhs, size_t rhs);
FixedWidthInt operator+(size_t lhs, const FixedWidthInt &rhs);
std::ostream &operator<<(std::ostream &out, const FixedWidthInt &rhs);

}  // namespace kickmix

#endif
