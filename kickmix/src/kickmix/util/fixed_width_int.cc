#include "kickmix/util/fixed_width_int.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <ostream>
#include <ratio>
#include <sstream>
#include <string>
#include <sys/stat.h>

#include "kickmix/util/word_ops.h"

using namespace kickmix;

FixedWidthInt::FixedWidthInt() : num_bits(0), num_words(0), words(nullptr) {
}

FixedWidthInt::FixedWidthInt(size_t num_bits)
    : num_bits(num_bits), num_words((num_bits + 63) / 64), words(new uint64_t[num_words]) {
    memset(words, 0, num_words * sizeof(uint64_t));
}
FixedWidthInt::FixedWidthInt(size_t num_bits, uint64_t value)
    : num_bits(num_bits), num_words((num_bits + 63) / 64), words(new uint64_t[num_words]) {
    memset(words, 0, num_words * sizeof(uint64_t));
    if (num_bits > 0) {
        words[0] = value;
        clip_hanging_bits();
    }
}
FixedWidthInt::FixedWidthInt(size_t num_bits, uint32_t value)
    : num_bits(num_bits), num_words((num_bits + 63) / 64), words(new uint64_t[num_words]) {
    memset(words, 0, num_words * sizeof(uint64_t));
    if (num_bits > 0) {
        words[0] = value;
        clip_hanging_bits();
    }
}
FixedWidthInt::FixedWidthInt(size_t num_bits, uint16_t value)
    : num_bits(num_bits), num_words((num_bits + 63) / 64), words(new uint64_t[num_words]) {
    memset(words, 0, num_words * sizeof(uint64_t));
    if (num_bits > 0) {
        words[0] = value;
        clip_hanging_bits();
    }
}
FixedWidthInt::FixedWidthInt(size_t num_bits, uint8_t value)
    : num_bits(num_bits), num_words((num_bits + 63) / 64), words(new uint64_t[num_words]) {
    memset(words, 0, num_words * sizeof(uint64_t));
    if (num_bits > 0) {
        words[0] = value;
        clip_hanging_bits();
    }
}
FixedWidthInt::FixedWidthInt(size_t num_bits, int64_t value)
    : num_bits(num_bits), num_words((num_bits + 63) / 64), words(new uint64_t[num_words]) {
    memset(words, 0, num_words * sizeof(uint64_t));
    if (num_bits > 0) {
        words[0] = static_cast<uint64_t>(value);
        clip_hanging_bits();
    }
    if (value < 0) {
        decrement(64);
    }
}
FixedWidthInt::FixedWidthInt(size_t num_bits, int32_t value)
    : num_bits(num_bits), num_words((num_bits + 63) / 64), words(new uint64_t[num_words]) {
    memset(words, 0, num_words * sizeof(uint64_t));
    if (num_bits > 0) {
        words[0] = static_cast<uint64_t>(value);
        clip_hanging_bits();
    }
    if (value < 0) {
        decrement(64);
    }
}
FixedWidthInt::FixedWidthInt(size_t num_bits, int16_t value)
    : num_bits(num_bits), num_words((num_bits + 63) / 64), words(new uint64_t[num_words]) {
    memset(words, 0, num_words * sizeof(uint64_t));
    if (num_bits > 0) {
        words[0] = static_cast<uint64_t>(value);
        clip_hanging_bits();
    }
    if (value < 0) {
        decrement(64);
    }
}
FixedWidthInt::FixedWidthInt(size_t num_bits, int8_t value)
    : num_bits(num_bits), num_words((num_bits + 63) / 64), words(new uint64_t[num_words]) {
    memset(words, 0, num_words * sizeof(uint64_t));
    if (num_bits > 0) {
        words[0] = static_cast<uint64_t>(value);
        clip_hanging_bits();
    }
    if (value < 0) {
        decrement(64);
    }
}
FixedWidthInt::FixedWidthInt(FixedWidthInt &&value) noexcept
    : num_bits(value.num_bits), num_words(value.num_words), words(value.words) {
    value.words = nullptr;
    value.num_words = 0;
    value.num_bits = 0;
}
FixedWidthInt::FixedWidthInt(const FixedWidthInt &value)
    : num_bits(value.num_bits), num_words(value.num_words), words(new uint64_t[value.num_words]) {
    if (num_words > 0) {
        memcpy(words, value.words, sizeof(uint64_t) * num_words);
    }
}
FixedWidthInt &FixedWidthInt::operator=(const FixedWidthInt &value) {
    if (this == &value) {
        return *this;
    }
    if (num_words != value.num_words) {
        if (words != nullptr) {
            delete[] words;
        }
        words = new uint64_t[value.num_words];
        num_words = value.num_words;
    }
    if (value.num_words > 0) {
        memcpy(words, value.words, sizeof(uint64_t) * value.num_words);
    }
    num_bits = value.num_bits;
    return *this;
}
FixedWidthInt &FixedWidthInt::operator=(FixedWidthInt &&value) noexcept {
    if (this == &value) {
        return *this;
    }
    if (words != nullptr) {
        delete[] words;
    }
    words = value.words;
    num_words = value.num_words;
    num_bits = value.num_bits;

    value.words = nullptr;
    value.num_words = 0;
    value.num_bits = 0;

    return *this;
}
FixedWidthInt::FixedWidthInt(size_t num_bits, std::string_view text) : num_bits(0), num_words(0), words(nullptr) {
    *this = std::move(from_str(text, num_bits));
}
FixedWidthInt::FixedWidthInt(std::string_view text) : num_bits(0), num_words(0), words(nullptr) {
    *this = std::move(from_str(text));
}
FixedWidthInt::~FixedWidthInt() {
    if (words != nullptr) {
        delete[] words;
        words = nullptr;
        num_words = 0;
        num_bits = 0;
    }
}

FixedWidthInt FixedWidthInt::from_str(std::string_view text, std::optional<size_t> num_bits) {
    // Trim.
    while (text.starts_with(' ')) {
        text = text.substr(1);
    }
    while (text.ends_with(' ')) {
        text = text.substr(0, text.size() - 1);
    }

    if (text.starts_with("0b")) {
        return from_str_binary(text, num_bits);
    } else if (text.starts_with("0x")) {
        return from_str_hex(text, num_bits);
    } else if (!text.empty()) {
        return from_str_dec(text, num_bits);
    } else {
        throw std::invalid_argument("Empty string can't be parsed as an int.");
    }
}

FixedWidthInt FixedWidthInt::from_str_binary(const std::string_view text, std::optional<size_t> num_bits) {
    std::string_view rest = text;
    if (rest.starts_with("0b")) {
        rest = rest.substr(2);
    }

    // Determine size to allocate.
    size_t actual_num_bits = 0;
    for (char c : rest) {
        actual_num_bits += c == '0' || c == '1';
    }
    if (num_bits.has_value() && actual_num_bits > num_bits.value()) {
        std::stringstream ss;
        ss << "The text '" << text << "' specifies ";
        ss << actual_num_bits << " bits, which is more than the expected number of bits (";
        ss << num_bits.value() << ").";
        throw std::invalid_argument(ss.str());
    }
    FixedWidthInt result(num_bits.value_or(actual_num_bits));

    // Fill contents.
    size_t bit_pos = 0;
    for (size_t k = rest.size(); k--;) {
        char c = rest[k];
        if (c == '_') {
            continue;
        }
        if (c == '1') {
            result.words[bit_pos / 64] |= static_cast<uint64_t>(1) << (bit_pos % 64);
            bit_pos += 1;
        } else if (c == '0') {
            bit_pos += 1;
        } else {
            std::stringstream ss;
            ss << "Invalid character '" << c << "' (" << static_cast<int>(c) << ") in '" << text << "'";
            throw std::invalid_argument(ss.str());
        }
    }

    return result;
}

FixedWidthInt FixedWidthInt::from_str_hex(const std::string_view text, std::optional<size_t> num_bits) {
    std::string_view rest = text;
    if (rest.starts_with("0x")) {
        rest = rest.substr(2);
    }

    // Determine size to allocate.
    size_t actual_num_bits = 0;
    uint8_t clip = 0xFF;
    for (char c : rest) {
        bool is_digit = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
        actual_num_bits += is_digit;
        if (clip == 0xFF && is_digit) {
            if (c >= '0' && c <= '1') {
                clip = 3;
            } else if (c >= '2' && c <= '3') {
                clip = 2;
            } else if (c >= '4' && c <= '7') {
                clip = 1;
            } else {
                clip = 0;
            }
        }
    }
    if (clip == 0xFF) {
        clip = 0;
    }

    actual_num_bits *= 4;
    actual_num_bits -= clip;
    if (num_bits.has_value() && actual_num_bits > num_bits.value()) {
        std::stringstream ss;
        ss << "The text '" << text << "' specifies at least ";
        ss << actual_num_bits << " bits, which is more than the expected number of bits (";
        ss << num_bits.value() << ").";
        throw std::invalid_argument(ss.str());
    }
    FixedWidthInt result(num_bits.value_or(actual_num_bits));

    // Fill contents.
    size_t bit_pos = 0;
    for (size_t k = rest.size(); k--;) {
        char c = rest[k];
        if (c == '_') {
            continue;
        }
        uint64_t hex_digit;
        if (c >= '0' && c <= '9') {
            hex_digit = c - '0';
        } else if (c >= 'A' && c <= 'F') {
            hex_digit = c - 'A' + 10;
        } else if (c >= 'a' && c <= 'f') {
            hex_digit = c - 'a' + 10;
        } else {
            std::stringstream ss;
            ss << "Invalid character '" << c << "' (" << static_cast<int>(c) << ") in '" << text << "'";
            throw std::invalid_argument(ss.str());
        }
        result.words[bit_pos / 64] ^= hex_digit << (bit_pos % 64);
        bit_pos += 4;
    }
    if (result.clip_hanging_bits()) {
        throw std::invalid_argument("Unexpected hanging bits in from_str_hex");
    }

    return result;
}

FixedWidthInt FixedWidthInt::from_str_dec(const std::string_view text, std::optional<size_t> num_bits) {
    FixedWidthInt result(num_bits.value_or(text.size() * 4));
    for (auto c : text) {
        if (c == '_') {
            continue;
        }
        uint64_t digit;
        if (c >= '0' && c <= '9') {
            digit = c - '0';
        } else {
            std::stringstream ss;
            ss << "Invalid character '" << c << "' (" << static_cast<int>(c) << ") in '" << text << "'";
            throw std::invalid_argument(ss.str());
        }

        result <<= 1;
        result.inplace_times5();
        result += digit;
    }
    if (!num_bits.has_value()) {
        result.resize_shrink(result.num_bits_in_use());
    }
    return result;
}

bool FixedWidthInt::decrement(size_t bit_offset) {
    if (bit_offset >= num_bits) {
        return true;
    }
    size_t k = bit_offset / 64;
    size_t offset = static_cast<uint64_t>(1) << (bit_offset & 63);
    auto old_value = words[k];
    words[k] = old_value - offset;
    bool borrow = false;
    if (words[k] > old_value) {
        borrow = true;
        // Carry into later words.
        k += 1;
        while (k < num_words) {
            words[k]--;
            if (words[k] != UINT64_MAX) {
                borrow = false;
                break;
            }
            k++;
        }
    }
    return clip_hanging_bits() || borrow;
}

bool FixedWidthInt::non_zero() const {
    for (size_t k = 0; k < num_words; k++) {
        if (words[k]) {
            return true;
        }
    }
    return false;
}
FixedWidthInt::operator bool() const {
    return non_zero();
}
FixedWidthInt::operator uint64_t() const {
    if (num_bits_in_use() > 64) {
        throw std::invalid_argument("Integer is too large to fit into a uint64_t.");
    }
    if (num_words == 0) {
        return 0;
    }
    return words[0];
}

double FixedWidthInt::to_approx_mantissa() const {
    size_t n = num_bits_in_use();
    if (n == 0) {
        return 0.0;
    }
    uint64_t leading_bits = read_bit_shifted_word(static_cast<int64_t>(n) - 53);
    leading_bits &= UINT64_MAX >> (64 - 52);
    leading_bits |= 1023ull << 52;
    return std::bit_cast<double>(leading_bits);
}

FixedWidthInt::operator double() const {
    size_t n = num_bits_in_use();
    if (n == 0) {
        return 0.0;
    }
    if (n >= 1025) {
        throw std::invalid_argument("Integer is too large to convert to a double (>= pow(2, 1024)).");
    }
    uint64_t leading_bits = read_bit_shifted_word(static_cast<int64_t>(n) - 53);
    leading_bits &= UINT64_MAX >> (64 - 52);
    leading_bits |= (n + 1022ull) << 52;
    return std::bit_cast<double>(leading_bits);
}

bool FixedWidthInt::increment(size_t bit_offset) {
    if (bit_offset >= num_bits) {
        return true;
    }
    size_t k = bit_offset / 64;
    size_t offset = static_cast<uint64_t>(1) << (bit_offset & 63);
    words[k] += offset;
    bool carry = words[k] < offset;
    if (carry) {
        // Carry into later words.
        k += 1;
        while (k < num_words) {
            words[k]++;
            if (words[k] != 0) {
                carry = false;
                break;
            }
            k++;
        }
    }
    return clip_hanging_bits() || carry;
}

void FixedWidthInt::negate_mod(const FixedWidthInt &modulus) {
    negate();
    *this += modulus;
    if (*this == modulus) {
        *this ^= modulus;
    }
}
void FixedWidthInt::negate() {
    for (size_t k = 0; k < num_words; k++) {
        words[k] = ~words[k];
    }
    increment();
}

bool FixedWidthInt::operator[](size_t index) const {
    size_t word_index = index / 64;
    size_t bit_index = index % 64;
    if (word_index >= num_words) {
        return false;
    }
    return (words[word_index] >> bit_index) & 1;
}

BitWordRef FixedWidthInt::bit_ref(size_t index) {
    if (index >= num_bits) {
        throw std::invalid_argument("index >= num_bits");
    }
    return BitWordRef(&words[index / 64], index % 64);
}

BitWordRef FixedWidthInt::front_ref() {
    return bit_ref(0);
}

BitWordRef FixedWidthInt::back_ref() {
    return bit_ref(num_bits - 1);
}

void FixedWidthInt::flip_bit(size_t index) {
    size_t word_index = index / 64;
    if (word_index >= num_words) {
        return;
    }
    size_t bit_index = index % 64;
    words[word_index] ^= static_cast<uint64_t>(1) << bit_index;
}

void FixedWidthInt::set_bit(size_t index, bool new_value) {
    size_t word_index = index / 64;
    if (word_index >= num_words) {
        return;
    }
    size_t bit_index = index % 64;
    if (new_value) {
        words[word_index] |= static_cast<uint64_t>(1) << bit_index;
    } else {
        words[word_index] &= ~(static_cast<uint64_t>(1) << bit_index);
    }
}

void FixedWidthInt::reset_bit(size_t index) {
    size_t word_index = index / 64;
    if (word_index >= num_words) {
        return;
    }
    size_t bit_index = index % 64;
    words[word_index] &= ~(static_cast<uint64_t>(1) << bit_index);
}

FixedWidthInt &FixedWidthInt::operator+=(const FixedWidthInt &offset) {
    iadd_carry(offset);
    return *this;
}
bool FixedWidthInt::iadd_carry(const FixedWidthInt &offset, bool carry_in) {
    size_t k = 0;
    char carry = carry_in;
    for (; k < offset.num_words && k < num_words; k++) {
        unsigned long long r;
        carry = add_carry_u64(carry, words[k], offset.words[k], &r);
        words[k] = r;
    }
    if (carry) {
        return increment(64 * k);
    }
    return clip_hanging_bits();
}

bool FixedWidthInt::iadd_carry(uint64_t offset, bool carry_in) {
    size_t k = 0;
    char carry = carry_in;
    unsigned long long r;
    carry = add_carry_u64(carry, words[k], offset, &r);
    words[k] = r;
    if (carry) {
        return increment(64 * k);
    }
    return clip_hanging_bits();
}

void FixedWidthInt::write_from(const FixedWidthInt &other) {
    clear_to_zero();
    *this ^= other;
}

FixedWidthInt &FixedWidthInt::operator^=(int mask) {
    if (num_words > 0) {
        words[0] ^= static_cast<uint64_t>(mask);
    }
    if (mask < 0) {
        for (size_t k = 1; k < num_words; k++) {
            words[k] ^= UINT64_MAX;
        }
    }
    clip_hanging_bits();
    return *this;
}
FixedWidthInt &FixedWidthInt::operator^=(const FixedWidthInt &offset) {
    size_t n = std::min(num_words, offset.num_words);
    for (size_t k = 0; k < n; k++) {
        words[k] ^= offset.words[k];
    }
    clip_hanging_bits();
    return *this;
}
FixedWidthInt &FixedWidthInt::operator&=(const FixedWidthInt &offset) {
    size_t n = std::min(num_words, offset.num_words);
    for (size_t k = 0; k < n; k++) {
        words[k] &= offset.words[k];
    }
    clip_hanging_bits();
    return *this;
}
FixedWidthInt &FixedWidthInt::operator|=(const FixedWidthInt &offset) {
    size_t n = std::min(num_words, offset.num_words);
    for (size_t k = 0; k < n; k++) {
        words[k] |= offset.words[k];
    }
    clip_hanging_bits();
    return *this;
}

void FixedWidthInt::iadd_mod(const FixedWidthInt &offset, const FixedWidthInt &modulus) {
    if (*this >= modulus) {
        throw std::invalid_argument("iadd_mod requires *this < modulus");
    }
    if (offset > modulus) {
        iadd_mod(offset % modulus, modulus);
        return;
    }

    bool carry = iadd_carry(offset);
    if (carry || *this >= modulus) {
        *this -= modulus;
    }
}

void FixedWidthInt::idouble_mod(const FixedWidthInt &modulus) {
    if (iadd_carry(*this) || *this >= modulus) {
        *this -= modulus;
    }
}

void FixedWidthInt::ihalve_mod(const FixedWidthInt &modulus) {
    if (!modulus[0]) {
        throw std::invalid_argument("Modulus must be odd for there to be a multiplicative inverse of 2.");
    }
    bool b = (*this)[0];
    if (b) {
        b = iadd_carry(modulus);
    }
    *this >>= 1;
    if (b) {
        set_bit(modulus.num_bits_in_use() - 1);
    }
}

void FixedWidthInt::isub_mod(const FixedWidthInt &offset, const FixedWidthInt &modulus) {
    if (*this >= modulus) {
        throw std::invalid_argument("isub_mod requires *this < modulus");
    }
    if (offset >= modulus) {
        isub_mod(offset % modulus, modulus);
        return;
    }

    if (isub_borrow(offset)) {
        *this += modulus;
    }
}

FixedWidthInt &FixedWidthInt::iadd_shifted(const FixedWidthInt &offset, size_t shift) {
    size_t k = shift / 64;
    char carry = 0;
    for (; k < num_words; k++) {
        unsigned long long r;
        carry = add_carry_u64(
            carry,
            words[k],
            offset.read_bit_shifted_word(static_cast<int64_t>(k) * 64 - static_cast<int64_t>(shift)),
            &r);
        words[k] = r;
    }
    if (carry) {
        increment(64 * k);
    }

    clip_hanging_bits();
    return *this;
}

FixedWidthInt &FixedWidthInt::isub_shifted(const FixedWidthInt &offset, size_t shift, bool *borrow_out) {
    size_t k = shift / 64;
    char borrow = 0;
    for (; k < num_words; k++) {
        unsigned long long r;
        borrow = sub_borrow_u64(
            borrow,
            words[k],
            offset.read_bit_shifted_word(static_cast<int64_t>(k) * 64 - static_cast<int64_t>(shift)),
            &r);
        words[k] = r;
    }
    if (borrow) {
        decrement(64 * k);
    }
    *borrow_out = borrow != 0;

    clip_hanging_bits();
    return *this;
}

FixedWidthInt &FixedWidthInt::operator+=(uint64_t offset) {
    if (num_words == 0) {
        return *this;
    }
    words[0] += offset;
    if (words[0] < offset) {
        // Carry into later words.
        increment(64);
    }

    clip_hanging_bits();
    return *this;
}

FixedWidthInt &FixedWidthInt::operator+=(int64_t offset) {
    if (num_words == 0) {
        return *this;
    }
    uint64_t u_offset = static_cast<uint64_t>(offset);
    words[0] += u_offset;

    // Carry into later words.
    if (offset >= 0 && words[0] < u_offset) {
        increment(64);
    } else if (offset < 0 && words[0] > u_offset) {
        decrement(64);
    }

    clip_hanging_bits();
    return *this;
}
FixedWidthInt &FixedWidthInt::operator-=(const FixedWidthInt &offset) {
    isub_borrow(offset);
    return *this;
}
bool FixedWidthInt::isub_borrow(const FixedWidthInt &offset) {
    size_t k = 0;
    char borrow = 0;
    for (; k < offset.num_words && k < num_words; k++) {
        unsigned long long r;
        borrow = sub_borrow_u64(borrow, words[k], offset.words[k], &r);
        words[k] = r;
    }
    if (borrow) {
        borrow = decrement(64 * k);
    }

    clip_hanging_bits();
    return borrow;
}

FixedWidthInt &FixedWidthInt::operator-=(uint64_t offset) {
    if (num_words == 0) {
        return *this;
    }
    auto old_value = words[0];
    words[0] -= offset;
    if (words[0] > old_value) {
        // Carry into later words.
        decrement(64);
    }

    clip_hanging_bits();
    return *this;
}

FixedWidthInt &FixedWidthInt::operator-=(int64_t offset) {
    if (num_words == 0) {
        return *this;
    }
    auto old_value = words[0];
    words[0] -= static_cast<uint64_t>(offset);

    // Carry into later words.
    if (offset >= 0 && words[0] > old_value) {
        decrement(64);
    } else if (offset < 0 && words[0] < old_value) {
        increment(64);
    }

    clip_hanging_bits();
    return *this;
}

std::string FixedWidthInt::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

std::string FixedWidthInt::bin() const {
    std::stringstream ss;
    for (size_t k = num_bits; k--;) {
        uint8_t digit = (words[k / 64] >> (k % 64)) & 1;
        ss << "01"[digit];
        if (k && k % 8 == 0) {
            ss << '_';
        }
    }
    return ss.str();
}

std::string FixedWidthInt::hex() const {
    std::stringstream ss;
    size_t k = num_bits;
    while (k & 3) {
        k++;
    }
    while (k) {
        k -= 4;
        uint8_t digit = (words[k / 64] >> (k % 64)) & 0xF;
        ss << "0123456789ABCDEF"[digit];
        if (k && k % 32 == 0) {
            ss << '_';
        }
    }
    return ss.str();
}

std::ostream &kickmix::operator<<(std::ostream &out, const FixedWidthInt &rhs) {
    out << "0x";
    bool first = true;
    for (size_t k = rhs.num_words; k--;) {
        if (first) {
            first = false;
        } else {
            out << '_';
        }
        for (size_t b = 0; b < 64; b += 4) {
            size_t k2 = 64 - 4 - b;
            if (k * 64 + k2 >= rhs.num_bits) {
                continue;
            }
            uint8_t digit = (rhs.words[k] >> k2) & 0xF;
            out << "0123456789ABCDEF"[digit];
        }
    }
    out << " [num_bits=";
    out << rhs.num_bits;
    out << "]";

    out << " (decimal=" << rhs.decimal() << ")";
    return out;
}

bool FixedWidthInt::operator<(const FixedWidthInt &other) const {
    size_t n = std::min(num_words, other.num_words);
    for (size_t k = n; k < num_words; k++) {
        if (words[k]) {
            return false;
        }
    }
    for (size_t k = n; k < other.num_words; k++) {
        if (other.words[k]) {
            return true;
        }
    }
    for (size_t k = n; k--;) {
        if (words[k] != other.words[k]) {
            return words[k] < other.words[k];
        }
    }
    return false;
}
bool FixedWidthInt::operator<=(const FixedWidthInt &other) const {
    size_t n = std::min(num_words, other.num_words);
    ;
    for (size_t k = n; k < num_words; k++) {
        if (words[k]) {
            return false;
        }
    }
    for (size_t k = n; k < other.num_words; k++) {
        if (other.words[k]) {
            return true;
        }
    }
    for (size_t k = n; k--;) {
        if (words[k] != other.words[k]) {
            return words[k] < other.words[k];
        }
    }
    return true;
}
bool FixedWidthInt::operator==(const FixedWidthInt &other) const {
    size_t n = std::min(num_words, other.num_words);
    for (size_t k = 0; k < n; k++) {
        if (words[k] != other.words[k]) {
            return false;
        }
    }
    for (size_t k = n; k < num_words; k++) {
        if (words[k]) {
            return false;
        }
    }
    for (size_t k = n; k < other.num_words; k++) {
        if (other.words[k]) {
            return false;
        }
    }
    return true;
}
bool FixedWidthInt::operator==(uint64_t other) const {
    if (num_words == 0) {
        return 0 == other;
    }
    if (words[0] != other) {
        return false;
    }
    for (size_t k = 1; k < num_words; k++) {
        if (words[k]) {
            return false;
        }
    }
    return true;
}
bool FixedWidthInt::operator==(int64_t other) const {
    if (other < 0) {
        return false;
    }
    return *this == static_cast<uint64_t>(other);
}

bool FixedWidthInt::operator<(uint64_t other) const {
    if (num_words == 0) {
        return 0 < other;
    }
    for (size_t k = 1; k < num_words; k++) {
        if (words[k]) {
            return false;
        }
    }
    return words[0] < other;
}
bool FixedWidthInt::operator<(int64_t other) const {
    if (other < 0) {
        return false;
    }
    return *this < static_cast<uint64_t>(other);
}

bool FixedWidthInt::operator<=(uint64_t other) const {
    if (num_words == 0) {
        return true;
    }
    for (size_t k = 1; k < num_words; k++) {
        if (words[k]) {
            return false;
        }
    }
    return words[0] <= other;
}
bool FixedWidthInt::operator<=(int64_t other) const {
    if (other < 0) {
        return false;
    }
    return *this <= static_cast<uint64_t>(other);
}

bool FixedWidthInt::inplace_times5() {
    uint64_t cur_offset = 0;
    uint64_t rollover = 0;
    unsigned long long carry = 0;
    for (size_t k = 0; k < num_words; k++) {
        unsigned long long result;
        cur_offset = (words[k] << 2) | rollover;
        rollover = words[k] >> 62;
        carry = add_carry_u64(carry, words[k], cur_offset, &result);
        words[k] = result;
    }
    return clip_hanging_bits() || carry != 0 || rollover != 0;
}

void FixedWidthInt::inplace_div5() {
    uint64_t rollover = 0;
    for (size_t k = num_words; k--;) {
        uint64_t rem = words[k] % 5;
        words[k] /= 5;
        words[k] += 0x3333333333333333 * rollover;
        rem += rollover;
        words[k] += rem / 5;
        rem %= 5;
        rollover = rem;
    }
}

void FixedWidthInt::resize_clear(size_t new_num_bits) {
    size_t new_num_words = (new_num_bits + 63) / 64;
    if (new_num_words != num_words) {
        auto new_words = new uint64_t[new_num_words];
        delete[] words;
        words = new_words;
        num_words = new_num_words;
    }
    num_bits = new_num_bits;
    clear_to_zero();
}

void FixedWidthInt::resize_shrink(size_t new_num_bits) {
    if (new_num_bits > num_bits) {
        throw std::invalid_argument("new_num_bits > num_bits");
    }
    size_t new_num_words = (new_num_bits + 63) / 64;
    num_bits = new_num_bits;
    num_words = new_num_words;
    clip_hanging_bits();
}

void FixedWidthInt::clear_to_zero() {
    if (num_words > 0) {
        memset(words, 0, num_words * sizeof(uint64_t));
    }
}

FixedWidthInt &FixedWidthInt::operator=(uint64_t value) {
    memset(words, 0, num_words * sizeof(uint64_t));
    if (num_bits > 0) {
        words[0] = value;
        clip_hanging_bits();
    }
    words[0] = static_cast<uint64_t>(value);
    clip_hanging_bits();
    return *this;
}
FixedWidthInt &FixedWidthInt::operator=(int64_t value) {
    memset(words, 0, num_words * sizeof(uint64_t));
    if (num_bits > 0) {
        words[0] = static_cast<uint64_t>(value);
        clip_hanging_bits();
    }
    if (value < 0) {
        decrement(64);
    }
    clip_hanging_bits();
    return *this;
}

FixedWidthInt FixedWidthInt::random(size_t num_bits, std::mt19937_64 &rng) {
    FixedWidthInt result(num_bits);
    for (size_t k = 0; k < result.num_words; k++) {
        result.words[k] = rng();
    }
    result.clip_hanging_bits();
    return result;
}

FixedWidthInt FixedWidthInt::random_mod(std::mt19937_64 &rng, const FixedWidthInt &modulus) {
    FixedWidthInt result(modulus.num_bits);
    result.randomize_mod(rng, modulus);
    return result;
}

void FixedWidthInt::randomize(std::mt19937_64 &rng) {
    for (size_t k = 0; k < num_words; k++) {
        words[k] = rng();
    }
    clip_hanging_bits();
}

size_t FixedWidthInt::num_bits_in_use() const {
    for (size_t k = num_words; k--;) {
        if (words[k] != 0) {
            return k * 64 + (64 - std::countl_zero(words[k]));
        }
    }
    return 0;
}

FixedWidthInt FixedWidthInt::mult_inverse(const FixedWidthInt &modulus) const {
    if (!modulus[0]) {
        throw std::invalid_argument("mult_inverse not implemented: even modulus.");
    }
    if (*this >= modulus) {
        return (*this % modulus).mult_inverse(modulus);
    }

    size_t n = modulus.num_bits;
    FixedWidthInt u(n);
    FixedWidthInt v(n);
    FixedWidthInt u2(n);
    FixedWidthInt v2(n);
    u ^= *this;
    v ^= modulus;
    u2 ^= 1;
    if (!u[0]) {
        std::swap(u, v);
        std::swap(u2, v2);
    }
    while (v.non_zero()) {
        if (!v[0]) {
            v >>= 1;
            v2.ihalve_mod(modulus);
        } else if (v > u) {
            v -= u;
            v >>= 1;

            v2.isub_mod(u2, modulus);
            v2.ihalve_mod(modulus);
        } else {
            std::swap(u, v);
            v -= u;
            v >>= 1;

            std::swap(u2, v2);
            v2.isub_mod(u2, modulus);
            v2.ihalve_mod(modulus);
        }
    }
    return u2;
}

bool FixedWidthInt::is_coprime_to(const FixedWidthInt &other) const {
    size_t n = std::max(num_bits, other.num_bits);
    FixedWidthInt u(n);
    FixedWidthInt v(n);
    u ^= *this;
    v ^= other;
    if (!u[0]) {
        std::swap(u, v);
    }
    if (!u[0]) {
        return false;
    }
    while (v.non_zero()) {
        if (!v[0]) {
            v >>= 1;
        } else if (v > u) {
            v -= u;
            v >>= 1;
        } else {
            std::swap(u, v);
            v -= u;
            v >>= 1;
        }
    }
    return u == 1;
}

void FixedWidthInt::randomize_coprime_mod(std::mt19937_64 &rng, const FixedWidthInt &modulus) {
    while (true) {
        randomize_mod(rng, modulus);
        if (is_coprime_to(modulus)) {
            return;
        }
    }
}

void FixedWidthInt::randomize_mod(std::mt19937_64 &rng, const FixedWidthInt &modulus) {
    size_t num_mod_bits = modulus.num_bits_in_use();
    if (num_mod_bits == 0) {
        throw std::invalid_argument("modulus == 0");
    }
    if (num_bits < num_mod_bits) {
        std::stringstream ss;
        ss << "num_bits=" << num_bits << " < num_mod_bits=" << num_mod_bits;
        throw std::invalid_argument(ss.str());
    }
    for (size_t k = 0; k < num_words; k++) {
        words[k] = 0;
    }
    while (true) {
        size_t num_mod_words = (num_mod_bits + 63) / 64;
        for (size_t k = 0; k < num_mod_words; k++) {
            words[k] = rng();
        }
        if (num_mod_bits & 63) {
            words[num_mod_words - 1] &= ~(-static_cast<uint64_t>(1) << (num_mod_bits & 63));
        }
        if (*this < modulus) {
            return;
        }
    }
}

FixedWidthInt &FixedWidthInt::operator<<=(int64_t shift) {
    if (shift < 0) {
        return *this >>= static_cast<uint64_t>(-shift);
    } else {
        return *this <<= static_cast<uint64_t>(shift);
    }
}
FixedWidthInt &FixedWidthInt::operator>>=(int64_t shift) {
    if (shift < 0) {
        return *this <<= static_cast<uint64_t>(-shift);
    } else {
        return *this >>= static_cast<uint64_t>(shift);
    }
}
FixedWidthInt &FixedWidthInt::operator<<=(uint64_t shift) {
    for (size_t k = num_words; k--;) {
        words[k] = read_bit_shifted_word(static_cast<int64_t>(k * 64 - shift));
    }
    clip_hanging_bits();
    return *this;
}
uint64_t FixedWidthInt::read_bit_shifted_word(size_t offset) const {
    return read_bit_shifted_word(static_cast<int64_t>(offset));
}
uint64_t FixedWidthInt::read_bit_shifted_word(int64_t offset) const {
    if (offset < 0) {
        if (offset <= -64 || num_words == 0) {
            return 0;
        }
        return words[0] << -offset;
    }
    size_t word_offset = offset / 64;
    if (word_offset >= num_words) {
        return 0;
    }
    size_t bit_offset = offset % 64;
    uint64_t result = words[word_offset];
    if (bit_offset) {
        result >>= bit_offset;
        if (word_offset + 1 < num_words) {
            result ^= words[word_offset + 1] << (64 - bit_offset);
        }
    }
    return result;
}

void FixedWidthInt::isigned_right_shift(uint64_t shift) {
    if (num_words == 0) {
        return;
    }
    bool shifted_in_val = (*this)[num_bits - 1];
    for (size_t k = 0; k < num_words; k++) {
        words[k] = read_bit_shifted_word(static_cast<size_t>(shift + k * 64));
    }
    if (shifted_in_val) {
        for (size_t k = 0; k < shift && k < num_bits; k++) {
            set_bit(num_bits - k - 1);
        }
    }
    clip_hanging_bits();
}

FixedWidthInt &FixedWidthInt::operator>>=(uint64_t shift) {
    for (size_t k = 0; k < num_words; k++) {
        words[k] = read_bit_shifted_word(static_cast<size_t>(shift + k * 64));
    }
    clip_hanging_bits();
    return *this;
}

FixedWidthInt &FixedWidthInt::operator*=(const FixedWidthInt &value) {
    for (size_t k = num_bits; k--;) {
        if ((*this)[k]) {
            flip_bit(k);
            iadd_shifted(value, k);
        }
    }
    return *this;
}

inline void full_fma64(
    uint64_t factor1, uint64_t factor2, uint64_t offset1, uint64_t offset2, uint64_t *out_low, uint64_t *out_high) {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

    auto result = factor1 * (unsigned __int128)factor2 + (unsigned __int128)offset1 + offset2;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

    *out_high = static_cast<uint64_t>(result >> 64);
    *out_low = static_cast<uint64_t>(result);
}

FixedWidthInt &FixedWidthInt::operator*=(uint64_t value) {
    uint64_t carry = 0;
    for (size_t k = 0; k < num_words; k++) {
        full_fma64(value, words[k], carry, 0, &words[k], &carry);
    }
    clip_hanging_bits();
    return *this;
}

FixedWidthInt FixedWidthInt::operator*(const FixedWidthInt &value) const {
    FixedWidthInt result(num_bits + value.num_bits, 0);
    result ^= *this;
    result *= value;
    return result;
}

void FixedWidthInt::isub_product(uint64_t v1, const FixedWidthInt &v2) {
    *this ^= -1;
    iadd_product(v1, v2);
    *this ^= -1;
}
void FixedWidthInt::isub_product(const FixedWidthInt &v1, uint64_t v2) {
    *this ^= -1;
    iadd_product(v1, v2);
    *this ^= -1;
}
void FixedWidthInt::isub_product_shifted(const FixedWidthInt &v1, uint64_t v2, size_t shift) {
    *this ^= -1;
    iadd_product_shifted(v1, v2, shift);
    *this ^= -1;
}
void FixedWidthInt::iadd_product(uint64_t v1, const FixedWidthInt &v2) {
    size_t m = std::min(num_words, v2.num_words);
    uint64_t carry = 0;
    size_t k2 = 0;
    while (k2 < m) {
        full_fma64(v1, v2.words[k2], carry, words[k2], &words[k2], &carry);
        k2++;
    }
    while (k2 < num_words && carry) {
        unsigned long long out;
        carry = add_carry_u64(0, words[k2], carry, &out);
        words[k2] = out;
        k2++;
    }
    clip_hanging_bits();
}

void FixedWidthInt::iadd_product_shifted(const FixedWidthInt &v1, uint64_t v2, size_t shift) {
    iadd_product_shifted(v2, v1, shift);
}
void FixedWidthInt::isub_product_shifted(uint64_t v1, const FixedWidthInt &v2, size_t shift) {
    isub_product_shifted(v2, v1, shift);
}
void FixedWidthInt::iadd_product_shifted(uint64_t v1, const FixedWidthInt &v2, size_t shift) {
    uint64_t carry = 0;
    size_t k2 = 0;
    while (k2 < num_words) {
        full_fma64(
            v1,
            v2.read_bit_shifted_word(static_cast<int64_t>(k2 * 64) - static_cast<int64_t>(shift)),
            carry,
            words[k2],
            &words[k2],
            &carry);
        k2++;
    }
    while (k2 < num_words && carry) {
        unsigned long long out;
        carry = add_carry_u64(0, words[k2], carry, &out);
        words[k2] = out;
        k2++;
    }
    clip_hanging_bits();
}

void FixedWidthInt::iadd_product(const FixedWidthInt &v1, uint64_t v2) {
    iadd_product(v2, v1);
}

void FixedWidthInt::iadd_product(const FixedWidthInt &v1, const FixedWidthInt &v2) {
    size_t n = std::min(num_words, v1.num_words);
    for (size_t k1 = 0; k1 < n; k1++) {
        size_t m = std::min(num_words - k1, v2.num_words);
        uint64_t carry = 0;
        size_t k2 = 0;
        while (k2 < m) {
            full_fma64(v1.words[k1], v2.words[k2], carry, words[k1 + k2], &words[k1 + k2], &carry);
            k2++;
        }
        k2 += k1;
        while (k2 < num_words && carry) {
            unsigned long long out;
            carry = add_carry_u64(0, words[k2], carry, &out);
            words[k2] = out;
            k2++;
        }
    }
    clip_hanging_bits();
}

FixedWidthInt FixedWidthInt::operator+(const FixedWidthInt &value) const {
    FixedWidthInt result(num_bits + value.num_bits + 1, 0);
    result ^= *this;
    result += value;
    return result;
}

FixedWidthInt FixedWidthInt::operator-() const {
    // Note: bits increased by +1 to ensure sign can be recovered by caller.
    FixedWidthInt result(num_bits + 1, 0);
    result -= *this;
    return result;
}

FixedWidthInt FixedWidthInt::operator-(const FixedWidthInt &value) const {
    // Note: bits increased by +2 instead of +1 to ensure sign can be recovered by caller.
    FixedWidthInt result(num_bits + value.num_bits + 2, 0);
    result ^= *this;
    result -= value;
    return result;
}

FixedWidthInt FixedWidthInt::operator%(const FixedWidthInt &modulus) const {
    FixedWidthInt result = *this;
    result %= modulus;
    FixedWidthInt sized_result(modulus.num_bits);
    sized_result ^= result;
    return sized_result;
}

FixedWidthInt &FixedWidthInt::operator%=(const FixedWidthInt &modulus) {
    if (!modulus) {
        throw std::invalid_argument("Division by zero.");
    }
    bool neg = false;
    double d_b = modulus.to_approx_mantissa();
    size_t n_b = modulus.num_bits_in_use();
    constexpr double POW2_53 = 0x1p53;

    // Process ~52 bits at a time using the double approximation
    // as a guide for word-sized FMAs to remove multiples of the
    // modulus from *this value.
    size_t n_a = num_bits_in_use();
    while (n_a > n_b) {
        double d_a = to_approx_mantissa();
        int64_t s = static_cast<int64_t>(n_a - n_b);
        double d_r = d_a / d_b;
        s -= 53;
        d_r *= POW2_53;
        if (s < 0) {
            d_r *= pow(2, s);
            s = 0;
        }
        uint64_t r = static_cast<uint64_t>(floor(d_r));
        isub_product_shifted(modulus, r, s);

        // If the subtraction underflowed, negate the number.
        // Keep track of how many negations, to fix it up at the end.
        if ((*this)[num_bits - 1]) {
            neg ^= true;
            negate();
        }

        n_a = num_bits_in_use();
    }

    // *this has the correct number of bits, but may need one more subtraction.
    while (*this >= modulus) {
        *this -= modulus;
    }

    // Cancel out remaining negations accumulated during the computation.
    if (neg && *this) {
        // Modular negation.
        *this ^= -1;
        iadd_carry(modulus, true);
    }

    return *this;
}

std::string FixedWidthInt::decimal() const {
    std::string result;
    FixedWidthInt acc = *this;
    while (acc) {
        uint64_t rem = 0;
        uint64_t mult = 1;
        for (size_t k = 0; k < num_words; k++) {
            rem += mult * (acc.words[k] % 10);
            mult *= 6;
            mult %= 10;
        }

        rem %= 10;
        result.push_back('0' + static_cast<char>(rem));
        acc -= rem;
        acc >>= 1;
        acc.inplace_div5();
    }
    if (result.empty()) {
        result.push_back('0');
    }
    for (size_t k = 0; k < result.size() - k - 1; k++) {
        std::swap(result[k], result[result.size() - k - 1]);
    }
    return result;
}

FixedWidthInt kickmix::operator*(const FixedWidthInt &lhs, size_t rhs) {
    return lhs * FixedWidthInt(sizeof(rhs) * 8, static_cast<uint64_t>(rhs));
}
FixedWidthInt kickmix::operator*(size_t lhs, const FixedWidthInt &rhs) {
    return FixedWidthInt(sizeof(lhs) * 8, static_cast<uint64_t>(lhs)) * rhs;
}
FixedWidthInt kickmix::operator-(const FixedWidthInt &lhs, size_t rhs) {
    return lhs - FixedWidthInt(sizeof(rhs) * 8, static_cast<uint64_t>(rhs));
}
FixedWidthInt kickmix::operator-(size_t lhs, const FixedWidthInt &rhs) {
    return FixedWidthInt(sizeof(lhs) * 8, static_cast<uint64_t>(lhs)) - rhs;
}
FixedWidthInt kickmix::operator+(const FixedWidthInt &lhs, size_t rhs) {
    return lhs + FixedWidthInt(sizeof(rhs) * 8, static_cast<uint64_t>(rhs));
}
FixedWidthInt kickmix::operator+(size_t lhs, const FixedWidthInt &rhs) {
    return FixedWidthInt(sizeof(lhs) * 8, static_cast<uint64_t>(lhs)) + rhs;
}
