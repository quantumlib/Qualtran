#include "kickmix/util/gf2_poly.h"

#include <bit>
#include <ostream>
#include <sstream>
#include <stdexcept>

using namespace kickmix;

GF2Poly GF2Poly::from_u64(uint64_t value) {
    GF2Poly result;
    result.words[0] = value;
    return result;
}

GF2Poly GF2Poly::monomial(size_t k) {
    if (k >= MAX_BITS) {
        throw std::invalid_argument("GF2Poly::monomial: k >= GF2Poly::MAX_BITS");
    }
    GF2Poly result;
    result.words[k / 64] = uint64_t{1} << (k & 63);
    return result;
}

GF2Poly GF2Poly::from_fixed_width_int(const FixedWidthInt &value) {
    if (value.num_bits > MAX_BITS) {
        throw std::invalid_argument("GF2Poly::from_fixed_width_int: value.num_bits > GF2Poly::MAX_BITS");
    }
    GF2Poly result;
    size_t n = std::min(value.num_words, MAX_WORDS);
    for (size_t k = 0; k < n; k++) {
        result.words[k] = value.words[k];
    }
    result.truncate(value.num_bits);
    return result;
}

namespace {

std::string_view trim_ws(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\n' || s.front() == '\r')) {
        s.remove_prefix(1);
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\n' || s.back() == '\r')) {
        s.remove_suffix(1);
    }
    return s;
}

}  // namespace

GF2Poly GF2Poly::from_str(std::string_view raw_text) {
    std::string_view text = trim_ws(raw_text);
    if (text.empty()) {
        throw std::invalid_argument(
            "Cannot parse empty string as a GF(2) polynomial. Expected e.g. \"x^4 + x + 1\" or \"0x13\".");
    }

    if ((text.starts_with("0x") || text.starts_with("0X") || text.starts_with("0b") || text.starts_with("0B")) &&
        text.find('+') == std::string_view::npos) {
        std::string normalized(text);
        if (normalized[1] == 'X') {
            normalized[1] = 'x';
        } else if (normalized[1] == 'B') {
            normalized[1] = 'b';
        }
        return GF2Poly::from_fixed_width_int(FixedWidthInt::from_str(normalized, MAX_BITS));
    }

    GF2Poly result;
    std::string_view remaining = text;
    auto add_term = [&](size_t exp, std::string_view token) {
        if (result.bit(exp)) {
            throw std::invalid_argument(
                "Polynomial term \"" + std::string(token) + "\" appears more than once in \"" + std::string(raw_text) +
                "\". Over GF(2) repeated terms cancel, so write each power of x at most once (e.g. "
                "\"x^4 + x + 1\").");
        }
        result.xor_bit(exp);
    };
    while (true) {
        size_t plus_pos = remaining.find('+');
        std::string_view token =
            trim_ws(plus_pos == std::string_view::npos ? remaining : remaining.substr(0, plus_pos));
        if (token.empty()) {
            throw std::invalid_argument(
                "Invalid empty polynomial term in \"" + std::string(raw_text) +
                "\". Expected terms separated by '+', e.g. \"x^4 + x + 1\".");
        }

        if (token == "0") {
            // Zero term contributes nothing.
        } else if (token == "1") {
            add_term(0, token);
        } else if (token.front() == 'x' || token.front() == 'X') {
            std::string_view rest = trim_ws(token.substr(1));
            if (rest.empty()) {
                add_term(1, token);
            } else if (rest.starts_with("^") || rest.starts_with("**")) {
                std::string_view exp_str = trim_ws(rest.substr(rest.starts_with("**") ? 2 : 1));
                if (exp_str.empty()) {
                    throw std::invalid_argument(
                        "Missing exponent in polynomial term \"" + std::string(token) + "\" in \"" +
                        std::string(raw_text) + "\". Expected e.g. \"x^4 + x + 1\".");
                }
                size_t exp = 0;
                for (char c : exp_str) {
                    if (c < '0' || c > '9') {
                        throw std::invalid_argument(
                            "Invalid exponent in polynomial term \"" + std::string(token) + "\" in \"" +
                            std::string(raw_text) + "\". Expected a non-negative integer, e.g. \"x^4 + x + 1\".");
                    }
                    exp = exp * 10 + static_cast<size_t>(c - '0');
                    if (exp >= MAX_BITS) {
                        throw std::invalid_argument(
                            "Polynomial term \"" + std::string(token) +
                            "\" has exponent >= " + std::to_string(MAX_BITS) + " (maximum supported field degree is " +
                            std::to_string(GF2_MAX_DEGREE) + ").");
                    }
                }
                add_term(exp, token);
            } else {
                throw std::invalid_argument(
                    "Invalid polynomial term \"" + std::string(token) + "\" in \"" + std::string(raw_text) +
                    "\". Expected terms like \"x^4\", \"x\", or \"1\" (e.g. \"x^4 + x + 1\").");
            }
        } else {
            throw std::invalid_argument(
                "Invalid polynomial term \"" + std::string(token) + "\" in \"" + std::string(raw_text) +
                "\". Expected terms like \"x^4\", \"x\", or \"1\" (e.g. \"x^4 + x + 1\"), or a hex/binary string "
                "like \"0x13\".");
        }

        if (plus_pos == std::string_view::npos) {
            break;
        }
        remaining = remaining.substr(plus_pos + 1);
    }

    return result;
}

void GF2Poly::set_bit(size_t k, bool value) {
    if (k >= MAX_BITS) {
        throw std::invalid_argument("GF2Poly::set_bit: k >= GF2Poly::MAX_BITS");
    }
    uint64_t mask = uint64_t{1} << (k & 63);
    if (value) {
        words[k / 64] |= mask;
    } else {
        words[k / 64] &= ~mask;
    }
}

void GF2Poly::xor_bit(size_t k) {
    if (k >= MAX_BITS) {
        throw std::invalid_argument("GF2Poly::xor_bit: k >= GF2Poly::MAX_BITS");
    }
    words[k / 64] ^= uint64_t{1} << (k & 63);
}

bool GF2Poly::is_zero() const {
    uint64_t acc = 0;
    for (size_t k = 0; k < MAX_WORDS; k++) {
        acc |= words[k];
    }
    return acc == 0;
}

size_t GF2Poly::degree() const {
    for (size_t k = MAX_WORDS; k--;) {
        if (words[k] != 0) {
            return k * 64 + (63 - std::countl_zero(words[k]));
        }
    }
    return SIZE_MAX;
}

size_t GF2Poly::num_bits_in_use() const {
    size_t d = degree();
    return d == SIZE_MAX ? 0 : d + 1;
}

size_t GF2Poly::weight() const {
    size_t result = 0;
    for (size_t k = 0; k < MAX_WORDS; k++) {
        result += std::popcount(words[k]);
    }
    return result;
}

void GF2Poly::truncate(size_t num_bits) {
    if (num_bits >= MAX_BITS) {
        return;
    }
    size_t word = num_bits / 64;
    size_t offset = num_bits & 63;
    if (offset != 0) {
        words[word] &= (uint64_t{1} << offset) - 1;
        word += 1;
    }
    for (size_t k = word; k < MAX_WORDS; k++) {
        words[k] = 0;
    }
}

GF2Poly &GF2Poly::operator^=(const GF2Poly &other) {
    for (size_t k = 0; k < MAX_WORDS; k++) {
        words[k] ^= other.words[k];
    }
    return *this;
}

GF2Poly &GF2Poly::operator<<=(size_t shift) {
    if (shift >= MAX_BITS) {
        words = {};
        return *this;
    }
    size_t word_shift = shift / 64;
    size_t bit_shift = shift & 63;
    if (bit_shift == 0) {
        for (size_t k = MAX_WORDS; k-- > word_shift;) {
            words[k] = words[k - word_shift];
        }
    } else {
        for (size_t k = MAX_WORDS; k-- > word_shift;) {
            uint64_t hi = words[k - word_shift] << bit_shift;
            uint64_t lo = k - word_shift == 0 ? 0 : words[k - word_shift - 1] >> (64 - bit_shift);
            words[k] = hi | lo;
        }
    }
    for (size_t k = 0; k < word_shift; k++) {
        words[k] = 0;
    }
    return *this;
}

GF2Poly &GF2Poly::operator>>=(size_t shift) {
    if (shift >= MAX_BITS) {
        words = {};
        return *this;
    }
    size_t word_shift = shift / 64;
    size_t bit_shift = shift & 63;
    if (bit_shift == 0) {
        for (size_t k = 0; k < MAX_WORDS - word_shift; k++) {
            words[k] = words[k + word_shift];
        }
    } else {
        for (size_t k = 0; k < MAX_WORDS - word_shift; k++) {
            uint64_t lo = words[k + word_shift] >> bit_shift;
            uint64_t hi = k + word_shift + 1 >= MAX_WORDS ? 0 : words[k + word_shift + 1] << (64 - bit_shift);
            words[k] = hi | lo;
        }
    }
    for (size_t k = MAX_WORDS - word_shift; k < MAX_WORDS; k++) {
        words[k] = 0;
    }
    return *this;
}

void GF2Poly::ixor_shifted(const GF2Poly &other, size_t shift) {
    if (shift >= MAX_BITS) {
        return;
    }
    size_t word_shift = shift / 64;
    size_t bit_shift = shift & 63;
    if (bit_shift == 0) {
        for (size_t k = MAX_WORDS; k-- > word_shift;) {
            words[k] ^= other.words[k - word_shift];
        }
    } else {
        for (size_t k = MAX_WORDS; k-- > word_shift;) {
            uint64_t hi = other.words[k - word_shift] << bit_shift;
            uint64_t lo = k - word_shift == 0 ? 0 : other.words[k - word_shift - 1] >> (64 - bit_shift);
            words[k] ^= hi | lo;
        }
    }
}

GF2Poly GF2Poly::mul(const GF2Poly &lhs, const GF2Poly &rhs) {
    size_t lhs_deg = lhs.degree();
    size_t rhs_deg = rhs.degree();
    if (lhs_deg == SIZE_MAX || rhs_deg == SIZE_MAX) {
        return GF2Poly();
    }
    if (lhs_deg + rhs_deg >= MAX_BITS) {
        throw std::invalid_argument("GF2Poly::mul: degree(lhs) + degree(rhs) >= GF2Poly::MAX_BITS");
    }

    // Multiply by the operand with the smaller degree, so that the comb runs over fewer words.
    const GF2Poly &a = lhs_deg >= rhs_deg ? lhs : rhs;
    const GF2Poly &b = lhs_deg >= rhs_deg ? rhs : lhs;
    size_t b_words = (std::min(lhs_deg, rhs_deg)) / 64 + 1;

    // Left-to-right comb method with a 4 bit window (Lopez-Dahab). Precomputing the 16 small
    // multiples of `a` lets us consume 4 bits of `b` per XOR instead of 1, which is roughly a
    // 4x speedup over the naive shift-and-xor loop.
    constexpr size_t WINDOW = 4;
    constexpr size_t TABLE_SIZE = size_t{1} << WINDOW;
    std::array<GF2Poly, TABLE_SIZE> table{};
    table[1] = a;
    for (size_t u = 2; u < TABLE_SIZE; u += 2) {
        // table[u] = table[u / 2] * x, and table[u + 1] = table[u] + a.
        table[u] = table[u / 2];
        table[u] <<= 1;
        table[u + 1] = table[u];
        table[u + 1] ^= a;
    }

    GF2Poly result;
    for (size_t j = 64 / WINDOW; j--;) {
        for (size_t i = 0; i < b_words; i++) {
            size_t u = (b.words[i] >> (WINDOW * j)) & (TABLE_SIZE - 1);
            if (u != 0) {
                result.ixor_shifted(table[u], i * 64);
            }
        }
        if (j > 0) {
            result <<= WINDOW;
        }
    }
    return result;
}

void GF2Poly::imod(const GF2Poly &modulus) {
    size_t mod_deg = modulus.degree();
    if (mod_deg == SIZE_MAX) {
        throw std::invalid_argument("GF2Poly::imod: modulus is zero");
    }
    size_t deg = degree();
    while (deg != SIZE_MAX && deg >= mod_deg) {
        ixor_shifted(modulus, deg - mod_deg);
        deg = degree();
    }
}

GF2Poly GF2Poly::gcd(GF2Poly lhs, GF2Poly rhs) {
    while (!rhs.is_zero()) {
        lhs.imod(rhs);
        std::swap(lhs, rhs);
    }
    return lhs;
}

GF2Poly GF2Poly::invert_mod(const GF2Poly &value, const GF2Poly &modulus) {
    size_t mod_deg = modulus.degree();
    if (mod_deg == SIZE_MAX) {
        throw std::invalid_argument("GF2Poly::invert_mod: modulus is zero");
    }

    // Extended Euclidean algorithm over GF(2)[x], tracking only the cofactor of `value`.
    GF2Poly r0 = value;
    r0.imod(modulus);
    GF2Poly r1 = modulus;
    GF2Poly t0 = GF2Poly::from_u64(1);
    GF2Poly t1;

    while (!r1.is_zero()) {
        // r0 = r0 - q * r1, t0 = t0 - q * t1, computed via repeated shift-and-subtract so that
        // the quotient never needs to be materialized.
        size_t deg0 = r0.degree();
        size_t deg1 = r1.degree();
        while (deg0 != SIZE_MAX && deg0 >= deg1) {
            size_t shift = deg0 - deg1;
            r0.ixor_shifted(r1, shift);
            t0.ixor_shifted(t1, shift);
            deg0 = r0.degree();
        }
        std::swap(r0, r1);
        std::swap(t0, t1);
    }

    // r0 is now gcd(value, modulus). The inverse exists only when the gcd is 1.
    if (r0 != GF2Poly::from_u64(1)) {
        return GF2Poly();
    }
    t0.imod(modulus);
    return t0;
}

FixedWidthInt GF2Poly::to_fixed_width_int(size_t num_bits) const {
    if (num_bits > MAX_BITS) {
        throw std::invalid_argument("GF2Poly::to_fixed_width_int: num_bits > GF2Poly::MAX_BITS");
    }
    FixedWidthInt result(num_bits);
    size_t n = std::min(result.num_words, MAX_WORDS);
    for (size_t k = 0; k < n; k++) {
        result.words[k] = words[k];
    }
    // Clear bits above num_bits that may have been copied in from the top word.
    if (num_bits < result.num_words * 64) {
        size_t word = num_bits / 64;
        size_t offset = num_bits & 63;
        if (offset != 0) {
            result.words[word] &= (uint64_t{1} << offset) - 1;
            word += 1;
        }
        for (size_t k = word; k < result.num_words; k++) {
            result.words[k] = 0;
        }
    }
    return result;
}

std::string GF2Poly::str() const {
    size_t num_bits = num_bits_in_use();
    if (num_bits == 0) {
        return "0x0";
    }
    std::stringstream ss;
    ss << "0x";
    bool leading = true;
    for (size_t k = (num_bits + 3) / 4; k--;) {
        uint64_t nibble = (words[k / 16] >> ((k & 15) * 4)) & 0xF;
        if (leading && nibble == 0) {
            continue;
        }
        leading = false;
        ss << "0123456789ABCDEF"[nibble];
    }
    return ss.str();
}

std::string GF2Poly::algebraic_str() const {
    if (is_zero()) {
        return "0";
    }
    size_t deg = degree();
    std::string out;
    for (size_t k = deg + 1; k--;) {
        if (!bit(k)) {
            continue;
        }
        if (!out.empty()) {
            out += " + ";
        }
        if (k == 0) {
            out += "1";
        } else if (k == 1) {
            out += "x";
        } else {
            out += "x^" + std::to_string(k);
        }
    }
    return out;
}

std::ostream &kickmix::operator<<(std::ostream &out, const GF2Poly &value) {
    out << value.str();
    return out;
}
