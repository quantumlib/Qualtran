#include "mod_int.h"

#include <cstring>
#include <iostream>
#include <sstream>

using namespace kickmix;

ModInt::ModInt(std::shared_ptr<const FixedWidthInt> modulus) : value(modulus->num_bits), modulus(modulus) {
}

ModInt::ModInt(int64_t init_value, std::shared_ptr<const FixedWidthInt> modulus)
    : value(FixedWidthInt(modulus->num_bits, init_value)), modulus(modulus) {
    if (init_value < 0) {
        value.negate();
        negate();
    }
}

ModInt::ModInt(std::string_view init_value, std::shared_ptr<const FixedWidthInt> modulus)
    : value(FixedWidthInt(modulus->num_bits)), modulus(modulus) {
    auto tmp = FixedWidthInt(init_value);
    tmp %= *modulus;
    value ^= tmp;
    if (init_value.starts_with("-")) {
        value.negate();
        negate();
    }
}

void ModInt::verify_invariants() const {
    // Caution: invariants are not satisfied after a `move` out of a field.

    if (modulus->num_bits != modulus->num_bits_in_use()) {
        std::stringstream ss;
        ss << "Invalid ModInt: " << *this << "\n";
        ss << "    modulus->num_bits=" << modulus->num_bits << "\n";
        ss << "    !=\n";
        ss << "    modulus->num_bits_in_use()" << modulus->num_bits_in_use() << "\n";
        throw std::invalid_argument(ss.str());
    }
    if (value >= *modulus) {
        std::stringstream ss;
        ss << "Invalid ModInt: " << *this << "\n";
        ss << "    value=" << value.decimal() << "\n";
        ss << "    >=\n";
        ss << "    modulus=" << modulus->decimal() << "\n";
        throw std::invalid_argument(ss.str());
    }
    if (value.num_bits != modulus->num_bits) {
        std::stringstream ss;
        ss << "Invalid ModInt: " << *this << "\n";
        ss << "    value.num_bits=" << value.num_bits << "\n";
        ss << "    !=\n";
        ss << "    modulus->num_bits=" << modulus->num_bits << "\n";
        throw std::invalid_argument(ss.str());
    }
    if (value.num_words != modulus->num_words) {
        std::stringstream ss;
        ss << "Invalid ModInt: " << *this << "\n";
        ss << "    value.num_words=" << value.num_words << "\n";
        ss << "    !=\n";
        ss << "    modulus->num_words=" << modulus->num_words << "\n";
        throw std::invalid_argument(ss.str());
    }
}

ModInt::ModInt(const FixedWidthInt &init_value, std::shared_ptr<const FixedWidthInt> modulus)
    : value(), modulus(modulus) {
    if (init_value.num_bits < modulus->num_bits || init_value < *modulus) {
        value = FixedWidthInt(modulus->num_bits);
        value ^= init_value;
    } else {
        FixedWidthInt tmp = init_value;
        tmp %= *modulus;
        if (tmp.num_words == modulus->num_words) {
            value = std::move(tmp);
            value.num_bits = modulus->num_bits;
        } else {
            value = FixedWidthInt(modulus->num_bits);
            value ^= tmp;
        }
    }
}

ModInt::ModInt(FixedWidthInt &&init_value, std::shared_ptr<const FixedWidthInt> modulus)
    : value(init_value), modulus(modulus) {
    if (value >= *modulus) {
        value %= *modulus;
    }
    if (value.num_words != modulus->num_words) {
        FixedWidthInt tmp(modulus->num_bits);
        tmp ^= value;
        value = std::move(tmp);
    }
    value.num_bits = modulus->num_bits;
}

ModInt ModInt::random(std::mt19937_64 &rng, std::shared_ptr<const FixedWidthInt> modulus) {
    ModInt result(modulus);
    result.value.randomize_mod(rng, *modulus);
    return result;
}

void ModInt::randomize(std::mt19937_64 &rng) {
    value.randomize_mod(rng, *modulus);
}
ModInt::operator bool() const {
    return value.non_zero();
}

bool ModInt::non_zero() const {
    return value.non_zero();
}

ModInt::ModInt(const ModInt &new_value) {
    value = new_value.value;
    modulus = new_value.modulus;
}
ModInt::ModInt(ModInt &&new_value) noexcept {
    // Caution: this makes `new_value` no longer satisfy invariants.
    value = std::move(new_value.value);
    modulus = new_value.modulus;
}
ModInt &ModInt::operator=(const ModInt &new_value) {
    value = new_value.value;
    modulus = new_value.modulus;
    return *this;
}
ModInt &ModInt::operator=(ModInt &&new_value) noexcept {
    // Caution: this makes `new_value` no longer satisfy invariants.
    value = std::move(new_value.value);
    modulus = new_value.modulus;
    return *this;
}

bool ModInt::operator==(const ModInt &other) const {
    return value == other.value;
}
ModInt &ModInt::operator*=(const ModInt &other) {
    FixedWidthInt tmp(modulus->num_bits * 2);
    tmp.iadd_product(value, other.value);
    tmp %= *modulus;
    memcpy(value.words, tmp.words, sizeof(uint64_t) * value.num_words);
    return *this;
}
ModInt ModInt::squared() const {
    FixedWidthInt tmp(modulus->num_bits * 2);
    tmp.iadd_product(value, value);
    tmp %= *modulus;
    tmp.num_words = modulus->num_words;
    tmp.num_bits = modulus->num_bits;
    return ModInt(std::move(tmp), modulus);
}
void ModInt::idouble() {
    value.idouble_mod(*modulus);
}

void ModInt::ihalve() {
    value.ihalve_mod(*modulus);
}
ModInt &ModInt::operator*=(int64_t other) {
    if (other < 0) {
        negate();
        other *= -1;
    }
    FixedWidthInt tmp(modulus->num_bits + 64);
    tmp.iadd_product(value, static_cast<uint64_t>(other));
    tmp %= *modulus;
    memcpy(value.words, tmp.words, sizeof(uint64_t) * value.num_words);
    return *this;
}
ModInt &ModInt::operator/=(const ModInt &other) {
    ModInt tmp = other;
    return *this /= std::move(tmp);
}
ModInt &ModInt::operator/=(int64_t other) {
    ModInt tmp = ModInt(other, modulus);
    return *this /= std::move(tmp);
}
ModInt &ModInt::operator/=(ModInt &&other) {
    if (!(*modulus)[0]) {
        throw std::invalid_argument("inverse not implemented: even modulus.");
    }

    FixedWidthInt u = std::move(other.value);
    FixedWidthInt v = *modulus;
    ModInt &u2 = *this;
    ModInt v2(modulus);
    if (!u[0]) {
        std::swap(u, v);
        std::swap(u2, v2);
    }
    while (v.non_zero()) {
        if (v[0]) {
            if (v < u) {
                std::swap(u, v);
                std::swap(u2, v2);
            }
            v -= u;
            v2 -= u2;
        }
        v >>= 1;
        v2.ihalve();
    }
    if (u != 1) {
        throw std::invalid_argument("Value incorrectly inverted because gcd(value, modulus) != 1.");
    }
    return *this;
}
ModInt &ModInt::operator+=(const ModInt &other) {
    if (value.iadd_carry(other.value) || value >= *modulus) {
        value -= *modulus;
    }
    return *this;
}
ModInt &ModInt::operator+=(int64_t other) {
    if (value.iadd_carry(other) || value >= *modulus) {
        value -= *modulus;
    }
    return *this;
}
ModInt &ModInt::operator-=(const ModInt &other) {
    if (value.isub_borrow(other.value)) {
        value += *modulus;
    }
    return *this;
}
void ModInt::negate() {
    if (value.non_zero()) {
        value ^= -1;
        value.iadd_carry(*modulus, true);
    }
}
void ModInt::increment() {
    value.increment();
    if (value == *modulus) {
        value.clear_to_zero();
    }
}
void ModInt::invert() {
    ModInt divisor(modulus);
    divisor.value ^= 1;
    std::swap(*this, divisor);
    *this /= std::move(divisor);
}
ModInt &ModInt::operator<<=(uint64_t other) {
    while (other > 0) {
        value.idouble_mod(*modulus);
        other--;
    }
    return *this;
}
ModInt &ModInt::operator<<=(int64_t other) {
    while (other > 0) {
        value.idouble_mod(*modulus);
        other--;
    }
    while (other < 0) {
        value.ihalve_mod(*modulus);
        other++;
    }
    return *this;
}
ModInt &ModInt::operator>>=(uint64_t other) {
    while (other > 0) {
        value.ihalve_mod(*modulus);
        other--;
    }
    return *this;
}
ModInt &ModInt::operator>>=(int64_t other) {
    while (other > 0) {
        value.ihalve_mod(*modulus);
        other--;
    }
    while (other < 0) {
        value.idouble_mod(*modulus);
        other++;
    }
    return *this;
}
ModInt &ModInt::operator>>=(uint32_t other) {
    return *this >>= uint64_t{other};
}
ModInt &ModInt::operator>>=(int32_t other) {
    return *this >>= int64_t{other};
}
ModInt &ModInt::operator<<=(uint32_t other) {
    return *this <<= uint64_t{other};
}
ModInt &ModInt::operator<<=(int32_t other) {
    return *this <<= int64_t{other};
}

std::ostream &kickmix::operator<<(std::ostream &out, const ModInt &rhs) {
    out << rhs.value.decimal() << " (mod " << rhs.modulus->decimal() << ")";
    return out;
}

std::string ModInt::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}
