#ifndef _KICKGEN_UTIL_MOD_INT_H
#define _KICKGEN_UTIL_MOD_INT_H

#include <cstdint>
#include <memory>
#include <random>

#include "kickmix/util/fixed_width_int.h"

namespace kickmix {

/// Represents a big int value modulo a big int modulus.
struct ModInt {
    kickmix::FixedWidthInt value;
    std::shared_ptr<const kickmix::FixedWidthInt> modulus;

    ModInt() = delete;
    explicit ModInt(std::shared_ptr<const kickmix::FixedWidthInt> modulus);
    ModInt(const kickmix::FixedWidthInt &init_value, std::shared_ptr<const kickmix::FixedWidthInt> modulus);
    ModInt(std::string_view init_value, std::shared_ptr<const kickmix::FixedWidthInt> modulus);
    ModInt(kickmix::FixedWidthInt &&init_value, std::shared_ptr<const kickmix::FixedWidthInt> modulus);
    ModInt(int64_t init_value, std::shared_ptr<const kickmix::FixedWidthInt> modulus);

    static ModInt random(std::mt19937_64 &rng, std::shared_ptr<const kickmix::FixedWidthInt> modulus);

    void randomize(std::mt19937_64 &rng);
    void verify_invariants() const;
    explicit operator bool() const;
    bool non_zero() const;

    ModInt(const ModInt &new_value);
    ModInt(ModInt &&new_value) noexcept;
    ModInt &operator=(const ModInt &new_value);
    ModInt &operator=(ModInt &&new_value) noexcept;

    bool operator==(const ModInt &other) const;
    void increment();
    ModInt &operator*=(const ModInt &other);
    ModInt &operator*=(int64_t other);
    ModInt &operator/=(ModInt &&other);
    ModInt &operator/=(const ModInt &other);
    ModInt &operator/=(int64_t value);
    ModInt &operator+=(int64_t value);
    ModInt &operator+=(const ModInt &other);
    ModInt &operator-=(const ModInt &other);
    void negate();
    void invert();
    void idouble();
    void ihalve();
    ModInt squared() const;
    ModInt &operator<<=(uint64_t other);
    ModInt &operator<<=(int64_t other);
    ModInt &operator>>=(uint64_t other);
    ModInt &operator>>=(int64_t other);
    ModInt &operator<<=(uint32_t other);
    ModInt &operator<<=(int32_t other);
    ModInt &operator>>=(uint32_t other);
    ModInt &operator>>=(int32_t other);
    std::string str() const;
};
std::ostream &operator<<(std::ostream &out, const ModInt &rhs);

}  // namespace kickmix

#endif
