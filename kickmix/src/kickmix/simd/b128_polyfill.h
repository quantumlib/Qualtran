#ifndef KICKMIX_SIMD_128_H
#define KICKMIX_SIMD_128_H

#include <ostream>
#include <random>

namespace kickmix {

struct alignas(sizeof(uint64_t) * 2) b128_polyfill {
    uint64_t v[2];

    constexpr static size_t BIT_SIZE = 128;
    constexpr static const char *NAME = "b128_polyfill";
    bool non_zero() const {
        return v[0] | v[1];
    }
    bool bit(size_t k) const {
        return (v[k / 64] >> (k % 64)) & 1;
    }
    inline void set_bit(size_t k, bool new_value) {
        uint64_t mask = static_cast<uint64_t>(1) << (k % 64);
        if (new_value) {
            v[k / 64] |= mask;
        } else {
            v[k / 64] &= ~mask;
        }
    }
    inline uint32_t u32(size_t k) const {
        return static_cast<uint32_t>(v[k / 2] >> ((k % 2) * 32));
    }
    inline uint64_t u64(size_t k) const {
        return v[k];
    }
    static b128_polyfill random(std::mt19937_64 &rng) {
        return {rng(), rng()};
    }
    static b128_polyfill from_u32_broadcast(uint32_t v) {
        uint64_t v2 = static_cast<uint64_t>(v);
        uint64_t v3 = v2 | (v2 << 32);
        return {v3, v3};
    }

    void randomize(std::mt19937_64 &rng) {
        v[0] = rng();
        v[1] = rng();
    }
    inline void clear_to_zero() {
        v[0] = 0;
        v[1] = 0;
    }
    inline void clear_to_max() {
        v[0] = UINT64_MAX;
        v[1] = UINT64_MAX;
    }
    inline bool operator==(const b128_polyfill &other) const {
        return v[0] == other.v[0] && v[1] == other.v[1];
    }
    inline b128_polyfill &u64_iadd(const b128_polyfill &other) {
        v[0] += other.v[0];
        v[1] += other.v[1];
        return *this;
    }
    inline b128_polyfill u64_add(const b128_polyfill &other) const {
        return {
            v[0] + other.v[0],
            v[1] + other.v[1],
        };
    }
    inline b128_polyfill &operator^=(const b128_polyfill &other) {
        v[0] ^= other.v[0];
        v[1] ^= other.v[1];
        return *this;
    }
    inline b128_polyfill &operator&=(const b128_polyfill &other) {
        v[0] &= other.v[0];
        v[1] &= other.v[1];
        return *this;
    }
    inline b128_polyfill &operator|=(const b128_polyfill &other) {
        v[0] |= other.v[0];
        v[1] |= other.v[1];
        return *this;
    }
    inline b128_polyfill operator~() const {
        return {~v[0], ~v[1]};
    }
    inline b128_polyfill operator^(const b128_polyfill &other) const {
        return {v[0] ^ other.v[0], v[1] ^ other.v[1]};
    }
    inline b128_polyfill u64_right_shift(size_t shift) const {
        return {v[0] >> shift, v[1] >> shift};
    }
    inline b128_polyfill u64_left_shift(size_t shift) const {
        return {v[0] << shift, v[1] << shift};
    }
    inline b128_polyfill operator&(const b128_polyfill &other) const {
        return {v[0] & other.v[0], v[1] & other.v[1]};
    }
    inline b128_polyfill operator|(const b128_polyfill &other) const {
        return {v[0] | other.v[0], v[1] | other.v[1]};
    }
    inline b128_polyfill u32_eq(const b128_polyfill &other) const {
        const uint32_t *v1 = (uint32_t *)&v;
        const uint32_t *v2 = (uint32_t *)&other.v;
        b128_polyfill result;
        uint32_t *v3 = (uint32_t *)&result.v;
        for (size_t k = 0; k < BIT_SIZE / 32; k++) {
            v3[k] = -(uint64_t)(v1[k] == v2[k]);
        }
        return result;
    }
};

inline std::ostream &operator<<(std::ostream &out, const b128_polyfill &v) {
    for (size_t k = 0; k < sizeof(v) * 8; k++) {
        out << "_1"[v.bit(k)];
    }
    return out;
}

}  // namespace kickmix

#endif
