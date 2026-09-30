#ifndef KICKMIX_SIMD_64_H
#define KICKMIX_SIMD_64_H

#include <ostream>
#include <random>

namespace kickmix {

struct alignas(sizeof(uint64_t) * 1) b64_polyfill {
    uint64_t v[1];

    constexpr static size_t BIT_SIZE = 64;
    constexpr static const char *NAME = "b64_polyfill";
    inline bool non_zero() const {
        return v[0];
    }
    inline static b64_polyfill from_u32_broadcast(uint32_t v) {
        uint64_t v2 = static_cast<uint64_t>(v);
        uint64_t v3 = v2 | (v2 << 32);
        return {v3};
    }
    inline bool bit(size_t k) const {
        return (v[0] >> k) & 1;
    }
    inline void set_bit(size_t k, bool new_value) {
        uint64_t mask = static_cast<uint64_t>(1) << k;
        if (new_value) {
            v[0] |= mask;
        } else {
            v[0] &= ~mask;
        }
    }
    inline uint32_t u32(size_t k) const {
        return static_cast<uint32_t>(v[0] >> (k * 32));
    }
    inline uint64_t u64(size_t k) const {
        return v[k];
    }
    inline b64_polyfill &u64_iadd(const b64_polyfill &other) {
        v[0] += other.v[0];
        return *this;
    }
    inline b64_polyfill u64_add(const b64_polyfill &other) const {
        return {v[0] + other.v[0]};
    }
    inline b64_polyfill u64_right_shift(size_t shift) const {
        return {v[0] >> shift};
    }
    inline b64_polyfill u64_left_shift(size_t shift) const {
        return {v[0] << shift};
    }
    inline static b64_polyfill random(std::mt19937_64 &rng) {
        return {rng()};
    }
    inline void randomize(std::mt19937_64 &rng) {
        v[0] = rng();
    }
    inline void clear_to_zero() {
        v[0] = 0;
    }
    inline void clear_to_max() {
        v[0] = UINT64_MAX;
    }
    inline bool operator==(const b64_polyfill &other) const {
        return v[0] == other.v[0];
    }
    inline b64_polyfill &operator^=(const b64_polyfill &other) {
        v[0] ^= other.v[0];
        return *this;
    }
    inline b64_polyfill &operator&=(const b64_polyfill &other) {
        v[0] &= other.v[0];
        return *this;
    }
    inline b64_polyfill &operator|=(const b64_polyfill &other) {
        v[0] |= other.v[0];
        return *this;
    }
    inline b64_polyfill operator~() const {
        return {~v[0]};
    }
    inline b64_polyfill operator^(const b64_polyfill &other) const {
        return {v[0] ^ other.v[0]};
    }
    inline b64_polyfill operator&(const b64_polyfill &other) const {
        return {v[0] & other.v[0]};
    }
    inline b64_polyfill operator|(const b64_polyfill &other) const {
        return {v[0] | other.v[0]};
    }
    inline std::string str() const {
        std::string out;
        for (size_t k = 0; k < sizeof(v) * 8; k++) {
            out.push_back("_1"[bit(k)]);
        }
        return out;
    }
    inline b64_polyfill u32_eq(const b64_polyfill &other) const {
        const uint32_t *v1 = (uint32_t *)&v;
        const uint32_t *v2 = (uint32_t *)&other.v;
        b64_polyfill result;
        uint32_t *v3 = (uint32_t *)&result.v;
        for (size_t k = 0; k < BIT_SIZE / 32; k++) {
            v3[k] = -(uint64_t)(v1[k] == v2[k]);
        }
        return result;
    }
};

inline std::ostream &operator<<(std::ostream &out, const b64_polyfill &v) {
    for (size_t k = 0; k < sizeof(v) * 8; k++) {
        out << "_1"[v.bit(k)];
    }
    return out;
}

}  // namespace kickmix

#endif
