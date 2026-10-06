#ifndef KICKMIX_SIMD_512_H
#define KICKMIX_SIMD_512_H

#include <ostream>
#include <random>

namespace kickmix {

struct alignas(sizeof(uint64_t) * 8) b512_polyfill {
    uint64_t v[8];

    constexpr static size_t BIT_SIZE = 512;
    constexpr static const char *NAME = "b512_polyfill";
    bool non_zero() const {
        return v[0] | v[1] | v[2] | v[3] | v[4] | v[5] | v[6] | v[7];
    }
    inline static b512_polyfill random(std::mt19937_64 &rng) {
        return {rng(), rng(), rng(), rng(), rng(), rng(), rng(), rng()};
    }
    inline bool bit(size_t k) const {
        return (v[k / 64] >> (k % 64)) & 1;
    }
    inline uint64_t u64(size_t k) const {
        return v[k];
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
    inline static b512_polyfill from_u32_broadcast(uint32_t v) {
        uint64_t v2 = static_cast<uint64_t>(v);
        uint64_t v3 = v2 | (v2 << 32);
        return {v3, v3, v3, v3, v3, v3, v3, v3};
    }

    inline b512_polyfill &u64_iadd(const b512_polyfill &other) {
        v[0] += other.v[0];
        v[1] += other.v[1];
        v[2] += other.v[2];
        v[3] += other.v[3];
        v[4] += other.v[4];
        v[5] += other.v[5];
        v[6] += other.v[6];
        v[7] += other.v[7];
        return *this;
    }
    inline b512_polyfill u64_add(const b512_polyfill &other) const {
        return {
            v[0] + other.v[0],
            v[1] + other.v[1],
            v[2] + other.v[2],
            v[3] + other.v[3],
            v[4] + other.v[4],
            v[5] + other.v[5],
            v[6] + other.v[6],
            v[7] + other.v[7],
        };
    }
    inline b512_polyfill u64_right_shift(size_t shift) const {
        return {
            v[0] >> shift,
            v[1] >> shift,
            v[2] >> shift,
            v[3] >> shift,
            v[4] >> shift,
            v[5] >> shift,
            v[6] >> shift,
            v[7] >> shift,
        };
    }
    inline b512_polyfill u64_left_shift(size_t shift) const {
        return {
            v[0] << shift,
            v[1] << shift,
            v[2] << shift,
            v[3] << shift,
            v[4] << shift,
            v[5] << shift,
            v[6] << shift,
            v[7] << shift,
        };
    }
    inline void randomize(std::mt19937_64 &rng) {
        v[0] = rng();
        v[1] = rng();
        v[2] = rng();
        v[3] = rng();
        v[4] = rng();
        v[5] = rng();
        v[6] = rng();
        v[7] = rng();
    }
    inline void clear_to_zero() {
        v[0] = 0;
        v[1] = 0;
        v[2] = 0;
        v[3] = 0;
        v[4] = 0;
        v[5] = 0;
        v[6] = 0;
        v[7] = 0;
    }
    inline void clear_to_max() {
        v[0] = UINT64_MAX;
        v[1] = UINT64_MAX;
        v[2] = UINT64_MAX;
        v[3] = UINT64_MAX;
        v[4] = UINT64_MAX;
        v[5] = UINT64_MAX;
        v[6] = UINT64_MAX;
        v[7] = UINT64_MAX;
    }
    inline bool operator==(const b512_polyfill &other) const {
        return v[0] == other.v[0] && v[1] == other.v[1] && v[2] == other.v[2] && v[3] == other.v[3] &&
               v[4] == other.v[4] && v[5] == other.v[5] && v[6] == other.v[6] && v[7] == other.v[7];
    }
    inline b512_polyfill &operator^=(const b512_polyfill &other) {
        v[0] ^= other.v[0];
        v[1] ^= other.v[1];
        v[2] ^= other.v[2];
        v[3] ^= other.v[3];
        v[4] ^= other.v[4];
        v[5] ^= other.v[5];
        v[6] ^= other.v[6];
        v[7] ^= other.v[7];
        return *this;
    }
    inline b512_polyfill &operator&=(const b512_polyfill &other) {
        v[0] &= other.v[0];
        v[1] &= other.v[1];
        v[2] &= other.v[2];
        v[3] &= other.v[3];
        v[4] &= other.v[4];
        v[5] &= other.v[5];
        v[6] &= other.v[6];
        v[7] &= other.v[7];
        return *this;
    }
    inline b512_polyfill &operator|=(const b512_polyfill &other) {
        v[0] |= other.v[0];
        v[1] |= other.v[1];
        v[2] |= other.v[2];
        v[3] |= other.v[3];
        v[4] |= other.v[4];
        v[5] |= other.v[5];
        v[6] |= other.v[6];
        v[7] |= other.v[7];
        return *this;
    }
    inline b512_polyfill operator~() const {
        return {~v[0], ~v[1], ~v[2], ~v[3], ~v[4], ~v[5], ~v[6], ~v[7]};
    }
    inline b512_polyfill operator^(const b512_polyfill &other) const {
        return {
            v[0] ^ other.v[0],
            v[1] ^ other.v[1],
            v[2] ^ other.v[2],
            v[3] ^ other.v[3],
            v[4] ^ other.v[4],
            v[5] ^ other.v[5],
            v[6] ^ other.v[6],
            v[7] ^ other.v[7],
        };
    }
    inline b512_polyfill operator&(const b512_polyfill &other) const {
        return {
            v[0] & other.v[0],
            v[1] & other.v[1],
            v[2] & other.v[2],
            v[3] & other.v[3],
            v[4] & other.v[4],
            v[5] & other.v[5],
            v[6] & other.v[6],
            v[7] & other.v[7],
        };
    }
    inline b512_polyfill operator|(const b512_polyfill &other) const {
        return {
            v[0] | other.v[0],
            v[1] | other.v[1],
            v[2] | other.v[2],
            v[3] | other.v[3],
            v[4] | other.v[4],
            v[5] | other.v[5],
            v[6] | other.v[6],
            v[7] | other.v[7],
        };
    }
    inline b512_polyfill u32_eq(const b512_polyfill &other) const {
        const uint32_t *v1 = (uint32_t *)&v;
        const uint32_t *v2 = (uint32_t *)&other.v;
        b512_polyfill result;
        uint32_t *v3 = (uint32_t *)&result.v;
        for (size_t k = 0; k < BIT_SIZE / 32; k++) {
            v3[k] = -(uint64_t)(v1[k] == v2[k]);
        }
        return result;
    }
};

inline std::ostream &operator<<(std::ostream &out, const b512_polyfill &v) {
    for (size_t k = 0; k < sizeof(v) * 8; k++) {
        out << "_1"[v.bit(k)];
    }
    return out;
}

}  // namespace kickmix

#endif
