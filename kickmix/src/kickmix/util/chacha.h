#ifndef KICKMIX_UTIL_CHACHA_H
#define KICKMIX_UTIL_CHACHA_H

#include <cstddef>
#include <cstdint>
#include <random>

namespace kickmix {

template <int K, typename TWord>
inline TWord rol32(const TWord &x) {
    return x.u32_left_shift(K) | x.u32_right_shift(32 - K);
}

template <typename TWord>
inline void chacha_quarter_round(TWord &a, TWord &b, TWord &c, TWord &d) {
    a = a.u32_add(b);
    d ^= a;
    d = rol32<16>(d);

    c = c.u32_add(d);
    b ^= c;
    b = rol32<12>(b);

    a = a.u32_add(b);
    d ^= a;
    d = rol32<8>(d);

    c = c.u32_add(d);
    b ^= c;
    b = rol32<7>(b);
}

/// Vertical SIMD ChaCha CSPRNG (RFC 8439).
///
/// Each 32-bit lane across `TWord` runs an independent `ROUNDS`-round ChaCha stream,
/// generating 16 `TWord`s per block evaluation.
template <typename TWord, size_t ROUNDS = 8>
struct ChaCha {
    static_assert(ROUNDS >= 2 && ROUNDS % 2 == 0, "ChaCha ROUNDS must be a positive even integer.");

    TWord state[16]{};
    TWord buffer[16]{};
    size_t buffer_idx = 16;

    void randomize(std::mt19937_64 &rng) {
        // RFC 8439 ChaCha constants: "expand 32-byte k"
        state[0] = TWord::from_u32_broadcast(0x61707865);
        state[1] = TWord::from_u32_broadcast(0x3320646e);
        state[2] = TWord::from_u32_broadcast(0x79622d32);
        state[3] = TWord::from_u32_broadcast(0x6b206574);
        // 256-bit key (words 4..11) per 32-bit lane
        for (size_t k = 4; k < 12; k++) {
            state[k].randomize(rng);
        }
        // 32-bit block counter (word 12) initialized to 0
        state[12].clear_to_zero();
        // 96-bit nonce (words 13..15) per 32-bit lane
        state[13].randomize(rng);
        state[14].randomize(rng);
        state[15].randomize(rng);
        buffer_idx = 16;
    }

#if defined(_MSC_VER)
    __declspec(noinline)
#else
    __attribute__((noinline))
#endif
    void refill() {
        TWord x0 = state[0];
        TWord x1 = state[1];
        TWord x2 = state[2];
        TWord x3 = state[3];
        TWord x4 = state[4];
        TWord x5 = state[5];
        TWord x6 = state[6];
        TWord x7 = state[7];
        TWord x8 = state[8];
        TWord x9 = state[9];
        TWord x10 = state[10];
        TWord x11 = state[11];
        TWord x12 = state[12];
        TWord x13 = state[13];
        TWord x14 = state[14];
        TWord x15 = state[15];

        for (size_t i = 0; i < ROUNDS / 2; i++) {
            // Column round.
            chacha_quarter_round(x0, x4, x8, x12);
            chacha_quarter_round(x1, x5, x9, x13);
            chacha_quarter_round(x2, x6, x10, x14);
            chacha_quarter_round(x3, x7, x11, x15);
            // Diagonal round.
            chacha_quarter_round(x0, x5, x10, x15);
            chacha_quarter_round(x1, x6, x11, x12);
            chacha_quarter_round(x2, x7, x8, x13);
            chacha_quarter_round(x3, x4, x9, x14);
        }

        buffer[0] = x0.u32_add(state[0]);
        buffer[1] = x1.u32_add(state[1]);
        buffer[2] = x2.u32_add(state[2]);
        buffer[3] = x3.u32_add(state[3]);
        buffer[4] = x4.u32_add(state[4]);
        buffer[5] = x5.u32_add(state[5]);
        buffer[6] = x6.u32_add(state[6]);
        buffer[7] = x7.u32_add(state[7]);
        buffer[8] = x8.u32_add(state[8]);
        buffer[9] = x9.u32_add(state[9]);
        buffer[10] = x10.u32_add(state[10]);
        buffer[11] = x11.u32_add(state[11]);
        buffer[12] = x12.u32_add(state[12]);
        buffer[13] = x13.u32_add(state[13]);
        buffer[14] = x14.u32_add(state[14]);
        buffer[15] = x15.u32_add(state[15]);

        state[12] = state[12].u32_add(TWord::from_u32_broadcast(1));
        buffer_idx = 0;
    }

    inline TWord next() {
        if (buffer_idx >= 16) [[unlikely]] {
            refill();
        }
        return buffer[buffer_idx++];
    }
};

template <typename TWord>
using ChaCha8 = ChaCha<TWord, 8>;

template <typename TWord>
using ChaCha20 = ChaCha<TWord, 20>;

}  // namespace kickmix

#endif
