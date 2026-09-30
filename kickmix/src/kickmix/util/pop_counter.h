#ifndef KICKMIX_UTIL_POP_COUNTER_H
#define KICKMIX_UTIL_POP_COUNTER_H

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace kickmix {

/// Tracks per-bit-position population counts of values given to the
/// `masked_increments` method.
template <typename TWord>
struct PopCounter {
    /// Bit-striped counters, with bit_totals[k] storing the 64 bits with weight 1<<k.
    constexpr static size_t BATCH_SIZE = sizeof(TWord) * 8;

    TWord bit_totals[6]{};
    /// Merged counter pairs, with each uint64_t storing two uint32_t's.
    TWord paired_totals[32]{};
    /// This counter is added to all 64 totals.
    uint32_t shared_count = 0;
    /// Tracks how many increment calls have happened, mod 32.
    uint8_t step = 0;

    void clear() {
        memset(this, 0, sizeof(PopCounter));
    }

    inline uint64_t compute_total(size_t k) const {
        uint64_t result = static_cast<uint64_t>(paired_totals[k % 32].u32(k / 32)) << 5;
        for (size_t r = 0; r < 6; r++) {
            result += static_cast<uint64_t>(bit_totals[r].bit(k)) << r;
        }
        return result + shared_count;
    }

    inline void masked_increments(const TWord &mask_ref) {
        TWord mask = mask_ref;
        // Transfer two bits from the striped 32-bits counter into a paired counter.
        step++;
        step &= 31;
        paired_totals[step].u64_iadd(bit_totals[5].u64_right_shift(step) & TWord::from_u32_broadcast(1));
        bit_totals[5] &= ~(TWord::from_u32_broadcast(1).u64_left_shift(step));

        // Conditionally increment the per-bit-position counters.
        bit_totals[0] ^= mask;
        mask &= ~bit_totals[0];
        bit_totals[1] ^= mask;
        mask &= ~bit_totals[1];
        bit_totals[2] ^= mask;
        mask &= ~bit_totals[2];
        bit_totals[3] ^= mask;
        mask &= ~bit_totals[3];
        bit_totals[4] ^= mask;
        mask &= ~bit_totals[4];
        bit_totals[5] ^= mask;
    }
};

}  // namespace kickmix

#endif
