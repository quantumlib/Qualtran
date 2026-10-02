#ifndef KICKMIX_MOD_RAND_H
#define KICKMIX_MOD_RAND_H

#include <cstdint>
#include <random>
#include <span>

#include "kickmix/util/fixed_width_int.h"

namespace kickmix {

/// Generates 64 bit striped random values, uniformly distributed modulo the given modulus.
///
/// Requires:
///    out.size() == modulus.num_bits
///    buf.size() == modulus.num_bits
///    modulus.num_bits > 0
///    modulus[modulus.num_bits - 1] == true
///
/// Args:
///    rng: The random number generate to pull entropy from.
///    out: Where to write the random values.
///        After the method runs, (out[k] >> j) & 1 is equal to bit k of output j.
///    buf: A workspace buffer available for use while generate the random values.
///    modulus: The generated values will be uniformly distributed over the range [0, modulus).
void generate_bit_striped_random_values_mod(
    std::mt19937_64 &rng, std::span<uint64_t> out, std::span<uint64_t> buf, const FixedWidthInt &modulus);

}  // namespace kickmix

#endif
