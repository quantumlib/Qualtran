#ifndef KICKMIX_CIRCUIT_UTIL_H
#define KICKMIX_CIRCUIT_UTIL_H

#include <span>

#include "circuit.h"

namespace kickmix {

size_t compute_reaction_depth(size_t num_qubits, size_t num_bits, std::span<const Op> operations);
size_t compute_num_touched_qubits(const Circuit &circuit);

}  // namespace kickmix

#endif
