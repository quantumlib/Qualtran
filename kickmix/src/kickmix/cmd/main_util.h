#ifndef KICKMIX_MAIN_UTIL_H
#define KICKMIX_MAIN_UTIL_H

#include <string>

#include "kickmix/sim/sim.h"
#include "kickmix/simd/simd.h"
#include "kickmix/util/fixed_width_int.h"

namespace kickmix {
using SIM_WORD = b64;

int main_generate_circuit(int argc, const char **argv);
int main_count_operations(int argc, const char **argv);
int main_count_operations_sampled(int argc, const char **argv);
int main_sample(int argc, const char **argv);
int main_spin(int argc, const char **argv);
int main_diagram(int argc, const char **argv);
int main_kmb2kmx(int argc, const char **argv);
int main_kmx2kmb(int argc, const char **argv);

std::string big_int_to_str(size_t num_bits, const FixedWidthInt &words, bool decimal, bool hex);

template <typename TWord, bool b>
void reset_sim_for_shot_using_init_instructions(
    Sim<TWord, b> &sim, std::span<const SimInitInstruction> init_instructions) {
    sim.clear_for_shot();
    for (const auto &inst : init_instructions) {
        const auto &r = sim.registers[inst.target_register.id];
        if (inst.randomize) {
            // DIDNTDO: generalize generate_bit_striped_random_values_mod so it can speed this up.
            if (inst.value.num_bits == 0) {
                for (size_t k = 0; k < sim.BATCH_SIZE; k++) {
                    sim.register_buffers[k][inst.target_register.id].randomize(sim.rng);
                }
            } else {
                for (size_t k = 0; k < sim.BATCH_SIZE; k++) {
                    sim.register_buffers[k][inst.target_register.id].randomize_mod(sim.rng, inst.value);
                }
            }
            sim.copy_single_register_buffer_into_bit_packed_state(inst.target_register);
        } else {
            for (size_t k = 0; k < r.contents.size(); k++) {
                auto &e = sim.val_for(inst.target_register, k);
                if (inst.value[k]) {
                    e.clear_to_max();
                } else {
                    e.clear_to_zero();
                }
            }
        }
    }
}
}  // namespace kickmix

#endif
