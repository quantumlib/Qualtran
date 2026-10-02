#include <random>

#include "kickmix/circuit/circuit.h"
#include "kickmix/sim/sim.h"
#include "kickmix/util/arg_parse.h"
#include "main_util.h"

using namespace kickmix;

static std::mt19937_64 externally_seeded_rng() {
    std::random_device d;
    std::seed_seq seq{d(), d(), d(), d(), d(), d(), d(), d()};
    std::mt19937_64 result(seq);
    return result;
}

int kickmix::main_sample(int argc, const char **argv) {
    bool decimal = find_bool_argument("--decimal", argc, argv);
    bool hex = find_bool_argument("--hex", argc, argv);
    uint32_t total_shots = static_cast<uint32_t>(find_int64_argument("--shots", 1, 0, UINT32_MAX, argc, argv));
    const char *init_str = find_argument("--init", argc, argv);
    try {
        check_for_unknown_arguments(
            {
                "--decimal",
                "--hex",
                "--init",
                "--shots",
            },
            {},
            "kickmix sample",
            nullptr,
            argc,
            argv);
    } catch (const std::invalid_argument &ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    std::vector<SimInitInstruction> init_instructions;
    if (init_str != nullptr) {
        init_instructions = SimInitInstruction::from_str_many(init_str);
    }

    Circuit circuit = Circuit::from_kmx_file(stdin);
    Sim<SIM_WORD, false> sim(externally_seeded_rng());
    sim.configure_for(circuit);
    auto registers = circuit.register_data;

    std::vector<size_t> col_sizes;
    for (size_t k = 0; k < registers.size(); k++) {
        if (registers[k].name.empty()) {
            std::cout << "(before)r" << k << ",";
            std::cout << "r" << k << ",";
        } else {
            std::cout << "(before)" << registers[k].name << k << ",";
            std::cout << registers[k].name << k << ",";
        }
        col_sizes.push_back(std::to_string(k).size() + 6);
    }
    std::cout << "global_phase\n";

    auto remaining_shots = total_shots;
    while (remaining_shots) {
        reset_sim_for_shot_using_init_instructions(sim, init_instructions);
        sim.copy_bit_packed_state_into_register_buffer();
        std::swap(sim.register_buffers, sim.register_buffers2);
        sim.apply(circuit);
        sim.copy_bit_packed_state_into_register_buffer();
        for (size_t shot_idx = 0; shot_idx < sim.BATCH_SIZE && shot_idx < remaining_shots; shot_idx++) {
            for (size_t reg_idx = 0; reg_idx < registers.size(); reg_idx++) {
                size_t m = sim.registers[reg_idx].contents.size();
                const auto &buf_cur = sim.register_buffers[shot_idx][reg_idx];
                const auto &buf_prev = sim.register_buffers2[shot_idx][reg_idx];
                auto vv = big_int_to_str(m, buf_prev, decimal, hex);
                for (size_t c = 0; c + vv.size() < col_sizes[reg_idx]; c++) {
                    putc(' ', stdout);
                }
                std::cout << vv;
                std::cout << ",";
                vv = big_int_to_str(m, buf_cur, decimal, hex);
                for (size_t c = 0; c + vv.size() < col_sizes[reg_idx]; c++) {
                    putc(' ', stdout);
                }
                std::cout << vv;
                std::cout << ",";
            }

            if (sim.global_phase_ref().bit(shot_idx)) {
                std::cout << "          -1\n";
            } else {
                std::cout << "           1\n";
            }
        }
        remaining_shots = std::max(static_cast<uint32_t>(sim.BATCH_SIZE), remaining_shots) - sim.BATCH_SIZE;
    }
    return 0;
}
