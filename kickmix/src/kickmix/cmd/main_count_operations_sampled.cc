#include <iostream>
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

int kickmix::main_count_operations_sampled(int argc, const char **argv) {
    check_for_unknown_arguments(
        {
            "--init",
            "--shots",
            "--groups",
            "--in",
            "--out",
        },
        {},
        "kickmix",
        "count_operations_sampled",
        argc,
        argv);

    uint64_t total_shots = static_cast<uint64_t>(find_int64_argument("--shots", -1, 0, INT64_MAX, argc, argv));
    bool groups = find_bool_argument("--groups", argc, argv);
    const char *init_str = find_argument("--init", argc, argv);

    FILE *in = find_open_file_argument("--in", stdin, "rb", argc, argv);
    Circuit circuit = Circuit::from_kmx_or_kmb_file(in);
    if (in != stdin) {
        fclose(in);
    }

    FILE *out = find_open_file_argument("--out", stdout, "wb", argc, argv);

    Sim<SIM_WORD, true> sim(externally_seeded_rng());
    sim.configure_for(circuit);
    sim.ignore_debug_print_operations = true;
    auto registers = circuit.register_data;

    std::vector<SimInitInstruction> init_instructions;
    if (init_str != nullptr) {
        init_instructions = SimInitInstruction::from_str_many(init_str);
    }

    auto remaining_shots = total_shots;
    uint64_t accumulated_counts[256]{};
    while (remaining_shots) {
        reset_sim_for_shot_using_init_instructions(sim, init_instructions);
        sim.apply(circuit);
        for (size_t k = 0; k < 256; k++) {
            for (size_t r = 0; r < remaining_shots && r < sim.BATCH_SIZE; r++) {
                accumulated_counts[k] += sim.new_op_counters[k].compute_total(r);
            }
            sim.new_op_counters[k].clear();
        }
        remaining_shots = std::max<uint64_t>(sim.BATCH_SIZE, remaining_shots) - sim.BATCH_SIZE;
    }
    fputs("                     type,         total,      average,       kept%\n", out);
    std::string line;

    uint64_t max_counts[256]{};
    for (size_t k = 0; k < circuit.num_ops; k++) {
        max_counts[(uint8_t)circuit.op_types[k]]++;
    }
    for (auto &e : max_counts) {
        e *= total_shots;
    }

    auto output_line = [&](std::string_view name, uint64_t op_count, uint64_t max_count) {
        line.clear();
        while (line.size() < 25 - name.size()) {
            line.push_back(' ');
        }
        line.append(name);
        line.push_back(',');

        auto total_str = std::to_string(op_count);
        while (line.size() < 25 + 15 - total_str.size()) {
            line.push_back(' ');
        }
        line.append(total_str);
        line.push_back(',');

        size_t safe_shots = std::max(total_shots, static_cast<uint64_t>(1));
        auto avg_str = std::to_string(op_count / safe_shots);
        while (line.size() < 25 + 15 + 12 - avg_str.size()) {
            line.push_back(' ');
        }
        line.append(avg_str);
        line.push_back('.');
        line.append(std::to_string((op_count * 10 / safe_shots) % 10));
        line.push_back(',');

        if (max_count == 0) {
            auto percent_str = std::to_string(100.0);
            while (line.size() < 25 + 15 + 12 + 12) {
                line.push_back(' ');
            }
            line.push_back('-');
            line.push_back('-');
            line.push_back('-');
            line.push_back('\n');
        } else {
            auto uv = 1000ULL * op_count / max_count;
            auto percent_str = std::to_string(uv / 10);
            while (line.size() < 25 + 15 + 12 + 12 - percent_str.size()) {
                line.push_back(' ');
            }
            line.append(percent_str);
            line.push_back('.');
            line.append(std::to_string(uv % 10));
            line.push_back('%');
            line.push_back('\n');
        }
        fputs(line.c_str(), out);
    };

    std::vector<std::vector<OpType>> gro{
        {OpType::NEG, OpType::NEG_IF},
        {OpType::BIT_INVERT, OpType::BIT_INVERT_IF},
        {OpType::BIT_STORE0, OpType::BIT_STORE0_IF},
        {OpType::BIT_STORE1, OpType::BIT_STORE1_IF},
        {OpType::X, OpType::X_IF},
        {OpType::Z, OpType::Z_IF},
        {OpType::R, OpType::R_IF},
        {OpType::HMR, OpType::HMR_IF},
        {OpType::CX, OpType::CX_IF},
        {OpType::CZ, OpType::CZ_IF},
        {OpType::SWAP, OpType::SWAP_IF},
        {OpType::CCX, OpType::CCX_IF},
        {OpType::CCZ, OpType::CCZ_IF},
        {OpType::Z_POW, OpType::Z_POW_IF},
        {OpType::DEBUG_PRINT_EMPTY,
         OpType::DEBUG_PRINT_Q,
         OpType::DEBUG_PRINT_C,
         OpType::DEBUG_PRINT_EMPTY_IF,
         OpType::DEBUG_PRINT_Q_IF,
         OpType::DEBUG_PRINT_C_IF},
        {OpType::POP_CONDITION},
        {OpType::PUSH_CONDITION},
    };
    for (const auto &e : gro) {
        uint64_t tot = 0;
        uint64_t max = 0;
        for (const auto &f : e) {
            uint64_t op_count = accumulated_counts[(uint8_t)f];
            if (f == OpType::POP_CONDITION || f == OpType::PUSH_CONDITION) {
                op_count = max_counts[(uint8_t)f];
            }
            tot += op_count;
            max += max_counts[(uint8_t)f];
        }
        output_line(OP_TYPE_NAME_TABLE[(uint8_t)e[0]], tot, max);
    }

    if (groups) {
        uint64_t magic_total = 0;
        uint64_t magic_max = 0;
        for (OpType kind : std::vector<OpType>{
                 OpType::CCX,
                 OpType::CCX_IF,
                 OpType::CCZ,
                 OpType::CCZ_IF,
                 OpType::Z_POW,
                 OpType::Z_POW_IF,
             }) {
            magic_total += accumulated_counts[static_cast<uint64_t>(kind)];
            magic_max += max_counts[static_cast<uint64_t>(kind)];
        }

        uint64_t stabilizer_total = 0;
        uint64_t stabilizer_max = 0;
        for (OpType kind : std::vector<OpType>{
                 OpType::CX_IF,
                 OpType::CZ_IF,
                 OpType::SWAP_IF,
                 OpType::R_IF,
                 OpType::HMR_IF,
                 OpType::CX,
                 OpType::CZ,
                 OpType::SWAP,
                 OpType::R,
                 OpType::HMR,
             }) {
            stabilizer_total += accumulated_counts[static_cast<uint64_t>(kind)];
            stabilizer_max += max_counts[static_cast<uint64_t>(kind)];
        }

        uint64_t pauli_op_total = 0;
        uint64_t pauli_op_max = 0;
        for (OpType kind : std::vector<OpType>{
                 OpType::X,
                 OpType::X_IF,
                 OpType::Z,
                 OpType::Z_IF,
             }) {
            pauli_op_total += accumulated_counts[static_cast<uint64_t>(kind)];
            pauli_op_max += max_counts[static_cast<uint64_t>(kind)];
        }

        uint64_t bit_op_total = 0;
        uint64_t bit_op_max = 0;
        for (OpType kind : std::vector<OpType>{
                 OpType::BIT_INVERT,
                 OpType::BIT_INVERT_IF,
                 OpType::BIT_STORE0,
                 OpType::BIT_STORE0_IF,
                 OpType::BIT_STORE1,
                 OpType::BIT_STORE1_IF,
             }) {
            bit_op_total += accumulated_counts[static_cast<uint64_t>(kind)];
            bit_op_max += max_counts[static_cast<uint64_t>(kind)];
        }

        uint64_t metadata_op_total = 0;
        uint64_t metadata_op_max = 0;
        for (OpType kind : std::vector<OpType>{
                 OpType::NEG,
                 OpType::NEG_IF,
             }) {
            metadata_op_total += accumulated_counts[static_cast<uint64_t>(kind)];
            metadata_op_max += max_counts[static_cast<uint64_t>(kind)];
        }

        uint64_t all_op_total = 0;
        uint64_t all_op_max = 0;
        for (OpType kind : std::vector<OpType>{
                 OpType::NEG,        OpType::NEG_IF,        OpType::BIT_INVERT, OpType::BIT_INVERT_IF,
                 OpType::BIT_STORE0, OpType::BIT_STORE0_IF, OpType::BIT_STORE1, OpType::BIT_STORE1_IF,
                 OpType::X,          OpType::X_IF,          OpType::Z,          OpType::Z_IF,
                 OpType::CX,         OpType::CX_IF,         OpType::CZ,         OpType::CZ_IF,
                 OpType::R,          OpType::R_IF,          OpType::HMR,        OpType::HMR_IF,
                 OpType::CCX,        OpType::CCX_IF,        OpType::CCZ,        OpType::CCZ_IF,
                 OpType::SWAP,       OpType::SWAP_IF,       OpType::Z_POW,      OpType::Z_POW_IF,
             }) {
            all_op_total += accumulated_counts[static_cast<uint64_t>(kind)];
            all_op_max += max_counts[static_cast<uint64_t>(kind)];
        }

        output_line("Metadata Operations", metadata_op_total, metadata_op_max);
        output_line("Bit Operations", bit_op_total, bit_op_max);
        output_line("Pauli Operations", pauli_op_total, pauli_op_max);
        output_line("Stabilizer Operations", stabilizer_total, stabilizer_max);
        output_line("Z_POW+CCX+CCZ Operations", magic_total, magic_max);
        output_line("Total Operations", all_op_total, all_op_max);
    }
    if (out != stdout) {
        fclose(out);
    }
    return 0;
}
