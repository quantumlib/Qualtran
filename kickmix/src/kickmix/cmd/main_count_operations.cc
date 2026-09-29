#include <array>
#include <iostream>
#include <random>

#include "kickmix/circuit/circuit.h"
#include "kickmix/util/arg_parse.h"
#include "main_util.h"

using namespace kickmix;

int kickmix::main_count_operations(int argc, const char **argv) {
    bool groups = find_bool_argument("--groups", argc, argv);

    try {
        check_for_unknown_arguments(
            {
                "--in",
                "--out",
                "--groups",
                "--magic",
            },
            {},
            "kickmix",
            "count_operations",
            argc,
            argv);
    } catch (const std::invalid_argument &ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    FILE *in = find_open_file_argument("--in", stdin, "rb", argc, argv);
    Circuit circuit = Circuit::from_kmx_or_kmb_file(in);
    if (in != stdin) {
        fclose(in);
    }
    size_t magic_total = 0;
    size_t stabilizer_total = 0;
    size_t pauli_op_total = 0;
    size_t bit_op_total = 0;
    size_t metadata_op_total = 0;
    size_t condition_op_total = 0;

    std::vector<uint64_t> counts;
    counts.resize(256);
    for (size_t k = 0; k < circuit.num_ops; k++) {
        counts[(uint8_t)circuit.op_types[k]]++;
    }

    magic_total += counts[(uint8_t)OpType::CCX];
    magic_total += counts[(uint8_t)OpType::CCZ];
    magic_total += counts[(uint8_t)OpType::CCX_IF];
    magic_total += counts[(uint8_t)OpType::CCZ_IF];
    magic_total += counts[(uint8_t)OpType::Z_POW];
    magic_total += counts[(uint8_t)OpType::Z_POW_IF];

    stabilizer_total += counts[(uint8_t)OpType::CX];
    stabilizer_total += counts[(uint8_t)OpType::CX_IF];
    stabilizer_total += counts[(uint8_t)OpType::CZ];
    stabilizer_total += counts[(uint8_t)OpType::CZ_IF];
    stabilizer_total += counts[(uint8_t)OpType::SWAP];
    stabilizer_total += counts[(uint8_t)OpType::SWAP_IF];
    stabilizer_total += counts[(uint8_t)OpType::R];
    stabilizer_total += counts[(uint8_t)OpType::R_IF];
    stabilizer_total += counts[(uint8_t)OpType::HMR];
    stabilizer_total += counts[(uint8_t)OpType::HMR_IF];

    pauli_op_total += counts[(uint8_t)OpType::X];
    pauli_op_total += counts[(uint8_t)OpType::X_IF];
    pauli_op_total += counts[(uint8_t)OpType::Z];
    pauli_op_total += counts[(uint8_t)OpType::Z_IF];

    bit_op_total += counts[(uint8_t)OpType::BIT_INVERT];
    bit_op_total += counts[(uint8_t)OpType::BIT_INVERT_IF];
    bit_op_total += counts[(uint8_t)OpType::BIT_STORE0];
    bit_op_total += counts[(uint8_t)OpType::BIT_STORE0_IF];
    bit_op_total += counts[(uint8_t)OpType::BIT_STORE1];
    bit_op_total += counts[(uint8_t)OpType::BIT_STORE1_IF];

    metadata_op_total += counts[(uint8_t)OpType::NEG];
    metadata_op_total += counts[(uint8_t)OpType::NEG_IF];

    condition_op_total += counts[(uint8_t)OpType::PUSH_CONDITION];
    condition_op_total += counts[(uint8_t)OpType::POP_CONDITION];

    FILE *out = find_open_file_argument("--out", stdout, "wb", argc, argv);
    if (find_bool_argument("--magic", argc, argv)) {
        fprintf(out, "%zu\n", magic_total);
        return 0;
    }

    std::vector<std::vector<OpType>> group_data{
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

    for (const auto &g : group_data) {
        size_t total = 0;
        for (const auto &e : g) {
            total += counts[(uint8_t)e];
        }
        for (size_t k2 = OP_TYPE_NAME_TABLE[(uint8_t)g[0]].size(); k2 < 25; k2++) {
            fputc(' ', out);
        }
        for (auto c : OP_TYPE_NAME_TABLE[(uint8_t)g[0]]) {
            fputc(c, out);
        }
        fprintf(out, ": %zu\n", total);
    }

    if (groups) {
        fprintf(out, "------------------------------------\n");
        fprintf(out, "                   Qubits: %zu\n", circuit.num_qubits);
        fprintf(out, "                     Bits: %zu\n", circuit.num_bits);
        fprintf(out, "      Metadata Operations: %zu\n", metadata_op_total);
        fprintf(out, "     Condition Operations: %zu\n", condition_op_total);
        fprintf(out, "           Bit Operations: %zu\n", bit_op_total);
        fprintf(out, "         Pauli Operations: %zu\n", pauli_op_total);
        fprintf(out, "    Stabilizer Operations: %zu\n", stabilizer_total);
        fprintf(out, " Z_POW+CCX+CCZ Operations: %zu\n", magic_total);
        fprintf(out, "------------------------------------\n");
        fprintf(out, "         Total Operations: %zu\n", circuit.num_ops);
    }

    if (out != stdout) {
        fclose(out);
    }
    return 0;
}
