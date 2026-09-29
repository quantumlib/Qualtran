#include <chrono>
#include <iostream>
#include <random>
#include <sstream>

#include "kickmix/sim/sim.h"
#include "kickmix/simd/simd.h"
#include "kickmix/util/arg_parse.h"
#include "kickmix/util/binary_file_tools.h"
#include "main_util.h"

using namespace kickmix;

static std::string si2(double val) {
    char unit = ' ';
    if (val < 1) {
        if (val < 1) {
            val *= 1000;
            unit = 'm';
        }
        if (val < 1) {
            val *= 1000;
            unit = 'u';
        }
        if (val < 1) {
            val *= 1000;
            unit = 'n';
        }
        if (val < 1) {
            val *= 1000;
            unit = 'p';
        }
    } else {
        if (val > 1000) {
            val /= 1000;
            unit = 'k';
        }
        if (val > 1000) {
            val /= 1000;
            unit = 'M';
        }
        if (val > 1000) {
            val /= 1000;
            unit = 'G';
        }
        if (val > 1000) {
            val /= 1000;
            unit = 'T';
        }
    }
    std::stringstream ss;
    if (1 <= val && val < 10) {
        ss << static_cast<size_t>(val) << '.' << (static_cast<size_t>(val * 10) % 10);
    } else if (10 <= val && val < 100) {
        ss << ' ' << static_cast<size_t>(val);
    } else if (100 <= val && val < 1000) {
        ss << static_cast<size_t>(val / 1) * 1;
    } else {
        ss << val;
    }
    ss << ' ' << unit;
    return ss.str();
}

template <typename TWord>
void spin_helper(const Circuit &circuit, size_t total_shots, std::ostream &out) {
    Sim<TWord, false> sim(std::mt19937_64{0});
    sim.configure_for(circuit);
    sim.clear_for_shot();
    sim.global_phase_ref().randomize(sim.rng);
    for (auto &e : sim.qubit_span()) {
        e.randomize(sim.rng);
    }
    for (auto &e : sim.bit_span()) {
        e.randomize(sim.rng);
    }
    for (size_t k = 0; k < total_shots; k += sim.BATCH_SIZE) {
        sim.condition_stack.clear();
        sim.apply(circuit);
    }
    TWord v{};
    for (auto &e : sim.state_block) {
        v ^= e;
    }
    out << "hash: " << v << "\n";
}

int kickmix::main_spin(int argc, const char **argv) {
    try {
        check_for_unknown_arguments(
            {
                "--shots",
                "--mode",
                "--in",
                "--out",
            },
            {},
            "kickmix",
            "spin",
            argc,
            argv);
    } catch (const std::invalid_argument &ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    uint64_t total_shots = static_cast<uint64_t>(find_int64_argument("--shots", -1, 0, INT64_MAX, argc, argv));
    auto mode = find_enum_argument<std::string_view>(
        "--mode",
        nullptr,
        {
            {"64", "64"},
            {"128", "128"},
            {"256", "256"},
            {"512", "512"},
            {"64-polyfill", "64-polyfill"},
            {"128-polyfill", "128-polyfill"},
            {"256-polyfill", "256-polyfill"},
            {"512-polyfill", "512-polyfill"},
        },
        argc,
        argv);

    FILE *in = find_open_file_argument("--in", stdin, "rb", argc, argv);
    FILE *out = find_open_file_argument("--out", stdout, "wb", argc, argv);
    Circuit circuit = Circuit::from_kmx_or_kmb_file(in);
    if (in != stdin) {
        fclose(in);
    }

    auto start = std::chrono::steady_clock::now();
    std::stringstream output;

    if (mode == "64") {
        spin_helper<b64>(circuit, total_shots, output);
    } else if (mode == "128") {
        spin_helper<b128>(circuit, total_shots, output);
    } else if (mode == "256") {
        spin_helper<b256>(circuit, total_shots, output);
    } else if (mode == "512") {
        spin_helper<b512>(circuit, total_shots, output);
    } else if (mode == "64-polyfill") {
        spin_helper<b64_polyfill>(circuit, total_shots, output);
    } else if (mode == "128-polyfill") {
        spin_helper<b128_polyfill>(circuit, total_shots, output);
    } else if (mode == "256-polyfill") {
        spin_helper<b256_polyfill>(circuit, total_shots, output);
    } else if (mode == "512-polyfill") {
        spin_helper<b512_polyfill>(circuit, total_shots, output);
    } else {
        throw std::invalid_argument("Unrecognized mode.");
    }

    auto end = std::chrono::steady_clock::now();
    int64_t micros = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    size_t ops = circuit.num_ops * total_shots;
    if (micros == 0) {
        micros = 1;
    }
    output << "total simulated operations: " << ops << "\n";
    output << "              time elapsed: " << si2(micros / 1000000.0) << "s\n";
    output << si2(ops * 1000000LL / micros) << "ops/sec\n";
    std::string text = output.str();
    fwrite_else_throw(text.data(), text.size(), out);
    if (out != stdout) {
        fclose(out);
    }

    return 0;
}
