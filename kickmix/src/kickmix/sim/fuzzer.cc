#include "fuzzer.h"

#include "kickmix/sim/sim.h"

using namespace kickmix;

static std::string grid_string(std::map<std::pair<size_t, size_t>, std::string> m) {
    size_t num_x = 0;
    size_t num_y = 0;
    for (const auto &kv : m) {
        num_x = std::max(num_x, kv.first.first + 1);
        num_y = std::max(num_y, kv.first.second + 1);
    }
    std::vector<size_t> column_widths;
    for (size_t x = 0; x < num_x; x++) {
        size_t w = 0;
        for (size_t y = 0; y < num_y; y++) {
            auto e = m.find({x, y});
            if (e != m.end()) {
                w = std::max(w, e->second.size());
            }
        }
        column_widths.push_back(w);
    }

    std::stringstream ss;
    for (size_t y = 0; y < num_y; y++) {
        if (y) {
            ss << "\n";
        }
        for (size_t x = 0; x < num_x; x++) {
            auto e = m.find({x, y});
            std::string_view text;
            if (e != m.end()) {
                text = e->second;
            } else {
                text = "";
            }
            for (size_t k = text.size(); k < column_widths[x]; k++) {
                ss << ' ';
            }
            if (x) {
                ss << ' ';
            }
            ss << text;
        }
    }
    return ss.str();
}

void CircuitFuzzer::process_shot_data(
    const std::vector<FixedWidthInt> &actual_before,
    const std::vector<FixedWidthInt> &actual_after,
    const std::vector<FixedWidthInt> &expected_after,
    FixedPrecisionAngle128 expected_global_phase,
    FixedPrecisionAngle128 actual_global_phase,
    std::stringstream &error_message) {
    seen_shots += 1;
    if (actual_after != expected_after || (!ignore_global_phase && actual_global_phase != expected_global_phase)) {
        seen_phase_errors += actual_global_phase != expected_global_phase;
        seen_errors += 1;
        if (seen_errors == 1) {
            std::map<std::pair<size_t, size_t>, std::string> grid;
            if (!ignore_global_phase) {
                grid[{1, 0}] = "phase";
            }
            grid[{0, 1}] = "input";
            grid[{0, 2}] = "output";
            grid[{0, 3}] = "expected_output";
            grid[{0, 4}] = "diff";

            if (!ignore_global_phase) {
                grid[{1, 1}] = "0.0π";
                grid[{1, 2}] = actual_global_phase.str();
                grid[{1, 3}] = expected_global_phase.str();
                grid[{1, 4}] = actual_global_phase != expected_global_phase ? "X" : "";
            }
            size_t col = 2;
            for (size_t k = 0; k < sim.registers.size(); k++) {
                grid[{col, 0}] = builder.mut.register_data[k].name + "(r" + std::to_string(k) + ")";
                grid[{col, 1}] = actual_before[k].bin();
                grid[{col, 2}] = actual_after[k].bin();
                grid[{col, 3}] = expected_after[k].bin();
                auto f = expected_after[k];
                f ^= actual_after[k];
                if (f.non_zero()) {
                    auto &s = grid[{col, 4}];
                    for (auto c : f.bin()) {
                        s.push_back(" X"[c == '1']);
                    }
                }
                col += 1;
            }

            error_message << "\nRegisters:\n";
            for (size_t k = 0; k < builder.mut.register_data.size(); k++) {
                const auto &e = builder.mut.register_data[k];
                size_t num_qubits = 0;
                size_t num_bits = 0;
                for (auto &r : e.contents) {
                    num_qubits += r.is_qubit();
                    num_bits += !r.is_qubit();
                }
                error_message << "    " << e.name << "(r" << k << "): ";
                if (num_bits == 0 && num_qubits == 0) {
                    error_message << "(empty)";
                } else if (num_bits == 0) {
                    error_message << num_qubits << "q";
                } else if (num_qubits == 0) {
                    error_message << num_bits << "b";
                } else {
                    for (auto &r : e.contents) {
                        error_message << "bq"[r.is_qubit()];
                    }
                }
                error_message << "\n";
            }

            error_message << "\nState:\n";
            error_message << grid_string(grid);
            error_message << "\n";

            if (circuit.num_qubits < 40 && circuit.num_ops < 1000) {
                error_message << "\nCircuit:\n";
                error_message << circuit.text_diagram();
            }
            error_message << "\n";
        }
    }
}

CircuitFuzzer::CircuitFuzzer(std::mt19937_64 &&moved_rng) : sim(std::move(moved_rng)) {
}

CircuitFuzzer::~CircuitFuzzer() {
    if (seen_shots == 0 && !disarmed_warning) {
        std::cerr << "WARNING: a circuit fuzzer didn't do any fuzzing!\n";
    }
}

void CircuitFuzzer::use_circuit_builder(std::function<void(CircuitBuilder &, std::mt19937_64 &)> v) {
    circuit_builder_sampler = v;
}
void CircuitFuzzer::use_input_sampler(std::function<void(InputSample &, std::mt19937_64 &)> v) {
    input_sampler = v;
}
void CircuitFuzzer::use_output_sampler(std::function<void(OutputSample &)> v) {
    output_sampler = v;
}
void CircuitFuzzer::use_input_and_output_sampler(
    std::function<void(InputSample &, OutputSample &, std::mt19937_64 &)> v) {
    input_and_output_sampler = v;
}

void CircuitFuzzer::switch_to_new_circuit() {
    if (!circuit_builder_sampler) {
        throw std::invalid_argument("Didn't call 'use_circuit_builder(...)'");
    }
    seen_configurations++;

    {
        CircuitBuilder new_builder;
        circuit_builder_sampler(new_builder, sim.rng);
        size_t nq = new_builder.mut.compute_cur_num_qubits();
        std::vector<bool> mq(nq, false);
        for (auto &e : new_builder.mut.register_data) {
            for (auto &q : e.contents) {
                if (q.is_qubit()) {
                    mq[q.untagged_id()] = true;
                }
            }
        }
        RegisterId r{new_builder.next_register_id};
        bool has_unassigned = false;
        for (auto e : mq) {
            has_unassigned |= !e;
        }
        if (has_unassigned) {
            new_builder.append_register(0, "@unassigned");
            for (size_t k = 0; k < nq; k++) {
                if (!mq[k]) {
                    new_builder.mut.register_data[r.id].contents.push_back(QubitId((uint32_t)k));
                }
            }
        }
        circuit = new_builder.finish_circuit();
        builder = std::move(new_builder);
    }
    sim.configure_for(circuit);
    reg_old = sim.register_buffers;

    for (size_t s = 0; s < sim.BATCH_SIZE; s++) {
        input_samples[s].registers.clear();
        output_samples[s].old_registers.clear();
        output_samples[s].registers.clear();
        for (size_t k = 0; k < builder.mut.register_data.size(); k++) {
            auto &e = builder.mut.register_data[k];
            input_samples[s].registers[e.name].reg = &sim.register_buffers[s][k];
            output_samples[s].old_registers[e.name] = &sim.register_buffers[s][k];
            output_samples[s].registers[e.name].reg = &sim.register_buffers2[s][k];
        }
    }
}

size_t CircuitFuzzer::fuzz_single_batch(size_t shots_left) {
    std::stringstream error_message;
    sim.clear_for_shot();

    size_t shots_performed = 0;
    for (size_t s = 0; s < sim.BATCH_SIZE && s < shots_left; s++) {
        shots_performed += 1;
        for (auto &kv : input_samples[s].registers) {
            kv.second.touched = false;
            if (kv.first.starts_with("@")) {
                kv.second.touched = true;
                if (kv.first == "@dirty") {
                    kv.second.reg->randomize(sim.rng);
                } else if (kv.first == "@clean") {
                    kv.second.reg->clear_to_zero();
                } else if (kv.first == "@unassigned") {
                    kv.second.reg->clear_to_zero();
                } else {
                    throw std::invalid_argument("Unrecognized special register name '" + std::string(kv.first) + "'");
                }
                output_samples[s][kv.first].write_from(*kv.second.reg);
            }
        }
        if (input_sampler && input_and_output_sampler) {
            throw std::invalid_argument("Specified both an input_sampler and an input_and_output_sampler");
        }
        if (output_sampler && input_and_output_sampler) {
            throw std::invalid_argument("Specified both an output_sampler and an input_and_output_sampler");
        }

        if (input_and_output_sampler) {
            output_samples[s].phase_half_turns = 0;
            input_and_output_sampler(input_samples[s], output_samples[s], sim.rng);
            for (auto &kv : input_samples[s].registers) {
                if (!kv.second.touched) {
                    std::stringstream ss;
                    ss << "input_and_output_sampler didn't touch the input register named '" << kv.first << "'.";
                    throw std::invalid_argument(ss.str());
                }
            }
            for (auto &kv : output_samples[s].registers) {
                if (!kv.second.touched) {
                    std::stringstream ss;
                    ss << "input_and_output_sampler didn't touch the output register named '" << kv.first << "'.";
                    throw std::invalid_argument(ss.str());
                }
            }
        } else {
            if (input_sampler) {
                input_sampler(input_samples[s], sim.rng);
                for (auto &kv : input_samples[s].registers) {
                    if (!kv.second.touched) {
                        std::stringstream ss;
                        ss << "input_sampler didn't touch the register named '" << kv.first << "'.";
                        throw std::invalid_argument(ss.str());
                    }
                }
            } else {
                for (auto &kv : input_samples[s].registers) {
                    if (!kv.first.starts_with("@")) {
                        kv.second.reg->randomize(sim.rng);
                    }
                }
            }
            output_samples[s].phase_half_turns = 0;
            for (auto &kv : output_samples[s].registers) {
                *kv.second.reg = *input_samples[s].registers[kv.first].reg;
            }
            if (output_sampler) {
                output_sampler(output_samples[s]);
            }
        }
    }

    sim.copy_register_buffer_into_bit_packed_state();
    std::swap(sim.register_buffers, reg_old);
    sim.apply(circuit);
    sim.copy_bit_packed_state_into_register_buffer();
    for (size_t s = 0; s < shots_performed; s++) {
        process_shot_data(
            reg_old[s],
            sim.register_buffers[s],
            sim.register_buffers2[s],
            output_samples[s].phase_half_turns,
            sim.read_shot_phase(s),
            error_message);
    }
    std::swap(sim.register_buffers, reg_old);
    if (seen_errors) {
        std::stringstream full_message;
        full_message << "Fuzz testing failed.\n";
        full_message << "    Circuit configurations: " << seen_configurations << "\n";
        full_message << "    Total shots: " << seen_shots << "\n";
        full_message << "    Total errors: " << seen_errors << "\n";
        full_message << "    Total phase errors: " << seen_phase_errors << "\n";
        full_message << "\n" << error_message.str();
        throw std::invalid_argument(full_message.str());
    }

    return shots_performed;
}

void CircuitFuzzer::fuzz_current_circuit(size_t shots) {
    while (shots > 0) {
        shots -= fuzz_single_batch(shots);
    }
}

void CircuitFuzzer::fuzz(size_t num_variations, size_t num_shots_per_variation) {
    for (size_t k = 0; k < num_variations; k++) {
        switch_to_new_circuit();
        fuzz_current_circuit(num_shots_per_variation);
    }
}
