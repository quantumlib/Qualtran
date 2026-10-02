#ifndef KICKGEN_UTIL_FUZZER_H
#define KICKGEN_UTIL_FUZZER_H

#include <functional>
#include <map>

#include "kickmix/build/circuit_builder.h"
#include "kickmix/sim/sim.h"
#include "kickmix/simd/simd.h"

namespace kickmix {
struct TouchedRegister {
    FixedWidthInt *reg = nullptr;
    bool touched = false;
};
struct InputSample {
    std::map<std::string_view, TouchedRegister> registers;
    FixedWidthInt &operator[](std::string_view name) {
        auto &result = registers.at(name);
        result.touched = true;
        return *result.reg;
    }
};
struct OutputSample {
    std::map<std::string_view, TouchedRegister> registers;
    std::map<std::string_view, FixedWidthInt *> old_registers;
    FixedPrecisionAngle128 phase_half_turns;

    const FixedWidthInt &old(std::string_view name) const {
        return *old_registers.at(name);
    }
    FixedWidthInt &operator[](std::string_view name) {
        auto &result = registers.at(name);
        result.touched = true;
        return *result.reg;
    }
};

struct CircuitFuzzer {
   private:
    Sim<b64, false> sim;
    std::function<void(CircuitBuilder &builder, std::mt19937_64 &rng)> circuit_builder_sampler{};
    std::function<void(InputSample &sample, std::mt19937_64 &rng)> input_sampler{};
    std::function<void(OutputSample &sample)> output_sampler{};
    std::function<void(InputSample &ins, OutputSample &outs, std::mt19937_64 &rng)> input_and_output_sampler{};

   public:
    Circuit circuit{};
    bool disarmed_warning = false;

   private:
    CircuitBuilder builder;
    std::array<InputSample, decltype(sim)::BATCH_SIZE> input_samples{};
    std::array<OutputSample, decltype(sim)::BATCH_SIZE> output_samples{};
    std::array<std::vector<FixedWidthInt>, decltype(sim)::BATCH_SIZE> reg_old{};
    size_t seen_errors = 0;
    size_t seen_phase_errors = 0;
    size_t seen_shots = 0;
    size_t seen_configurations = 0;

    size_t fuzz_single_batch(size_t shots_left);
    void process_shot_data(
        const std::vector<FixedWidthInt> &actual_before,
        const std::vector<FixedWidthInt> &actual_after,
        const std::vector<FixedWidthInt> &expected_after,
        FixedPrecisionAngle128 expected_global_phase,
        FixedPrecisionAngle128 actual_global_phase,
        std::stringstream &error_message);

   public:
    bool ignore_global_phase = false;
    explicit CircuitFuzzer(std::mt19937_64 &&moved_rng);
    ~CircuitFuzzer();
    CircuitFuzzer(CircuitFuzzer &&) noexcept = default;
    CircuitFuzzer(const CircuitFuzzer &) = default;
    CircuitFuzzer &operator=(CircuitFuzzer &&) noexcept = default;
    CircuitFuzzer &operator=(const CircuitFuzzer &) = default;

    /// Specifies the function used to create circuit variations.
    ///
    /// Args:
    ///     func: The function that builds the circuit. It takes these arguments:
    ///         builder: The circuit builder to append operations into.
    ///         rng: A random number generator to use when making choices about what to create.
    void use_circuit_builder(std::function<void(CircuitBuilder &, std::mt19937_64 &)> func);

    /// Specifies the function used to sample inputs for the circuit.
    ///
    /// Registers whose names begin with "@" are "special registers". Other registers
    /// are "normal registers".
    ///
    /// If an input sampler isn't specified, normal registers are initialized randomly.
    ///
    /// Before the input sampler runs, special registers are initialized automatically,
    /// with the initializing depending on the special register's name. The sampler can
    /// overwrite the register's value in order to override the automatic initialization:
    ///     "@clean": Initialized to zero.
    ///     "@dirty": Initialized randomly.
    ///     "@unassigned" (i.e. qubits not part of a register): Initialized to zero.
    ///
    /// If a normal register isn't accessed by the input sampler, an exception will
    /// be raised when fuzzing (as this suggests that you forgot to initialize
    /// one of the registers).
    ///
    /// Args:
    ///     func: The function that samples inputs. It takes these arguments:
    ///         sample: The InputSample object to write register values into. Registers are accessed
    ///             by name (`sample["register_name"]`), and stored as FixedWidthInt values.
    ///         rng: A random number generator to use when making choices about what inputs to use.
    ///
    /// Example:
    ///     fuzzer.use_input_sample([](InputSample &sample, std::mt19937_64 &rng) {
    ///         // Pick a random modulus, ensuring it is odd and fills the register:
    ///         sample["modulus"].randomize(rng);
    ///         sample["modulus"].front_ref() = True;
    ///         sample["modulus"].back_ref() = True;
    ///         // Pick random values to add:
    ///         sample["target"].randomize_mod(rng, sample["modulus"]);
    ///         sample["offset"].randomize_mod(rng, sample["modulus"]);
    ///     });
    void use_input_sampler(std::function<void(InputSample &, std::mt19937_64 &)> func);

    /// Specifies the function used to compute expected outputs for the circuit.
    ///
    /// By default, a register's output value is equal to it's input value. When a
    /// register's output is supposed to differ from the input, the output sampler's job
    /// is to write the expected output into the sample object.
    ///
    /// Args:
    ///     func: The function that writes the outputs. It takes these arguments:
    ///         sample: The OutputSample object to write register values into. Registers are accessed
    ///             by name (`sample["register_name"]`), and stored as FixedWidthInt values. The
    ///             initial value of a register is its input value. After edits, the original input value
    ///             can be still be referenced via `sample.old("register_name")`.
    ///
    /// Example:
    ///     fuzzer.use_output_sampler([](OutputSample &sample) {
    ///         sample["target"].iadd_mod(sample["offset"], sample["modulus"]);
    ///     });
    ///
    ///     fuzzer.use_output_sampler([](OutputSample &sample) {
    ///         sample.phase ^= sample["target"] > 5;
    ///     });
    void use_output_sampler(std::function<void(OutputSample &)> func);

    /// Specifies a function used to compute both inputs and expected outputs for a circuit.
    ///
    /// This method exists for cases where it is tedious to generate outputs without the context
    /// used to generate the inputs. For example, it avoids the output sampler needing to recreate
    /// custom data structures used by the input sampler.
    ///
    /// Args:
    ///     func: The function that writes the inputs and outputs. It takes these arguments:
    ///         ins: An InputSample object to write input values to.
    ///         outs: An OutputSample object to write expected output values to. Note that,
    ///             unlike when using `use_output_sampler`, it is not the case that expected
    ///             outputs to being the same as their input. ALl expected outputs must be
    ///             specified.
    ///         rng: A random number generator to use when making choices about what inputs to use.
    ///
    /// Example:
    ///     fuzzer.use_input_and_output_sampler([](InputSample &ins, OutputSample &outs, std::mt19937_64 &rng) {
    ///         // Pick a random modulus, ensuring it is odd and fills the register:
    ///         ins["modulus"].randomize(rng);
    ///         ins["modulus"].front_ref() = True;
    ///         ins["modulus"].back_ref() = True;
    ///         // Pick random values to add:
    ///         ins["target"].randomize_mod(rng, ins["modulus"]);
    ///         ins["offset"].randomize_mod(rng, ins["modulus"]);
    ///         // Compute expected outputs.
    ///         outs["modulus"] = ins["modulus"]
    ///         outs["offset"] = ins["offset"]
    ///         outs["target"].iadd_mod(outs["offset"], outs["modulus"]);
    ///     });
    void use_input_and_output_sampler(std::function<void(InputSample &, OutputSample &, std::mt19937_64 &)> func);

    void switch_to_new_circuit();
    void fuzz_current_circuit(size_t shots);
    void fuzz(size_t num_variations, size_t num_shots_per_variation);
};
}  // namespace kickmix

#endif
