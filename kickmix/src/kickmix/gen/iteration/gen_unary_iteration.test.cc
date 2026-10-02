#include "gen_unary_iteration.h"

#include <bit>

#include "gtest/gtest.h"

#include "kickmix/sim/fuzzer.h"
#include "kickmix/util/circuit_testing.test.h"
#include "test.test.h"

using namespace kickmix;

namespace {

/// Random parameters for a single fuzz test case.
struct FuzzConfig {
    size_t num_address_bits = 0;
    /// Whether the iteration is controlled by a qubit or bit rather than unconditional.
    bool controlled = false;
    std::vector<uint64_t> address_values;
};

/// Returns the Toffoli count for visiting in-range `visited` values in order.
///
/// Positioning on the first value costs `num_address_bits` Toffolis when controlled
/// by a qubit, or `num_address_bits - 1` when uncontrolled. Each change of value
/// from `a` to `b` then adds `bit_width(a ^ b) - 1` Toffolis.
uint64_t expected_toffolis(size_t num_address_bits, bool control_is_qubit, const std::vector<uint64_t> &visited) {
    if (num_address_bits == 0) {
        return 0;
    }
    uint64_t total = (uint64_t)(num_address_bits - 1) + (control_is_qubit ? 1 : 0);
    for (size_t k = 1; k < visited.size(); k++) {
        if (visited[k - 1] != visited[k]) {
            total += (uint64_t)std::bit_width(visited[k - 1] ^ visited[k]) - 1;
        }
    }
    return total;
}

}  // namespace

TEST(UnaryIterationCursor, fuzz_equality) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    FuzzConfig config;

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        config.num_address_bits = rng() % 5;
        config.controlled = rng() % 2;
        uint64_t num_values = uint64_t{1} << config.num_address_bits;
        size_t address_space_size = rng() % 5 + 1;
        config.address_values.clear();
        for (size_t k = 0; k < address_space_size; k++) {
            config.address_values.push_back(rng() % num_values);
        }

        auto address = builder.append_register(config.num_address_bits, "address");
        auto ctrl = builder.append_register(1, "ctrl");
        auto target = builder.append_register(address_space_size, "target");
        auto clean = builder.append_register(std::max(config.num_address_bits, size_t{1}), "@clean");

        QubitOrTrue control = config.controlled ? QubitOrTrue(ctrl[0]) : QubitOrTrue(true);
        UnaryIterationCursor cursor(builder, CircuitGenCtx{.clean_workspace = clean}, address, control);
        for (size_t k = 0; k < address_space_size; k++) {
            cursor.move_to(config.address_values[k]);
            builder.cx(cursor.match_qubit(), target[k]);
        }
        cursor.close();
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        uint64_t address = (uint64_t)sample["address"];
        bool control = !config.controlled || (bool)sample["ctrl"];
        for (size_t k = 0; k < config.address_values.size(); k++) {
            sample["target"].bit_ref(k) ^= control && address == config.address_values[k];
        }
    });

    fuzzer.fuzz(50, 64);
}

TEST(UnaryIterationCursor, fuzz_unrepresentable_address_values) {
    // Moving to values >= address_space_size() emits no Toffolis and leaves
    // `match_qubit()` at |0>.
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    FuzzConfig config;

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        config.num_address_bits = rng() % 4 + 1;
        config.controlled = rng() % 2;
        uint64_t num_values = uint64_t{1} << config.num_address_bits;
        size_t address_space_size = rng() % 5 + 1;
        config.address_values.clear();
        for (size_t k = 0; k < address_space_size; k++) {
            config.address_values.push_back(rng() % (2 * num_values));
        }

        auto address = builder.append_register(config.num_address_bits, "address");
        auto ctrl = builder.append_register(1, "ctrl");
        auto target = builder.append_register(address_space_size, "target");
        auto clean = builder.append_register(config.num_address_bits, "@clean");

        QubitOrTrue control = config.controlled ? QubitOrTrue(ctrl[0]) : QubitOrTrue(true);
        UnaryIterationCursor cursor(builder, CircuitGenCtx{.clean_workspace = clean}, address, control);
        for (size_t k = 0; k < address_space_size; k++) {
            cursor.move_to(config.address_values[k]);
            builder.cx(cursor.match_qubit(), target[k]);
        }
        cursor.close();
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        uint64_t address = (uint64_t)sample["address"];
        bool control = !config.controlled || (bool)sample["ctrl"];
        for (size_t k = 0; k < config.address_values.size(); k++) {
            sample["target"].bit_ref(k) ^= control && address == config.address_values[k];
        }
    });

    fuzzer.fuzz(50, 64);
}

TEST(UnaryIterationCursor, fuzz_classical_bit_control) {
    // A classical bit control is applied via `raii_push_condition` around the cursor's lifetime.
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    FuzzConfig config;

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        config.num_address_bits = rng() % 4;
        config.controlled = rng() % 2;
        uint64_t num_values = uint64_t{1} << config.num_address_bits;
        size_t address_space_size = rng() % 5 + 1;
        config.address_values.clear();
        for (size_t k = 0; k < address_space_size; k++) {
            config.address_values.push_back(rng() % num_values);
        }

        auto address = builder.append_register(config.num_address_bits, "address");
        auto target = builder.append_register(address_space_size, "target");
        auto clean = builder.append_register(std::max(config.num_address_bits, size_t{1}), "@clean");

        auto control_bit = builder.alloc_clean_raii_bit();
        if (config.controlled) {
            builder.bit_invert(control_bit.bit);
        }
        auto pushed = builder.raii_push_condition(control_bit.bit);
        UnaryIterationCursor cursor(builder, CircuitGenCtx{.clean_workspace = clean}, address);
        for (size_t k = 0; k < address_space_size; k++) {
            cursor.move_to(config.address_values[k]);
            builder.cx(cursor.match_qubit(), target[k]);
        }
        cursor.close();
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        uint64_t address = (uint64_t)sample["address"];
        for (size_t k = 0; k < config.address_values.size(); k++) {
            sample["target"].bit_ref(k) ^= config.controlled && address == config.address_values[k];
        }
    });

    fuzzer.fuzz(50, 64);
}

TEST(gen_unary_iteration, fuzz_equality) {
    // Visits the given values in order and closes the cursor before returning.
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    FuzzConfig config;

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        config.num_address_bits = rng() % 5;
        config.controlled = rng() % 2;
        uint64_t num_values = uint64_t{1} << config.num_address_bits;
        size_t address_space_size = rng() % 6;
        config.address_values.clear();
        for (size_t k = 0; k < address_space_size; k++) {
            config.address_values.push_back(rng() % (2 * num_values));
        }

        auto address = builder.append_register(config.num_address_bits, "address");
        auto ctrl = builder.append_register(1, "ctrl");
        auto target = builder.append_register(std::max(address_space_size, size_t{1}), "target");
        auto clean = builder.append_register(std::max(config.num_address_bits, size_t{1}), "@clean");

        QubitOrTrue control = config.controlled ? QubitOrTrue(ctrl[0]) : QubitOrTrue(true);
        size_t k = 0;
        gen_unary_iteration(
            builder,
            CircuitGenCtx{.clean_workspace = clean},
            address,
            control,
            config.address_values,
            [&](uint64_t address_value, QubitId match_qubit) {
                ASSERT_EQ(address_value, config.address_values[k]);
                builder.cx(match_qubit, target[k]);
                k++;
            });
        ASSERT_EQ(k, address_space_size);
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        uint64_t address = (uint64_t)sample["address"];
        bool control = !config.controlled || (bool)sample["ctrl"];
        for (size_t k = 0; k < config.address_values.size(); k++) {
            sample["target"].bit_ref(k) ^= control && address == config.address_values[k];
        }
    });

    fuzzer.fuzz(50, 64);
}

TEST(UnaryIterationCursor, toffoli_count_of_a_controlled_sweep) {
    // Sweeping [0, 73) on an 8-bit address with a qubit control costs 78 Toffolis.
    CircuitBuilder builder;
    auto address = builder.append_register(8);
    auto ctrl = builder.append_register(1);
    auto clean = builder.reserve_qubits(8);

    std::vector<uint64_t> address_values;
    for (uint64_t v = 0; v < 73; v++) {
        address_values.push_back(v);
    }
    {
        UnaryIterationCursor cursor(builder, CircuitGenCtx{.clean_workspace = clean}, address, ctrl[0]);
        for (uint64_t v : address_values) {
            cursor.move_to(v);
        }
        cursor.close();
    }

    ASSERT_EQ(expected_toffolis(8, true, address_values), 78);
    ASSERT_EQ(builder.finish_circuit().max_magic(), 78);
}

TEST(UnaryIterationCursor, toffoli_count_of_an_uncontrolled_sweep) {
    // Omitting the control saves one Toffoli.
    CircuitBuilder builder;
    auto address = builder.append_register(8);
    auto clean = builder.reserve_qubits(8);

    std::vector<uint64_t> address_values;
    for (uint64_t v = 0; v < 73; v++) {
        address_values.push_back(v);
    }
    {
        UnaryIterationCursor cursor(builder, CircuitGenCtx{.clean_workspace = clean}, address, true);
        for (uint64_t v : address_values) {
            cursor.move_to(v);
        }
        cursor.close();
    }

    ASSERT_EQ(expected_toffolis(8, false, address_values), 77);
    ASSERT_EQ(builder.finish_circuit().max_magic(), 77);
}

TEST(UnaryIterationCursor, toffoli_count_of_a_bit_controlled_sweep) {
    // A classical bit condition around an uncontrolled cursor uses the same number of Toffolis.
    CircuitBuilder builder;
    auto address = builder.append_register(8);
    auto clean = builder.reserve_qubits(8);

    std::vector<uint64_t> address_values;
    for (uint64_t v = 0; v < 73; v++) {
        address_values.push_back(v);
    }
    {
        auto control_bit = builder.alloc_clean_raii_bit();
        builder.bit_invert(control_bit.bit);
        auto pushed = builder.raii_push_condition(control_bit.bit);
        UnaryIterationCursor cursor(builder, CircuitGenCtx{.clean_workspace = clean}, address);
        for (uint64_t v : address_values) {
            cursor.move_to(v);
        }
        cursor.close();
    }

    ASSERT_EQ(expected_toffolis(8, false, address_values), 77);
    ASSERT_EQ(builder.finish_circuit().max_magic(), 77);
}

TEST(UnaryIterationCursor, moving_between_unrepresentable_and_representable_address_values) {
    // Moving to an out-of-range value costs 0 Toffolis, and moving back into range
    // costs a full initial position.
    for (bool controlled : {false, true}) {
        CircuitBuilder builder;
        auto address = builder.append_register(4);
        auto ctrl = builder.append_register(1);
        auto clean = builder.reserve_qubits(4);
        QubitOrTrue control = controlled ? QubitOrTrue(ctrl[0]) : QubitOrTrue(true);
        uint64_t full_build = 3 + (controlled ? 1 : 0);
        {
            UnaryIterationCursor cursor(builder, CircuitGenCtx{.clean_workspace = clean}, address, control);
            cursor.move_to(5);
            cursor.move_to(20);
            cursor.move_to(21);
            cursor.move_to(6);
            cursor.close();
        }
        ASSERT_EQ(builder.finish_circuit().max_magic(), 2 * full_build) << "controlled=" << controlled;
    }
}

TEST(UnaryIterationCursor, address_wider_than_63_qubits_is_rejected) {
    CircuitBuilder builder;
    auto address = builder.append_register(64);
    auto clean = builder.reserve_qubits(64);
    ASSERT_THROW(
        { UnaryIterationCursor cursor(builder, CircuitGenCtx{.clean_workspace = clean}, address, true); },
        std::invalid_argument);
}

TEST(UnaryIterationCursor, toffoli_count_of_random_access_matches_the_closed_form) {
    // Jumping between arbitrary in-range values costs `bit_width(a ^ b) - 1` per jump.
    std::mt19937_64 rng = INDEPENDENT_TEST_RNG();
    for (size_t rep = 0; rep < 20; rep++) {
        size_t num_address_bits = rng() % 7 + 1;
        bool controlled = rng() % 2;
        std::vector<uint64_t> address_values;
        for (size_t k = 0; k < rng() % 40 + 1; k++) {
            address_values.push_back(rng() % (uint64_t{1} << num_address_bits));
        }

        CircuitBuilder builder;
        auto address = builder.append_register(num_address_bits);
        auto ctrl = builder.append_register(1);
        auto clean = builder.reserve_qubits(num_address_bits);
        QubitOrTrue control = controlled ? QubitOrTrue(ctrl[0]) : QubitOrTrue(true);
        {
            UnaryIterationCursor cursor(builder, CircuitGenCtx{.clean_workspace = clean}, address, control);
            for (uint64_t v : address_values) {
                cursor.move_to(v);
            }
            cursor.close();
        }

        ASSERT_EQ(builder.finish_circuit().max_magic(), expected_toffolis(num_address_bits, controlled, address_values))
            << "num_address_bits=" << num_address_bits << " controlled=" << controlled;
    }
}

TEST(UnaryIterationCursor, values_outside_the_address_space_are_free) {
    CircuitBuilder builder;
    auto address = builder.append_register(4);
    auto clean = builder.reserve_qubits(4);
    {
        UnaryIterationCursor cursor(builder, CircuitGenCtx{.clean_workspace = clean}, address, true);
        for (uint64_t v = 16; v < 32; v++) {
            cursor.move_to(v);
        }
        cursor.close();
    }

    CircuitBuilder reference;
    reference.append_register(4);
    ASSERT_EQ(builder.finish_circuit(), reference.finish_circuit());
}

TEST(UnaryIterationCursor, construction_emits_nothing_and_parks_past_the_last_address_value) {
    // Construction starts at `address_space_size()` with `match_qubit()` at |0>.
    CircuitBuilder builder;
    auto address = builder.append_register(3);
    auto clean = builder.reserve_qubits(3);
    {
        UnaryIterationCursor cursor(builder, CircuitGenCtx{.clean_workspace = clean}, address);
        ASSERT_EQ(cursor.cur_address_value(), cursor.address_space_size());
        ASSERT_EQ(cursor.match_qubit(), clean[0]);
        cursor.move_to(9);
        ASSERT_EQ(cursor.match_qubit(), clean[0]);
        cursor.close();
    }

    CircuitBuilder reference;
    reference.append_register(3);
    ASSERT_EQ(builder.finish_circuit(), reference.finish_circuit());
}

TEST(UnaryIterationCursor, close_parks_the_cursor_and_is_idempotent_and_reusable) {
    // Closing parks at `address_space_size()`, is idempotent, and allows reuse.
    CircuitBuilder builder;
    auto address = builder.append_register(3);
    auto clean = builder.reserve_qubits(3);
    UnaryIterationCursor cursor(builder, CircuitGenCtx{.clean_workspace = clean}, address);
    ASSERT_EQ(cursor.cur_address_value(), cursor.address_space_size());

    cursor.move_to(5);
    ASSERT_EQ(cursor.cur_address_value(), 5);
    cursor.close();
    ASSERT_EQ(cursor.cur_address_value(), cursor.address_space_size());
    cursor.close();
    ASSERT_EQ(cursor.cur_address_value(), cursor.address_space_size());
    ASSERT_EQ(cursor.match_qubit(), clean[0]);

    cursor.move_to(3);
    ASSERT_EQ(cursor.cur_address_value(), 3);
    cursor.close();

    ASSERT_EQ(builder.finish_circuit().max_magic(), 4);
}

TEST(UnaryIterationCursor, empty_address_tracks_the_control) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        auto ctrl = builder.append_register(1, "ctrl");
        auto target = builder.append_register(1, "target");
        auto clean = builder.append_register(1, "@clean");
        stride_span<const QubitId> no_address(nullptr, 1, 0);
        UnaryIterationCursor cursor(builder, CircuitGenCtx{.clean_workspace = clean}, no_address, ctrl[0]);
        cursor.move_to(0);
        builder.cx(cursor.match_qubit(), target[0]);
        cursor.close();
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        sample["target"].bit_ref(0) ^= (bool)sample["ctrl"];
    });

    fuzzer.fuzz(1, 64);
}
