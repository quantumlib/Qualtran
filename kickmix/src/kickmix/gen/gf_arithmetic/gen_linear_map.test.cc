#include "gen_linear_map.h"

#include "gtest/gtest.h"

#include "kickmix/sim/fuzzer.h"
#include "test.test.h"

using namespace kickmix;

static GF2Matrix random_matrix(std::mt19937_64 &rng, size_t n) {
    GF2Matrix result(n, n);
    for (size_t r = 0; r < n; r++) {
        for (size_t c = 0; c < n; c++) {
            result.set(r, c, rng() & 1);
        }
    }
    return result;
}

static GF2Matrix random_invertible_matrix(std::mt19937_64 &rng, size_t n) {
    while (true) {
        GF2Matrix m = random_matrix(rng, n);
        if (m.rank() == n) {
            return m;
        }
    }
}

/// Writes the low bits of a polynomial into a fuzzer register, leaving its width alone.
static void write_poly(FixedWidthInt &target, const GF2Poly &value) {
    for (size_t k = 0; k < target.num_bits; k++) {
        target.bit_ref(k) = value.bit(k);
    }
}

TEST(gen_linear_map, fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    auto matrix = std::make_shared<GF2Matrix>();

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = 1 + rng() % 8;
        *matrix = random_invertible_matrix(rng, n);
        auto target = builder.append_register(n, "target");
        gen_linear_map(builder, CircuitGenCtx{}, target, *matrix);
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        FixedWidthInt &target = sample["target"];
        write_poly(target, (*matrix) * GF2Poly::from_fixed_width_int(target));
    });

    fuzzer.fuzz(50, 64);
}

TEST(gen_linear_map, adjoint_undoes_forward_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = 1 + rng() % 8;
        GF2Matrix matrix = random_invertible_matrix(rng, n);
        auto target = builder.append_register(n, "target");
        gen_linear_map(builder, CircuitGenCtx{}, target, matrix);
        gen_linear_map_adjoint(builder, CircuitGenCtx{}, target, matrix);
    });

    // No output sampler: the composition must leave the register exactly as it was found.
    fuzzer.fuzz(30, 64);
}

TEST(gen_linear_map, adjoint_applies_inverse_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    auto matrix = std::make_shared<GF2Matrix>();

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = 1 + rng() % 8;
        *matrix = random_invertible_matrix(rng, n);
        auto target = builder.append_register(n, "target");
        gen_linear_map_adjoint(builder, CircuitGenCtx{}, target, *matrix);
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        FixedWidthInt &target = sample["target"];
        write_poly(target, matrix->inverse() * GF2Poly::from_fixed_width_int(target));
    });

    fuzzer.fuzz(30, 64);
}

TEST(gen_linear_map, identity_emits_nothing) {
    CircuitBuilder builder;
    auto target = builder.append_register(8, "target");
    gen_linear_map(builder, CircuitGenCtx{}, target, GF2Matrix::identity(8));
    Circuit circuit = builder.finish_circuit();
    ASSERT_EQ(circuit.num_ops, 0);
}

TEST(gen_linear_map, permutation_is_swaps) {
    // The reversal permutation on 4 qubits needs 2 swaps.
    GF2Matrix matrix(4, 4);
    for (size_t k = 0; k < 4; k++) {
        matrix.set(k, 3 - k, true);
    }

    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        auto target = builder.append_register(4, "target");
        gen_linear_map(builder, CircuitGenCtx{}, target, matrix);
    });
    fuzzer.use_output_sampler([&](OutputSample &sample) {
        FixedWidthInt &target = sample["target"];
        write_poly(target, matrix * GF2Poly::from_fixed_width_int(target));
    });
    fuzzer.fuzz(1, 64);
}

TEST(gen_linear_map, single_qubit) {
    CircuitBuilder builder;
    auto target = builder.append_register(1, "target");
    // The only invertible 1x1 matrix is the identity.
    gen_linear_map(builder, CircuitGenCtx{}, target, GF2Matrix::identity(1));
    ASSERT_EQ(builder.finish_circuit().num_ops, 0);
}

TEST(gen_linear_map, empty_register) {
    CircuitBuilder builder;
    std::vector<QubitId> target;
    gen_linear_map(builder, CircuitGenCtx{}, target, GF2Matrix(0, 0));
    ASSERT_EQ(builder.finish_circuit().num_ops, 0);
}

TEST(gen_linear_map, rejects_bad_input) {
    CircuitBuilder builder;
    auto target = builder.append_register(4, "target");

    // Not square.
    ASSERT_THROW({ gen_linear_map(builder, CircuitGenCtx{}, target, GF2Matrix(4, 3)); }, std::invalid_argument);
    // Size mismatch with the register.
    ASSERT_THROW({ gen_linear_map(builder, CircuitGenCtx{}, target, GF2Matrix::identity(5)); }, std::invalid_argument);
    // Singular, so the map is not reversible.
    ASSERT_THROW({ gen_linear_map(builder, CircuitGenCtx{}, target, GF2Matrix(4, 4)); }, std::invalid_argument);
    ASSERT_THROW({ gen_linear_map_adjoint(builder, CircuitGenCtx{}, target, GF2Matrix(4, 4)); }, std::invalid_argument);
}

TEST(gen_linear_map, large_matrix_gate_count) {
    // A dense 64x64 map must stay within the two-triangular-factors bound.
    auto rng = INDEPENDENT_TEST_RNG();
    size_t n = 64;
    GF2Matrix matrix = random_invertible_matrix(rng, n);

    CircuitBuilder builder;
    auto target = builder.append_register(n, "target");
    gen_linear_map(builder, CircuitGenCtx{}, target, matrix);
    Circuit circuit = builder.finish_circuit();

    // Each triangular factor contributes at most n*(n-1)/2 CNOTs, plus at most n/2 swaps.
    ASSERT_LE(circuit.num_ops, n * (n - 1) + n / 2);
    ASSERT_GT(circuit.num_ops, 0);
}
