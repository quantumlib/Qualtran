#include "kickmix/gen/gf_arithmetic/gen_gf_imul_x.h"

#include <numeric>
#include <vector>

#include "kickmix/build/circuit_gen_ctx.h"
#include "kickmix/gen/gf_arithmetic/gen_linear_map.h"

using namespace kickmix;

/// Cyclically moves the value at position j to position (j + amount) % m, using m - gcd(m, amount)
/// swaps instead of the amount * (m - 1) swaps that repeated single step rotation would cost.
static void cyclic_rotate(CircuitBuilder &builder, stride_span<const QubitId> reg, size_t amount) {
    size_t m = reg.size();
    if (m < 2) {
        return;
    }
    amount %= m;
    if (amount == 0) {
        return;
    }
    size_t num_cycles = std::gcd(m, amount);
    for (size_t start = 0; start < num_cycles; start++) {
        // Carry the value that begins at `start` around its cycle. Each swap deposits the carried
        // value at its destination and picks up the value that was there.
        for (size_t j = (start + amount) % m; j != start; j = (j + amount) % m) {
            builder.swap(reg[start], reg[j]);
        }
    }
}

/// Returns the exponents of the reduction polynomial's terms below x^m, excluding the constant term.
static std::vector<size_t> reduction_terms(const GF2Field &field) {
    std::vector<size_t> result;
    for (size_t e = 1; e < field.degree(); e++) {
        if (field.modulus().bit(e)) {
            result.push_back(e);
        }
    }
    return result;
}

/// True when synthesizing the whole multiplication as one linear map beats stepping k times.
static bool prefer_linear_map(size_t m, size_t k, size_t num_terms) {
    // Stepping costs about k * num_terms CNOTs; a dense linear map costs about m^2 CNOTs.
    // Dividing rather than multiplying avoids size_t overflow when k is very large.
    return k > (m * m) / num_terms;
}

void kickmix::gen_gf_imul_x(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    size_t k) {
    size_t m = field.degree();
    throw_unless(Q_target.size() == m, "gen_gf_imul_x: Q_target.size() != field.degree()");
    throw_unless(field.modulus().bit(0), "gen_gf_imul_x: the field polynomial has a zero constant term");
    if (m <= 1 || k == 0) {
        // In GF(2) the only non-zero element is 1, so multiplying by x (which is 1) does nothing.
        return;
    }

    std::vector<size_t> terms = reduction_terms(field);
    if (prefer_linear_map(m, k, terms.size() + 1)) {
        gen_linear_map(builder, ctx, Q_target, field.constant_mul_matrix(field.pow(field.x(), k)));
        return;
    }

    auto mark = builder.raii_mark_block_entry("gf_imul_x");

    // Track the rotation lazily. After s steps the coefficient the textbook algorithm would hold at
    // logical position i physically lives at position (i - s) mod m.
    for (size_t s = 1; s <= k; s++) {
        size_t zero_pos = (m - (s % m)) % m;
        // The overflowed coefficient now sits at logical position 0, and adding the reduction
        // polynomial means XORing it into every other term of the polynomial. The constant term is
        // already accounted for, because the overflow bit is sitting in that position.
        for (size_t e : terms) {
            builder.cx(Q_target[zero_pos], Q_target[(e + zero_pos) % m]);
        }
    }

    cyclic_rotate(builder, Q_target, k % m);
}

void kickmix::gen_gf_idiv_x(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    size_t k) {
    size_t m = field.degree();
    throw_unless(Q_target.size() == m, "gen_gf_idiv_x: Q_target.size() != field.degree()");
    throw_unless(field.modulus().bit(0), "gen_gf_idiv_x: the field polynomial has a zero constant term");
    if (m <= 1 || k == 0) {
        return;
    }

    std::vector<size_t> terms = reduction_terms(field);
    if (prefer_linear_map(m, k, terms.size() + 1)) {
        gen_linear_map(builder, ctx, Q_target, field.constant_mul_matrix(field.invert(field.pow(field.x(), k))));
        return;
    }

    auto mark = builder.raii_mark_block_entry("gf_idiv_x");

    // The mirror image of gen_gf_imul_x: the reduction is applied before each rotation rather than
    // after, and the deferred rotation runs the other way.
    for (size_t s = 0; s < k; s++) {
        size_t zero_pos = s % m;
        for (size_t e : terms) {
            builder.cx(Q_target[zero_pos], Q_target[(e + zero_pos) % m]);
        }
    }

    cyclic_rotate(builder, Q_target, m - (k % m));
}
