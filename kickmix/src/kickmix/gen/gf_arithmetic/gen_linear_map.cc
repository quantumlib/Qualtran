#include "kickmix/gen/gf_arithmetic/gen_linear_map.h"

#include <bit>
#include <numeric>
#include <vector>

#include "kickmix/build/circuit_gen_ctx.h"

using namespace kickmix;

/// Calls `callback(j)` for each set entry of `row` with lo <= j < hi, cheaply skipping zero runs.
template <typename F>
static inline void for_each_set_bit(const uint64_t *row, size_t lo, size_t hi, F &&callback) {
    if (lo >= hi) {
        return;
    }
    size_t first_word = lo / 64;
    size_t last_word = (hi - 1) / 64;
    for (size_t w = first_word; w <= last_word; w++) {
        uint64_t bits = row[w];
        if (w == first_word) {
            bits &= ~uint64_t{0} << (lo & 63);
        }
        if (w == last_word && (hi & 63) != 0) {
            bits &= (uint64_t{1} << (hi & 63)) - 1;
        }
        while (bits != 0) {
            callback(w * 64 + std::countr_zero(bits));
            bits &= bits - 1;
        }
    }
}

void kickmix::gen_linear_map(
    CircuitBuilder &builder, const CircuitGenCtx &ctx, stride_span<const QubitId> Q_target, const GF2Matrix &matrix) {
    (void)ctx;
    throw_unless(matrix.rows == matrix.cols, "gen_linear_map: matrix is not square");
    throw_unless(matrix.rows == Q_target.size(), "gen_linear_map: matrix.rows != Q_target.size()");
    size_t n = matrix.rows;
    if (n == 0) {
        return;
    }

    GF2Matrix p;
    GF2Matrix l;
    GF2Matrix u;
    throw_unless(matrix.plu_decompose(p, l, u), "gen_linear_map: matrix is singular (the map is not reversible)");

    auto mark = builder.raii_mark_block_entry("linear_map");

    // Apply U, the unit upper triangular factor. Within one row every CNOT shares a target, so they
    // commute and may be emitted in any order, but the rows must be processed in increasing order.
    for (size_t i = 0; i < n; i++) {
        const uint64_t *row = u.row(i);
        for_each_set_bit(row, i + 1, n, [&](size_t j) {
            builder.cx(Q_target[j], Q_target[i]);
        });
    }

    // Apply L, the unit lower triangular factor, in decreasing row order.
    for (size_t i = n; i--;) {
        const uint64_t *row = l.row(i);
        for_each_set_bit(row, 0, i, [&](size_t j) {
            builder.cx(Q_target[j], Q_target[i]);
        });
    }

    // Apply P, the permutation factor, as a sequence of transpositions.
    std::vector<size_t> column(n);
    std::iota(column.begin(), column.end(), size_t{0});
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            if (p.get(i, column[j])) {
                builder.swap(Q_target[i], Q_target[j]);
                std::swap(column[i], column[j]);
                break;
            }
        }
    }
}

void kickmix::gen_linear_map_adjoint(
    CircuitBuilder &builder, const CircuitGenCtx &ctx, stride_span<const QubitId> Q_target, const GF2Matrix &matrix) {
    throw_unless(matrix.rows == matrix.cols, "gen_linear_map_adjoint: matrix is not square");
    GF2Matrix inverse = matrix.inverse();
    throw_unless(inverse.rows == matrix.rows, "gen_linear_map_adjoint: matrix is singular (the map is not reversible)");
    gen_linear_map(builder, ctx, Q_target, inverse);
}
