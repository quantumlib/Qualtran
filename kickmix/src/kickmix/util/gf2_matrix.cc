#include "kickmix/util/gf2_matrix.h"

#include <bit>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>

using namespace kickmix;

GF2Matrix::GF2Matrix(size_t rows, size_t cols)
    : rows(rows), cols(cols), row_words((cols + 63) / 64), data(rows * ((cols + 63) / 64), 0) {
}

GF2Matrix GF2Matrix::identity(size_t n) {
    GF2Matrix result(n, n);
    for (size_t k = 0; k < n; k++) {
        result.set(k, k, true);
    }
    return result;
}

void GF2Matrix::xor_row_into(size_t src, size_t dst) {
    const uint64_t *s = row(src);
    uint64_t *d = row(dst);
    for (size_t k = 0; k < row_words; k++) {
        d[k] ^= s[k];
    }
}

void GF2Matrix::swap_rows(size_t r1, size_t r2) {
    if (r1 == r2) {
        return;
    }
    uint64_t *a = row(r1);
    uint64_t *b = row(r2);
    for (size_t k = 0; k < row_words; k++) {
        std::swap(a[k], b[k]);
    }
}

bool GF2Matrix::is_row_zero(size_t r) const {
    const uint64_t *a = row(r);
    uint64_t acc = 0;
    for (size_t k = 0; k < row_words; k++) {
        acc |= a[k];
    }
    return acc == 0;
}

bool GF2Matrix::operator==(const GF2Matrix &other) const {
    return rows == other.rows && cols == other.cols && data == other.data;
}

GF2Matrix GF2Matrix::operator*(const GF2Matrix &rhs) const {
    if (cols != rhs.rows) {
        throw std::invalid_argument("GF2Matrix::operator*: lhs.cols != rhs.rows");
    }
    GF2Matrix result(rows, rhs.cols);
    for (size_t i = 0; i < rows; i++) {
        const uint64_t *lhs_row = row(i);
        uint64_t *out_row = result.row(i);
        for (size_t word = 0; word < row_words; word++) {
            uint64_t bits = lhs_row[word];
            // Iterate only over the set entries of the row, skipping runs of zeroes.
            while (bits != 0) {
                size_t k = word * 64 + std::countr_zero(bits);
                bits &= bits - 1;
                const uint64_t *add = rhs.row(k);
                for (size_t w = 0; w < rhs.row_words; w++) {
                    out_row[w] ^= add[w];
                }
            }
        }
    }
    return result;
}

GF2Poly GF2Matrix::operator*(const GF2Poly &v) const {
    if (rows > GF2Poly::MAX_BITS || cols > GF2Poly::MAX_BITS) {
        throw std::invalid_argument("GF2Matrix::operator*: matrix dimension exceeds GF2Poly::MAX_BITS");
    }
    GF2Poly result;
    for (size_t i = 0; i < rows; i++) {
        const uint64_t *r = row(i);
        uint64_t acc = 0;
        for (size_t w = 0; w < row_words; w++) {
            acc ^= r[w] & v.words[w];
        }
        if (std::popcount(acc) & 1) {
            result.set_bit(i, true);
        }
    }
    return result;
}

GF2Matrix GF2Matrix::transposed() const {
    GF2Matrix result(cols, rows);
    for (size_t i = 0; i < rows; i++) {
        const uint64_t *r = row(i);
        for (size_t word = 0; word < row_words; word++) {
            uint64_t bits = r[word];
            while (bits != 0) {
                size_t j = word * 64 + std::countr_zero(bits);
                bits &= bits - 1;
                result.set(j, i, true);
            }
        }
    }
    return result;
}

bool GF2Matrix::plu_decompose(GF2Matrix &out_p, GF2Matrix &out_l, GF2Matrix &out_u) const {
    if (rows != cols) {
        return false;
    }
    size_t n = rows;

    GF2Matrix e = *this;
    std::vector<size_t> pi(n);
    std::iota(pi.begin(), pi.end(), size_t{0});

    for (size_t i = 0; i < n; i++) {
        // Find a pivot in column i, at or below row i.
        size_t pivot = SIZE_MAX;
        for (size_t r = i; r < n; r++) {
            if (e.get(r, i)) {
                pivot = r;
                break;
            }
        }
        if (pivot == SIZE_MAX) {
            return false;  // Singular.
        }
        if (pivot != i) {
            e.swap_rows(i, pivot);
            std::swap(pi[i], pi[pivot]);
        }

        // Eliminate below the diagonal. Only columns strictly greater than i are updated, so that
        // column i keeps the multiplier that becomes the lower triangular factor.
        size_t pivot_word = i / 64;
        size_t pivot_offset = i & 63;
        uint64_t high_mask = pivot_offset == 63 ? 0 : ~((uint64_t{1} << (pivot_offset + 1)) - 1);
        const uint64_t *src = e.row(i);
        for (size_t r = i + 1; r < n; r++) {
            if (!e.get(r, i)) {
                continue;
            }
            uint64_t *dst = e.row(r);
            dst[pivot_word] ^= src[pivot_word] & high_mask;
            for (size_t w = pivot_word + 1; w < row_words; w++) {
                dst[w] ^= src[w];
            }
        }
    }

    out_l = GF2Matrix(n, n);
    out_u = GF2Matrix(n, n);
    out_p = GF2Matrix(n, n);
    for (size_t r = 0; r < n; r++) {
        out_l.set(r, r, true);
        for (size_t c = 0; c < r; c++) {
            out_l.set(r, c, e.get(r, c));
        }
        for (size_t c = r; c < n; c++) {
            out_u.set(r, c, e.get(r, c));
        }
    }
    for (size_t c = 0; c < n; c++) {
        out_p.set(pi[c], c, true);
    }
    return true;
}

GF2Matrix GF2Matrix::inverse() const {
    if (rows != cols) {
        return GF2Matrix();
    }
    size_t n = rows;

    // Gauss-Jordan elimination on the augmented matrix [this | I].
    GF2Matrix aug(n, 2 * n);
    for (size_t r = 0; r < n; r++) {
        for (size_t c = 0; c < n; c++) {
            if (get(r, c)) {
                aug.set(r, c, true);
            }
        }
        aug.set(r, n + r, true);
    }

    for (size_t i = 0; i < n; i++) {
        size_t pivot = SIZE_MAX;
        for (size_t r = i; r < n; r++) {
            if (aug.get(r, i)) {
                pivot = r;
                break;
            }
        }
        if (pivot == SIZE_MAX) {
            return GF2Matrix();  // Singular.
        }
        aug.swap_rows(i, pivot);
        for (size_t r = 0; r < n; r++) {
            if (r != i && aug.get(r, i)) {
                aug.xor_row_into(i, r);
            }
        }
    }

    GF2Matrix result(n, n);
    for (size_t r = 0; r < n; r++) {
        for (size_t c = 0; c < n; c++) {
            if (aug.get(r, n + c)) {
                result.set(r, c, true);
            }
        }
    }
    return result;
}

size_t GF2Matrix::rank() const {
    GF2Matrix e = *this;
    size_t result = 0;
    for (size_t c = 0; c < cols && result < rows; c++) {
        size_t pivot = SIZE_MAX;
        for (size_t r = result; r < rows; r++) {
            if (e.get(r, c)) {
                pivot = r;
                break;
            }
        }
        if (pivot == SIZE_MAX) {
            continue;
        }
        e.swap_rows(result, pivot);
        for (size_t r = 0; r < rows; r++) {
            if (r != result && e.get(r, c)) {
                e.xor_row_into(result, r);
            }
        }
        result++;
    }
    return result;
}

std::string GF2Matrix::str() const {
    std::stringstream ss;
    for (size_t r = 0; r < rows; r++) {
        if (r > 0) {
            ss << "\n";
        }
        for (size_t c = 0; c < cols; c++) {
            ss << (get(r, c) ? '1' : '.');
        }
    }
    return ss.str();
}

std::ostream &kickmix::operator<<(std::ostream &out, const GF2Matrix &value) {
    out << value.str();
    return out;
}
