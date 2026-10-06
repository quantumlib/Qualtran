#ifndef _KICKMIX_UTIL_GF2_MATRIX_H
#define _KICKMIX_UTIL_GF2_MATRIX_H

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "kickmix/util/gf2_poly.h"

namespace kickmix {

/// A dense matrix over GF(2), stored with bit packed rows.
///
/// Row r occupies data[r * row_words ... r * row_words + row_words), with column c stored in bit
/// (c % 64) of word (c / 64). Packing the bits means a row operation costs ceil(cols / 64) word
/// operations instead of `cols` byte operations, which is what makes Gaussian elimination on the
/// 512x512 matrices used by GF(2^512) circuit synthesis practical (roughly a 64x speedup over a
/// byte-per-entry representation).
struct GF2Matrix {
    size_t rows = 0;
    size_t cols = 0;
    /// Number of 64 bit words per row, i.e. ceil(cols / 64).
    size_t row_words = 0;
    std::vector<uint64_t> data;

    GF2Matrix() = default;
    /// Creates a zero filled matrix.
    GF2Matrix(size_t rows, size_t cols);

    /// Returns the identity matrix of the given size.
    static GF2Matrix identity(size_t n);

    /// Returns a pointer to the packed words of a row.
    inline uint64_t *row(size_t r) {
        return data.data() + r * row_words;
    }
    inline const uint64_t *row(size_t r) const {
        return data.data() + r * row_words;
    }

    /// Returns the entry at the given row and column.
    inline bool get(size_t r, size_t c) const {
        return (data[r * row_words + (c / 64)] >> (c & 63)) & 1;
    }
    /// Sets the entry at the given row and column.
    inline void set(size_t r, size_t c, bool value) {
        uint64_t mask = uint64_t{1} << (c & 63);
        uint64_t &word = data[r * row_words + (c / 64)];
        word = value ? (word | mask) : (word & ~mask);
    }
    /// Toggles the entry at the given row and column.
    inline void xor_bit(size_t r, size_t c) {
        data[r * row_words + (c / 64)] ^= uint64_t{1} << (c & 63);
    }

    /// Performs row[dst] ^= row[src].
    void xor_row_into(size_t src, size_t dst);
    /// Exchanges two rows.
    void swap_rows(size_t r1, size_t r2);
    /// Returns true if the row contains no set entries.
    bool is_row_zero(size_t r) const;

    bool operator==(const GF2Matrix &other) const;
    inline bool operator!=(const GF2Matrix &other) const {
        return !(*this == other);
    }

    /// Returns the matrix product over GF(2).
    GF2Matrix operator*(const GF2Matrix &rhs) const;

    /// Applies the matrix to a column vector whose entries are the low `cols` bits of `v`.
    ///
    /// Bit i of the result is the parity of row i anded with v. This is the operation that maps a
    /// GF(2^m) field element through a linear transformation such as squaring.
    GF2Poly operator*(const GF2Poly &v) const;

    /// Returns the transpose of the matrix.
    GF2Matrix transposed() const;

    /// Decomposes this matrix as `*this == P * L * U`.
    ///
    /// P is a permutation matrix, L is lower triangular with ones on the diagonal, and U is upper
    /// triangular with ones on the diagonal.
    ///
    /// Returns false (leaving the outputs unspecified) if the matrix is not square or is singular.
    bool plu_decompose(GF2Matrix &out_p, GF2Matrix &out_l, GF2Matrix &out_u) const;

    /// Returns the inverse of the matrix, or an empty matrix if it is not square and invertible.
    GF2Matrix inverse() const;

    /// Returns the rank of the matrix.
    size_t rank() const;

    /// Returns a multi-line string with one character per entry.
    std::string str() const;
};

std::ostream &operator<<(std::ostream &out, const GF2Matrix &value);

}  // namespace kickmix

#endif
