#ifndef KICKGEN_QCARRAY_XZ_H
#define KICKGEN_QCARRAY_XZ_H

#include "array_z.h"
#include "bit_or_false.h"
#include "kickmix/mem/stride_span.h"
#include "kickmix/util/fixed_width_int.h"
#include "qubit_or_bit_or_bool.h"
#include "stride_span_xz.h"
#include "stride_span_z.h"

namespace kickmix {
struct QubitOrFalse;
struct QubitOrBit;

/// A collection of qubit/bit/xbit/bool/xbool data.
struct array_xz {
    /// Pointer to the XZ data.
    /// (Note: this class allocates and is responsible for freeing the data.)
    QubitOrXZBitOrXZBool *items{};
    /// Number of items in collections.
    size_t count{};
    /// Summary information about the type of the contents.
    QXZTypeTag8 common_type{};

    array_xz();
    array_xz(array_xz &&) noexcept;
    array_xz(array_z &&) noexcept;
    array_xz &operator=(array_xz &&) noexcept;
    array_xz(const array_xz &) = delete;
    array_xz &operator=(const array_xz &) = delete;
    ~array_xz();
    void recompute_common_type();

    static array_xz alloc_noinit(size_t count, QXZTypeTag8 tag = QXZTypeTag8::MAY_BE_MIXED);

    template <typename T>
    static array_xz copy_of(const T &items) {
        array_xz result = alloc_noinit(items.size(), QXZTypeTag8::MAY_BE_MIXED);
        QubitOrXZBitOrXZBool *out = result.items;
        for (const auto &e : items) {
            *out++ = e;
        }
        result.recompute_common_type();
        return result;
    }
    static array_xz copy_of_concat(const stride_span_xz &lhs, const stride_span_xz &rhs);

    inline QubitOrXZBitOrXZBool &operator[](size_t index) {
        return items[index];
    }
    inline stride_span_xz as_ptr() const {
        return stride_span_xz(items, 1, count, common_type);
    }
    inline operator stride_span_xz() const {
        return as_ptr();
    }
    bool operator==(const array_xz &other) const;
    std::string str() const;
};
std::ostream &operator<<(std::ostream &out, const array_xz &v);

}  // namespace kickmix

#endif
