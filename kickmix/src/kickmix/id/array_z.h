#ifndef KICKGEN_QCARRAY_H
#define KICKGEN_QCARRAY_H

#include <memory>

#include "bit_or_false.h"
#include "kickmix/mem/stride_span.h"
#include "kickmix/util/fixed_width_int.h"
#include "qubit_or_bit.h"
#include "qubit_or_bit_or_bool.h"
#include "qubit_or_false.h"
#include "stride_span_z.h"

namespace kickmix {

/// A collection of data that can be used as a Z observable.
struct array_z {
    /// Pointer to the Z data.
    /// (Note: this class allocates and is responsible for freeing the data.)
    QubitOrBitOrBool *items{};
    /// Indicates how many items are in this collection.
    size_t count{};

    /// If this is QXZTypeTag8::MIXED, each individual item may have a different QCType.
    /// If it is a specific type, like QXZTypeTag8::QUBIT_ID, then all items have that type.
    QXZTypeTag8 common_type{};

    array_z();
    array_z(array_z &&) noexcept;
    array_z &operator=(array_z &&) noexcept;
    array_z(const array_z &) = delete;
    array_z &operator=(const array_z &) = delete;
    ~array_z();

    static array_z alloc_noinit(size_t count, QXZTypeTag8 tag = QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE);
    template <typename T>
    static array_z copy_of(const T &items) {
        array_z result = alloc_noinit(items.size(), QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE);
        QubitOrBitOrBool *out = result.items;
        for (const auto &e : items) {
            *out++ = e;
        }
        result.recompute_common_type();
        return result;
    }
    static array_z copy_of_concat(const stride_span_z &lhs, const stride_span_z &rhs);

    void recompute_common_type();
    inline stride_span_iter<const QubitOrBitOrBool> begin() const {
        return {items, 1, count};
    }
    inline stride_span_iter_end end() const {
        return {};
    }
    inline QubitOrBitOrBool &front() {
        return (*this)[0];
    }
    inline QubitOrBitOrBool &back() {
        return (*this)[size() - 1];
    }
    inline bool empty() const {
        return count == 0;
    }
    inline size_t size() const {
        return count;
    }
    inline stride_span_z as_ptr() const {
        return stride_span_z(items, 1, count, common_type);
    }
    inline QubitOrBitOrBool &operator[](size_t index) {
        return items[index];
    }
    inline operator stride_span_z() const {
        return as_ptr();
    }
    bool operator==(const array_z &other) const;
    std::string str() const;
};
std::ostream &operator<<(std::ostream &out, const array_z &v);

}  // namespace kickmix

#endif
