#ifndef KICKGEN_QC_STRIDE_SPAN_XZ_H
#define KICKGEN_QC_STRIDE_SPAN_XZ_H

#include <memory>

#include "kickmix/mem/stride_span.h"
#include "qubit_or_bit_or_bool.h"
#include "qubit_or_xbit_or_xbool.h"
#include "qubit_or_xzbit_or_xzbool.h"
#include "stride_span_x.h"
#include "stride_span_z.h"

namespace kickmix {

/// A stride_span<QubitOrXZBitOrXZBool> with extra features.
///
/// Has a common_type field for indicating if the data is known to all
/// be of one specific type, and utility methods for converting to/from
/// more specific types of stride_span.
struct stride_span_xz {
    /// Base pointer to qubit/bit/etc data.
    const QubitOrXZBitOrXZBool *ptr;
    /// Indicates how far to step when going from item to item in the collection.
    int64_t stride;
    /// Indicates how many items are in this collection.
    size_t count;

    /// Type hint information about the content of the array.
    /// Note that this
    ///    BOOL_VAL: All values are bools.
    ///    BIT_ID: All values are BitIds.
    ///    QUBIT_ID: All values are QubitIds.
    ///    XBOOL_VAL: All values are XBools.
    ///    XBIT_ID: All values are XBitIds.
    ///    MAY_BE_MIXED_BUT_IS_ALL_X: All values are QubitIds, XBitIds, or XBools (and may be a single more specific
    ///    thing). MAY_BE_MIXED_BUT_IS_ALL_Z: All values are QubitIds, BitIds, or bools (and may be a single more
    ///    specific thing). MAY_BE_MIXED: Vacuous. All values are possible, and all values may be a single more specific
    ///    type of thing.
    QXZTypeTag8 common_type;

    inline stride_span_xz() : ptr(nullptr), stride(0), count(0), common_type(QXZTypeTag8::QUBIT_ID) {
    }
    inline stride_span_xz(const QubitOrXZBitOrXZBool *ptr, int64_t stride, size_t count, QXZTypeTag8 common_type)
        : ptr(ptr), stride(stride), count(count), common_type(common_type) {
    }

    bool has_same_contents_as(const stride_span_xz &other) const;

    inline stride_span_xz(const stride_span_z &v)
        : ptr((const QubitOrXZBitOrXZBool *)v.ptr), stride(v.stride), count(v.count), common_type(v.common_type) {
    }
    inline stride_span_xz(const stride_span_x &v)
        : ptr((const QubitOrXZBitOrXZBool *)v.ptr), stride(v.stride), count(v.count), common_type(v.common_type) {
    }
    inline stride_span_xz(stride_span<const QubitId> v)
        : ptr((const QubitOrXZBitOrXZBool *)v.data()),
          stride(v.stride),
          count(v.size()),
          common_type(QXZTypeTag8::QUBIT_ID) {
    }
    inline stride_span_xz(stride_span<const BitId> v)
        : ptr((const QubitOrXZBitOrXZBool *)v.data()),
          stride(v.stride),
          count(v.size()),
          common_type(QXZTypeTag8::BIT_ID) {
    }
    inline stride_span_xz(stride_span<const XBitId> v)
        : ptr((const QubitOrXZBitOrXZBool *)v.data()),
          stride(v.stride),
          count(v.size()),
          common_type(QXZTypeTag8::XBIT_ID) {
    }
    inline stride_span_xz(stride_span<const XBool> v)
        : ptr((const QubitOrXZBitOrXZBool *)v.data()),
          stride(v.stride),
          count(v.size()),
          common_type(QXZTypeTag8::XBOOL_VAL) {
    }
    inline stride_span_xz(stride_span<const QubitOrBit> v)
        : ptr((const QubitOrXZBitOrXZBool *)v.data()),
          stride(v.stride),
          count(v.size()),
          common_type(QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE) {
    }
    inline stride_span_xz(stride_span<const QubitOrBitOrBool> v)
        : ptr((const QubitOrXZBitOrXZBool *)v.data()),
          stride(v.stride),
          count(v.size()),
          common_type(QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE) {
    }
    inline stride_span_xz(stride_span<const QubitOrXBitOrXBool> v)
        : ptr((const QubitOrXZBitOrXZBool *)v.data()),
          stride(v.stride),
          count(v.size()),
          common_type(QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_XLIKE) {
    }
    inline stride_span_xz(stride_span<const QubitOrXZBitOrXZBool> v)
        : ptr(v.data()), stride(v.stride), count(v.size()), common_type(QXZTypeTag8::MAY_BE_MIXED) {
    }

    static stride_span_xz repeat_false(size_t n);
    inline stride_span_iter<const QubitOrXZBitOrXZBool> begin() const {
        return {(const QubitOrXZBitOrXZBool *)ptr, stride, count};
    }
    inline stride_span_iter_end end() const {
        return {};
    }
    inline QubitOrXZBitOrXZBool front() const {
        return (*this)[0];
    }
    inline QubitOrXZBitOrXZBool back() const {
        return (*this)[size() - 1];
    }
    inline size_t size() const {
        return count;
    }
    inline bool empty() const {
        return count == 0;
    }

    inline QubitOrXZBitOrXZBool operator[](size_t index) const {
        int64_t k = (int64_t)index;
        k *= stride;
        return ((QubitOrXZBitOrXZBool *)ptr)[k];
    }

    bool is_all_qubit_ids() const;
    bool is_all_bit_ids() const;
    bool is_all_xbit_ids() const;
    bool is_all_xbools() const;
    bool is_all_bools() const;
    bool is_all_xlike() const;
    bool is_all_zlike() const;
    bool is_all_qubit_or_bit() const;
    bool is_all_classical_and_xlike() const;
    bool is_all_classical_and_zlike() const;

    stride_span<const QubitId> checked_cast_to_qubit_ids(const char *value_name_for_error) const;
    stride_span<const QubitOrBit> checked_cast_to_qubit_or_bit(const char *value_name_for_error) const;
    stride_span<const BitId> checked_cast_to_bit_ids(const char *value_name_for_error) const;
    stride_span<const XBitId> checked_cast_to_xbit_ids(const char *value_name_for_error) const;
    stride_span_z checked_cast_to_qubit_or_bit_or_bool(const char *value_name_for_error) const;
    stride_span<const QubitOrXBitOrXBool> checked_cast_to_qubit_or_xbit_or_xbool(
        const char *value_name_for_error) const;

    std::string str() const;

    stride_span_xz reversed() const {
        return stride_span_xz(ptr + stride * ((int64_t)count - 1), -stride, count, common_type);
    }
    stride_span_xz skip(size_t skip_count) const {
        skip_count = std::min(skip_count, size());
        return stride_span_xz(ptr + stride * (int64_t)skip_count, stride, count - skip_count, common_type);
    }
    stride_span_xz keep(size_t keep_count) const {
        keep_count = std::min(keep_count, size());
        return stride_span_xz(ptr, stride, keep_count, common_type);
    }
    stride_span_xz skip_last(size_t skip_count) const {
        return keep(size() - std::min(skip_count, size()));
    }
    stride_span_xz keep_last(size_t keep_count) const {
        return skip(size() - std::min(keep_count, size()));
    }

    inline operator stride_span<const QubitOrXZBitOrXZBool>() const {
        return {(QubitOrXZBitOrXZBool *)ptr, stride, count};
    }
    inline operator stride_ptr<const QubitOrXZBitOrXZBool>() const {
        return {(QubitOrXZBitOrXZBool *)ptr, stride};
    }

    template <typename T>
    stride_span<const T> cast_data() const {
        return {(const T *)ptr, stride, count};
    }

    QXZTypeTag8 compute_common_type() const;
};

template <>
inline stride_span<const bool> stride_span_xz::cast_data() const {
    if constexpr (std::endian::native == std::endian::big) {
        return {(const bool *)(ptr) + (int)sizeof(uint32_t) - 1, stride * (int)sizeof(uint32_t), count};
    } else {
        return {(const bool *)ptr, stride * (int)sizeof(uint32_t), count};
    }
}
std::ostream &operator<<(std::ostream &out, const stride_span_xz &v);

}  // namespace kickmix

#endif
