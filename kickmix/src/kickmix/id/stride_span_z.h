#ifndef KICKGEN_QC_STRIDE_SPAN_H
#define KICKGEN_QC_STRIDE_SPAN_H

#include <memory>

#include "kickmix/mem/stride_span.h"
#include "qubit_or_bit.h"
#include "qubit_or_bit_or_bool.h"

namespace kickmix {

/// A stride_span<QubitOrBitOrBool> with extra features.
///
/// Has a common_type field for indicating if the data is known to all
/// be of one specific type, and utility methods for converting to/from
/// more specific types of stride_span.
struct stride_span_z {
    /// Base pointer to qubit/bit/etc data.
    const QubitOrBitOrBool *ptr;

    /// Indicates how far to step when going from item to item in the collection.
    int64_t stride;
    /// Indicates how many items are in this collection.
    size_t count;

    /// If this is QXZTypeTag8::MIXED, each individual item may have a different QCType.
    /// If it is a specific type, like QXZTypeTag8::QUBIT_ID, then all items have that type.
    QXZTypeTag8 common_type;

    inline stride_span_z() : ptr(nullptr), stride(0), count(0), common_type(QXZTypeTag8::BOOL_VAL) {
    }
    inline stride_span_z(const QubitOrBitOrBool *items, int64_t stride, size_t len, QXZTypeTag8 common_type)
        : ptr(items), stride(stride), count(len), common_type(common_type) {
    }

    inline stride_span_z(const std::vector<QubitId> &v)
        : ptr((const QubitOrBitOrBool *)v.data()), stride(1), count(v.size()), common_type(QXZTypeTag8::QUBIT_ID) {
    }
    inline stride_span_z(const std::vector<BitId> &v)
        : ptr((const QubitOrBitOrBool *)v.data()), stride(1), count(v.size()), common_type(QXZTypeTag8::BIT_ID) {
    }
    inline stride_span_z(const std::vector<QubitOrBitOrBool> &v)
        : ptr((const QubitOrBitOrBool *)v.data()),
          stride(1),
          count(v.size()),
          common_type(QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE) {
    }
    inline stride_span_z(std::span<const QubitId> v)
        : ptr((const QubitOrBitOrBool *)v.data()), stride(1), count(v.size()), common_type(QXZTypeTag8::QUBIT_ID) {
    }
    inline stride_span_z(std::span<const BitId> v)
        : ptr((const QubitOrBitOrBool *)v.data()), stride(1), count(v.size()), common_type(QXZTypeTag8::BIT_ID) {
    }
    inline stride_span_z(std::span<const QubitOrBitOrBool> v)
        : ptr((const QubitOrBitOrBool *)v.data()),
          stride(1),
          count(v.size()),
          common_type(QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE) {
    }
    inline stride_span_z(std::span<const QubitOrBit> v)
        : ptr((const QubitOrBitOrBool *)v.data()),
          stride(1),
          count(v.size()),
          common_type(QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE) {
    }
    inline stride_span_z(stride_span<const QubitId> v)
        : ptr((const QubitOrBitOrBool *)v.ptr), stride(v.stride), count(v.count), common_type(QXZTypeTag8::QUBIT_ID) {
    }
    inline stride_span_z(stride_span<const BitId> v)
        : ptr((const QubitOrBitOrBool *)v.ptr), stride(v.stride), count(v.count), common_type(QXZTypeTag8::BIT_ID) {
    }
    inline stride_span_z(stride_span<const QubitOrBitOrBool> v)
        : ptr((const QubitOrBitOrBool *)v.ptr),
          stride(v.stride),
          count(v.size()),
          common_type(QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE) {
    }

    static stride_span_z repeat_false(size_t n);
    inline stride_span_iter<const QubitOrBitOrBool> begin() const {
        return {ptr, stride, count};
    }
    inline stride_span_iter_end end() const {
        return {};
    }
    inline QubitOrBitOrBool front() const {
        return (*this)[0];
    }
    inline QubitOrBitOrBool back() const {
        return (*this)[size() - 1];
    }
    inline size_t size() const {
        return count;
    }
    inline bool empty() const {
        return count == 0;
    }
    inline std::vector<QubitOrBitOrBool> to_vector() const {
        std::vector<QubitOrBitOrBool> result;
        for (size_t k = 0; k < size(); k++) {
            result.push_back((*this)[k]);
        }
        return result;
    }
    const QubitOrBitOrBool *data() const {
        return ptr;
    }

    inline QubitOrBitOrBool operator[](size_t index) const {
        return ptr[(int64_t)index * stride];
    }

    bool is_all_qubits() const;
    bool is_all_classical() const;
    bool is_all_bits() const;
    stride_span<const QubitId> checked_cast_to_qubit_ids(const char *value_name_for_error) const;
    stride_span<const BitId> checked_cast_to_bit_ids(const char *value_name_for_error) const;

    std::string str() const;

    stride_span_z skip(size_t skip_count) const {
        skip_count = std::min(skip_count, size());
        return stride_span_z(ptr + skip_count * stride, stride, count - skip_count, common_type);
    }
    stride_span_z keep(size_t keep_count) const {
        keep_count = std::min(keep_count, size());
        return stride_span_z(ptr, stride, keep_count, common_type);
    }
    stride_span_z skip_last(size_t skip_count) const {
        return keep(size() - std::min(skip_count, size()));
    }
    stride_span_z keep_last(size_t keep_count) const {
        return skip(size() - std::min(keep_count, size()));
    }

    template <typename T>
        requires(!std::same_as<T, bool>)
    inline stride_span<const T> cast_data() const {
        return {(const T *)ptr, stride, count};
    }

    explicit inline operator stride_span<const bool>() const {
        if constexpr (std::endian::native == std::endian::big) {
            return {(bool *)ptr + (int)sizeof(uint32_t) - 1, stride * (int)sizeof(uint32_t), count};
        } else {
            return {(bool *)ptr, stride * (int)sizeof(uint32_t), count};
        }
    }
    explicit inline operator stride_ptr<const bool>() const {
        return (stride_ptr<const bool>)(stride_span<const bool>)*this;
    }

    void recompute_common_type_data();
};
std::ostream &operator<<(std::ostream &out, const stride_span_z &v);

}  // namespace kickmix

#endif
