#ifndef KICKMIX_MEM_STRIDE_SPAN_H
#define KICKMIX_MEM_STRIDE_SPAN_H

#include <algorithm>
#include <cstddef>
#include <ostream>
#include <span>
#include <stdexcept>
#include <vector>

#include "kickmix/build/iota.h"
#include "stride_ptr.h"

namespace kickmix {

/// A special type that represents the end of iterating over a stride span.
/// When stride_span_iter is compared (==) to this type, it instead just checks
/// if it has any items left to iterate.
struct stride_span_iter_end {};

/// Iterator class for iterating over a stride span.
/// Note: needs the `items_left` field to support the case where `stride == 0`.
template <typename T>
struct stride_span_iter {
    T *ptr;
    int64_t stride;
    size_t items_left;
    inline T &operator*() const {
        return *ptr;
    }
    inline stride_span_iter<T> &operator++() {
        ptr += stride;
        items_left--;
        return *this;
    }
    inline bool operator==(stride_span_iter_end) const {
        return items_left == 0;
    }
};

/// A view of evenly-spaced elements in memory.
///
/// Basically this is a std::span<T> but with support for strides other than 1.
template <typename T>
struct stride_span {
    T *ptr;
    int64_t stride;
    size_t count;

    stride_span() : ptr(nullptr), stride(0), count(0) {
    }
    stride_span(const stride_span<T> &other) = default;
    stride_span(stride_span<T> &&other) noexcept = default;
    stride_span<T> &operator=(stride_span<T> &&other) noexcept = default;
    stride_span<T> &operator=(const stride_span<T> &other) = default;

    template <typename T2>
    stride_span<T2> cast_data() const {
        return stride_span<T2>((T2 *)ptr, stride, count);
    }

    stride_span(T *ptr, int64_t stride, size_t count) : ptr(ptr), stride(stride), count(count) {
    }
    stride_span(std::span<T> s) : ptr(s.data()), stride(1), count(s.size()) {
    }
    stride_span(std::conditional_t<std::is_const_v<T>, const std::vector<std::remove_const_t<T>>, std::vector<T>> &s)
        : ptr(s.data()), stride(1), count(s.size()) {
    }

    inline T &operator[](size_t index) const {
        return *(ptr + stride * index);
    }
    inline operator stride_ptr<T>() const {
        return stride_ptr<T>{.ptr = ptr, .stride = stride};
    }
    inline operator stride_span<const T>() const {
        return stride_span<const T>(ptr, stride, count);
    }
    stride_ptr<const T> operator[](iota k) const {
        return stride_ptr<const T>(ptr + k.offset * stride, k.negated ? -stride : +stride);
    }

    // === Methods for making this class behave similar to standard library classes. ===
    inline T &at(size_t index) const {
        if (index >= count) {
            throw std::out_of_range("index out of range");
        }
        return *(ptr + stride * index);
    }
    inline T *data() const {
        return ptr;
    }
    inline bool empty() const {
        return count == 0;
    }
    inline T &back() {
        return *(ptr + stride * (count - 1));
    }
    inline T &front() {
        return *ptr;
    }
    inline const T &back() const {
        return *(ptr + stride * (count - 1));
    }
    inline const T &front() const {
        return *ptr;
    }
    inline size_t size() const {
        return count;
    }
    inline stride_span_iter<T> begin() {
        return stride_span_iter<T>{.ptr = ptr, .stride = stride, .items_left = size()};
    }
    inline stride_span_iter_end end() {
        return {};
    }
    inline stride_span_iter<const T> begin() const {
        return stride_span_iter<const T>{.ptr = ptr, .stride = stride, .items_left = size()};
    }
    inline stride_span_iter_end end() const {
        return {};
    }

    /// Returns a reversed view of the same items.
    stride_span<T> reversed() const {
        if (!count) {
            // Avoid moving the ptr when the span is empty (causes warnings when ptr is null).
            return stride_span(ptr, -stride, count);
        }
        return stride_span(ptr + stride * count - stride, -stride, count);
    }

    /// Returns a view of the same items, but with the given number skipped at the beginning.
    ///
    /// Args:
    ///     skip_count: The number of items to skip. If this is larger than the number
    ///         of items in the stride span, the returned stride_span will be empty.
    inline stride_span<T> skip(size_t skip_count) const {
        skip_count = std::min(skip_count, count);
        return stride_span<T>(ptr + skip_count * stride, stride, count - skip_count);
    }

    /// Returns a view of the same items, but keeping only the given number of items at the beginning.
    ///
    /// Args:
    ///     keep_count: The number of items to keep. If this is larger than the number
    ///         of items in the stride span, the output stride span is identical to the
    ///         input.
    inline stride_span<T> keep(size_t keep_count) const {
        keep_count = std::min(keep_count, count);
        return stride_span<T>(ptr, stride, keep_count);
    }

    /// Returns a view of the same items, but with the given number skipped at the end.
    ///
    /// Args:
    ///     skip_count: The number of items to skip. If this is larger than the number
    ///         of items in the stride span, the returned stride_span will be empty.
    inline stride_span<T> skip_last(size_t skip_count) const {
        skip_count = std::min(skip_count, count);
        return stride_span<T>(ptr, stride, count - skip_count);
    }

    /// Returns a view of the same items, but keeping only the given number of items at the end.
    ///
    /// Args:
    ///     keep_count: The number of items to keep. If this is larger than the number
    ///         of items in the stride span, the output stride span is identical to the
    ///         input.
    inline stride_span<T> keep_last(size_t keep_count) const {
        keep_count = std::min(keep_count, count);
        return stride_span<T>(ptr + (count - keep_count) * stride, stride, keep_count);
    }

    inline stride_span<T> subspan(size_t skip_count) const {
        return skip(skip_count);
    }

    inline stride_span<T> subspan(size_t skip_count, size_t keep_count) const {
        return skip(skip_count).keep(keep_count);
    }
};
template <typename T>
std::ostream &operator<<(std::ostream &out, const stride_span<T> &v) {
    out << "stride_span{";
    out << ".ptr=" << (uintptr_t)v.ptr;
    out << ", .stride=" << v.stride;
    out << ", .count=" << v.count;
    out << ", contents={";
    bool first = true;
    for (const auto &e : v) {
        if (first) {
            first = false;
        } else {
            out << ", ";
        }
        out << e;
    }
    out << "}}";
    return out;
}

}  // namespace kickmix

#endif
