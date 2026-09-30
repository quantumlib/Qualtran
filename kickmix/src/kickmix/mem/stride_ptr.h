#ifndef KICKMIX_MEM_STRIDE_PTR_H
#define KICKMIX_MEM_STRIDE_PTR_H

#include <cstdint>

namespace kickmix {

/// A pointer that moves in larger jumps when offset.
template <typename T>
struct stride_ptr {
    T *ptr;          // The underlying pointer.
    int64_t stride;  // How far to move the underlying pointer when incrementing the stride_ptr.

    inline T &operator*() const {
        return *ptr;
    }
    inline T &operator[](int64_t offset) const {
        return *(ptr + stride * offset);
    }
    inline stride_ptr<T> operator+(int64_t offset) const {
        return stride_ptr<T>{.ptr = ptr + stride * offset, .stride = stride};
    }
    inline stride_ptr<T> &operator+=(int64_t offset) {
        ptr += stride * offset;
        return *this;
    }
    inline stride_ptr<T> &operator-=(int64_t offset) {
        ptr -= stride * offset;
        return *this;
    }
    inline stride_ptr<T> operator-(int64_t offset) const {
        return stride_ptr<T>{.ptr = ptr - stride * offset, .stride = stride};
    }
    inline stride_ptr<T> &operator++() {
        ptr += stride;
        return *this;
    }
    inline stride_ptr<T> operator++(int) {
        stride_ptr<T> result = *this;
        ptr += stride;
        return result;
    }
    inline bool operator==(const stride_ptr &other) const {
        return ptr == other.ptr;
    }
};
template <typename T>
inline stride_ptr<T> operator+(int64_t offset, stride_ptr<T> ptr) {
    return ptr + offset;
}

}  // namespace kickmix

#endif
