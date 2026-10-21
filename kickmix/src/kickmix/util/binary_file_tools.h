#ifndef KICKMIX_CIRCUIT_GEN_BINARY_FILE_TOOLS_H
#define KICKMIX_CIRCUIT_GEN_BINARY_FILE_TOOLS_H

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace kickmix {

inline uint32_t bswap32(uint32_t val) {
#if defined(_MSC_VER)
    return _byteswap_ulong(val);
#else
    return __builtin_bswap32(val);
#endif
}

inline uint64_t bswap64(uint64_t val) {
#if defined(_MSC_VER)
    return _byteswap_uint64(val);
#else
    return __builtin_bswap64(val);
#endif
}

inline FILE *open_tmpfile_else_throw() {
    FILE *f = tmpfile();
    f = tmpfile();
    if (f == nullptr) {
        throw std::runtime_error("Failed to create temporary file.");
    }
    return f;
}

inline void fread_else_throw(void *ptr, size_t size, FILE *file) {
    if (size && fread(ptr, size, 1, file) != 1) {
        throw std::invalid_argument("Failed to read from file.");
    }
}
inline void fwrite_else_throw(const void *ptr, size_t size, FILE *file) {
    if (size && fwrite(ptr, size, 1, file) != 1) {
        throw std::invalid_argument("Failed to write to file.");
    }
}

inline void write_u32_be(FILE *file, uint32_t val) {
    if constexpr (std::endian::native == std::endian::big) {
        val = bswap32(val);
    }
    fwrite_else_throw(&val, sizeof(uint32_t), file);
}

inline void write_u64_be(FILE *file, uint64_t val) {
    if constexpr (std::endian::native == std::endian::big) {
        val = bswap64(val);
    }
    fwrite_else_throw(&val, sizeof(uint64_t), file);
}

void write_u32_block_be(FILE *file, const uint32_t *data, size_t count);

}  // namespace kickmix

#endif
