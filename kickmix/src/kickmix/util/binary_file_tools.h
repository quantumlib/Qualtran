#ifndef KICKMIX_CIRCUIT_GEN_BINARY_FILE_TOOLS_H
#define KICKMIX_CIRCUIT_GEN_BINARY_FILE_TOOLS_H

#include <bit>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace kickmix {

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
        val = __builtin_bswap32(val);
    }
    fwrite_else_throw(&val, sizeof(uint32_t), file);
}

inline void write_u64_be(FILE *file, uint64_t val) {
    if constexpr (std::endian::native == std::endian::big) {
        val = __builtin_bswap64(val);
    }
    fwrite_else_throw(&val, sizeof(uint64_t), file);
}

bool write_u32_block_be(FILE *file, const uint32_t *data, size_t count);

}  // namespace kickmix

#endif
