#ifndef KICKMIX_CIRCUIT_GEN_BINARY_FILE_TOOLS_H
#define KICKMIX_CIRCUIT_GEN_BINARY_FILE_TOOLS_H

#include <bit>
#include <cstdint>
#include <cstdio>

namespace kickmix {

#ifdef __APPLE__
// macOS has no fread_unlocked/fwrite_unlocked; use the locking stdio APIs there.
inline size_t fread_unlocked(void *ptr, size_t size, size_t count, FILE *file) {
    return fread(ptr, size, count, file);
}
inline size_t fwrite_unlocked(const void *ptr, size_t size, size_t count, FILE *file) {
    return fwrite(ptr, size, count, file);
}
#endif

inline bool write_u32_be(FILE *file, uint32_t val) {
    if constexpr (std::endian::native == std::endian::big) {
        val = __builtin_bswap32(val);
    }
    return fwrite_unlocked(&val, sizeof(uint32_t), 1, file) != 1;
}

inline bool write_u64_be(FILE *file, uint64_t val) {
    if constexpr (std::endian::native == std::endian::big) {
        val = __builtin_bswap64(val);
    }
    return fwrite_unlocked(&val, sizeof(uint64_t), 1, file) != 1;
}

bool write_u32_block_be(FILE *file, const uint32_t *data, size_t count);

}  // namespace kickmix

#endif
