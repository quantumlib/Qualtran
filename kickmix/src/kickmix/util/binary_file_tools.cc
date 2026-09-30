#include "binary_file_tools.h"

#include <array>
#include <cstring>

using namespace kickmix;

void kickmix::write_u32_block_be(FILE *file, const uint32_t *data, size_t count) {
    if constexpr (std::endian::native == std::endian::little) {
        fwrite_else_throw(data, count * sizeof(uint32_t), file);
    } else {
        std::array<uint32_t, 128> buf;
        for (size_t k = 0; k < count; k += 128) {
            size_t n = std::min((size_t)128, count - k);
            memcpy(buf.data(), data, n * sizeof(uint32_t));
            data += n;
            for (size_t k2 = 0; k2 < n; k2++) {
                buf[k2] = __builtin_bswap32(buf[k2]);
            }
            fwrite_else_throw(buf.data(), n * sizeof(uint32_t), file);
        }
    }
}
