#include "util.h"

#include <algorithm>
#include <cstring>

uint32_t kickmix::max_u32(const uint32_t *items, size_t num_items) {
    uint32_t result = 0;
    for (size_t k = 0; k < num_items; k++) {
        result = std::max(result, items[k]);
    }
    return result;
}

void kickmix::memcpy_repeat(void *dst, const void *src, size_t src_len, size_t repetitions) {
    if (repetitions == 0) {
        return;
    }
    char *dst_char = static_cast<char *>(dst);
    const char *src_char = static_cast<const char *>(src);

    memcpy(dst_char, src_char, src_len);
    size_t num_reps_written = 1;
    size_t num_bytes_written = src_len;
    while (num_reps_written * 2 < repetitions) {
        memcpy(dst_char + num_bytes_written, dst_char, num_bytes_written);
        num_bytes_written <<= 1;
        num_reps_written <<= 1;
    }
    if (num_reps_written < repetitions) {
        memcpy(dst_char + num_bytes_written, dst_char, src_len * repetitions - num_bytes_written);
    }
}
