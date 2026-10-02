#ifndef KICKGEN_MEM_UTIL_H
#define KICKGEN_MEM_UTIL_H

#include <cstddef>
#include <cstdint>

namespace kickmix {

uint32_t max_u32(const uint32_t *items, size_t num_items);
void memcpy_repeat(void *dst, const void *src, size_t src_len, size_t repetitions);

}  // namespace kickmix

#endif
