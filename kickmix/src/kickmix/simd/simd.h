#ifndef KICKMIX_SIMD_H
#define KICKMIX_SIMD_H

#include "b128_polyfill.h"
#include "b128_sse.h"
#include "b256_avx.h"
#include "b256_polyfill.h"
#include "b512_avx.h"
#include "b512_polyfill.h"
#include "b64_polyfill.h"

namespace kickmix {

using b64 = b64_polyfill;

#ifdef __SSE__
using b128 = b128_sse;
#else
using b128 = b128_polyfill;
#endif

#ifdef __AVX__
using b256 = b256_avx;
#else
using b256 = b256_polyfill;
#endif

#ifdef __AVX512F__
using b512 = b512_avx;
#else
using b512 = b512_polyfill;
#endif

}  // namespace kickmix

#endif
