#ifndef KICKMIX_SIMD_PERF_H
#define KICKMIX_SIMD_PERF_H

#include "simd.h"

#ifdef __AVX512F__
#define BENCHMARK_EACH_SIZE_W_AVX512(name, ...) \
    BENCHMARK(name##_512_avx) {                 \
        using W = b512_avx;                     \
        __VA_ARGS__                             \
    }
#else
#define BENCHMARK_EACH_SIZE_W_AVX512(name, ...)
#endif

#ifdef __AVX__
#define BENCHMARK_EACH_SIZE_W_AVX(name, ...) \
    BENCHMARK(name##_256_avx) {              \
        using W = b256_avx;                  \
        __VA_ARGS__                          \
    }
#else
#define BENCHMARK_EACH_SIZE_W_AVX(name, ...)
#endif

#ifdef __SSE__
#define BENCHMARK_EACH_SIZE_W_SSE(name, ...) \
    BENCHMARK(name##_128_sse) {              \
        using W = b128_sse;                  \
        __VA_ARGS__                          \
    }
#else
#define BENCHMARK_EACH_SIZE_W_SSE(name, ...)
#endif

#define BENCHMARK_EACH_SIZE_W(name, ...)            \
    BENCHMARK_EACH_SIZE_W_AVX512(name, __VA_ARGS__) \
    BENCHMARK_EACH_SIZE_W_AVX(name, __VA_ARGS__)    \
    BENCHMARK_EACH_SIZE_W_SSE(name, __VA_ARGS__)    \
    BENCHMARK(name##_512_polyfill) {                \
        using W = b512_polyfill;                    \
        __VA_ARGS__                                 \
    }                                               \
    BENCHMARK(name##_256_polyfill) {                \
        using W = b256_polyfill;                    \
        __VA_ARGS__                                 \
    }                                               \
    BENCHMARK(name##_128_polyfill) {                \
        using W = b128_polyfill;                    \
        __VA_ARGS__                                 \
    }                                               \
    BENCHMARK(name##_64_polyfill) {                 \
        using W = b64_polyfill;                     \
        __VA_ARGS__                                 \
    }

#endif
