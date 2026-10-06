#ifndef KICKMIX_SIMD_TEST_H
#define KICKMIX_SIMD_TEST_H

#include "simd.h"

#ifdef __AVX512F__
#define TEST_EACH_SIZE_W_AVX512(group, name, ...) \
    TEST(group, name##_512_avx) {                 \
        using W = b512_avx;                       \
        __VA_ARGS__                               \
    }
#else
#define TEST_EACH_SIZE_W_AVX512(group, name, ...)
#endif

#ifdef __AVX__
#define TEST_EACH_SIZE_W_AVX(group, name, ...) \
    TEST(group, name##_256_avx) {              \
        using W = b256_avx;                    \
        __VA_ARGS__                            \
    }
#else
#define TEST_EACH_SIZE_W_AVX(group, name, ...)
#endif

#ifdef __SSE__
#define TEST_EACH_SIZE_W_SSE(group, name, ...) \
    TEST(group, name##_128_sse) {              \
        using W = b128_sse;                    \
        __VA_ARGS__                            \
    }
#else
#define TEST_EACH_SIZE_W_SSE(group, name, ...)
#endif

#define TEST_EACH_SIZE_W(group, name, ...)            \
    TEST_EACH_SIZE_W_AVX512(group, name, __VA_ARGS__) \
    TEST_EACH_SIZE_W_AVX(group, name, __VA_ARGS__)    \
    TEST_EACH_SIZE_W_SSE(group, name, __VA_ARGS__)    \
    TEST(group, name##_512_polyfill) {                \
        using W = b512_polyfill;                      \
        __VA_ARGS__                                   \
    }                                                 \
    TEST(group, name##_256_polyfill) {                \
        using W = b256_polyfill;                      \
        __VA_ARGS__                                   \
    }                                                 \
    TEST(group, name##_128_polyfill) {                \
        using W = b128_polyfill;                      \
        __VA_ARGS__                                   \
    }                                                 \
    TEST(group, name##_64_polyfill) {                 \
        using W = b64_polyfill;                       \
        __VA_ARGS__                                   \
    }

#endif
