#include "kickmix/util/chacha.h"

#include "gtest/gtest.h"

#include "kickmix/simd/simd.test.h"

using namespace kickmix;

namespace {
constexpr uint32_t RFC8439_INIT_WORDS[16] = {
    0x61707865,
    0x3320646e,
    0x79622d32,
    0x6b206574,
    0x03020100,
    0x07060504,
    0x0b0a0908,
    0x0f0e0d0c,
    0x13121110,
    0x17161514,
    0x1b1a1918,
    0x1f1e1d1c,
    0x00000001,
    0x09000000,
    0x4a000000,
    0x00000000,
};
}  // namespace

// Verifies ChaCha8<W> across all SIMD widths.
TEST_EACH_SIZE_W(chacha8, test_vector, {
    ChaCha8<W> rng;
    for (size_t i = 0; i < 16; i++) {
        rng.state[i] = W::from_u32_broadcast(RFC8439_INIT_WORDS[i]);
    }
    rng.buffer_idx = 16;

    const uint32_t expected_block[16] = {
        0xfb9dadee,
        0x3e4460bc,
        0xba11689d,
        0x3a0ae6b8,
        0x0d1e00c6,
        0x655f98fb,
        0xa40ecbef,
        0x1c415424,
        0xf77e7464,
        0xe066473d,
        0x20190ec2,
        0x17b15c8e,
        0x2687d477,
        0x5de65231,
        0x7f94ffc5,
        0x2b3bb2ca,
    };

    for (size_t i = 0; i < 16; i++) {
        W out = rng.next();
        ASSERT_EQ(out, W::from_u32_broadcast(expected_block[i])) << "i=" << i;
    }
    // Block counter should have incremented from 1 to 2.
    ASSERT_EQ(rng.state[12], W::from_u32_broadcast(2));
})

// Verifies ChaCha20<W> against the RFC 8439 Section 2.3.2 test vector across all SIMD widths.
TEST_EACH_SIZE_W(chacha20, rfc8439_test_vector, {
    ChaCha20<W> rng;
    for (size_t i = 0; i < 16; i++) {
        rng.state[i] = W::from_u32_broadcast(RFC8439_INIT_WORDS[i]);
    }
    rng.buffer_idx = 16;

    const uint32_t expected_block[16] = {
        0xe4e7f110,
        0x15593bd1,
        0x1fdd0f50,
        0xc47120a3,
        0xc7f4d1c7,
        0x0368c033,
        0x9aaa2204,
        0x4e6cd4c3,
        0x466482d2,
        0x09aa9f07,
        0x05d7c214,
        0xa2028bd9,
        0xd19c12b5,
        0xb94e16de,
        0xe883d0cb,
        0x4e3c50a2,
    };

    for (size_t i = 0; i < 16; i++) {
        W out = rng.next();
        ASSERT_EQ(out, W::from_u32_broadcast(expected_block[i])) << "i=" << i;
    }
    // Block counter should have incremented from 1 to 2.
    ASSERT_EQ(rng.state[12], W::from_u32_broadcast(2));
})
