#include "kickmix/util/chacha.h"

#include "gtest/gtest.h"

#include "kickmix/simd/simd.test.h"

using namespace kickmix;

// Verifies ChaCha8<W> against IETF draft-strombergson-chacha-test-vectors-01 Section 3 TC8
// (256-bit key, 8 rounds, page 38):
// https://www.ietf.org/archive/id/draft-strombergson-chacha-test-vectors-01.txt
TEST_EACH_SIZE_W(chacha8, test_vector, {
    ChaCha8<W> rng;
    // "Internal state after init" from TC8 (256-bit key, Rounds: 8, page 38 of
    // https://www.ietf.org/archive/id/draft-strombergson-chacha-test-vectors-01.txt):
    const uint32_t init_words[16] = {
        0x61707865,
        0x3320646e,
        0x79622d32,
        0x6b206574,
        0xb1c16ec4,
        0x78a8e88c,
        0xe7375a72,
        0x35b7df80,
        0x2eed681f,
        0xfb794c19,
        0xe1beaec6,
        0x5d9767a6,
        0x00000000,
        0x00000000,
        0xd531da1a,
        0x218268cf,
    };
    for (size_t i = 0; i < 16; i++) {
        rng.state[i] = W::from_u32_broadcast(init_words[i]);
    }
    rng.buffer_idx = 16;

    // Keystream block 0 (little-endian 32-bit words from TC8, 256-bit key, Rounds: 8, page 38 of
    // https://www.ietf.org/archive/id/draft-strombergson-chacha-test-vectors-01.txt):
    const uint32_t expected_block[16] = {
        0xb4518783,
        0x8add8d2d,
        0x88f4773d,
        0x75baa225,
        0x7c04f42c,
        0x97a508b3,
        0x9774f28e,
        0xc974e33b,
        0x0648d86a,
        0x7b417158,
        0xe634f008,
        0xa946fe81,
        0x615c7d3f,
        0x146630d1,
        0x57f2aad4,
        0x8bf0cfa7,
    };

    for (size_t i = 0; i < 16; i++) {
        W out = rng.next();
        ASSERT_EQ(out, W::from_u32_broadcast(expected_block[i])) << "i=" << i;
    }
    // Block counter should have incremented from 0 to 1.
    ASSERT_EQ(rng.state[12], W::from_u32_broadcast(1));
})

// Verifies ChaCha20<W> against the RFC 8439 Section 2.3.2 test vector across all SIMD widths:
// https://www.rfc-editor.org/info/rfc8439/#section-2.3.2
TEST_EACH_SIZE_W(chacha20, rfc8439_test_vector, {
    ChaCha20<W> rng;
    // Initial state from https://www.rfc-editor.org/info/rfc8439/#section-2.3.2
    const uint32_t init_words[16] = {
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
    for (size_t i = 0; i < 16; i++) {
        rng.state[i] = W::from_u32_broadcast(init_words[i]);
    }
    rng.buffer_idx = 16;

    // Output state after 20 rounds + feedforward addition from
    // https://www.rfc-editor.org/info/rfc8439/#section-2.3.2
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
