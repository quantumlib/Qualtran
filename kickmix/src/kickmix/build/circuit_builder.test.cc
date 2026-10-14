#include "circuit_builder.h"

#include "gtest/gtest.h"

#include "kickmix/sim/sim.h"
#include "kickmix/simd/simd.h"
#include "kickmix/util/circuit_testing.test.h"
#include "test.test.h"

using namespace kickmix;

TEST(CircuitBuilder, basic) {
    CircuitBuilder builder;
    builder.ccx(QubitId(0), QubitId(1), QubitId(3));
    ASSERT_EQ(builder.finish_circuit(), Circuit(R"CIRCUIT(
        CCX q0 q1 q3
    )CIRCUIT"));
}

TEST(CircuitBuilder, store_and) {
    CircuitBuilder builder;
    builder.bit_store_and(BitId(0), BitId(1), BitId(2));
    Sim<b64, false> sim(INDEPENDENT_TEST_RNG());
    auto circuit = builder.finish_circuit();
    ASSERT_EQ(circuit, Circuit(R"CIRCUIT(
        BIT_STORE0 b2
        PUSH_CONDITION if b0
        BIT_STORE1 b2 if b1
        POP_CONDITION
    )CIRCUIT"));
    sim.configure_for(circuit);
    for (auto &e : sim.bit_span()) {
        e.randomize(sim.rng);
    }
    sim.apply(circuit);
    ASSERT_EQ(sim.bit_span()[0] & sim.bit_span()[1], sim.bit_span()[2]);
}

TEST(CircuitBuilder, cccz_qqqqa) {
    CircuitBuilder builder;
    builder.cccz(QubitId{1}, QubitId{2}, QubitId{3}, QubitId{4}, std::vector<QubitId>{QubitId{5}});
    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
        q0: ----------------------

        q1: -----@----------@-----
                 |          |
        q2: -----@----------Z**b0-
                 |
        q3: -----|-@--------------
                 | |
        q4: -----|-Z--------------
                 | |
        q5:  |0>-X-@-HMR=b0
    )DIAGRAM");
}

TEST(CircuitBuilder, cccz_qbqqa) {
    CircuitBuilder builder;
    builder.cccz(QubitId{1}, BitId{2}, QubitId{3}, QubitId{4}, std::vector<QubitId>{QubitId{5}});
    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
        q0: -------

        q1: -@-----
             |
        q2: -|-----
             |
        q3: -@-----
             |
        q4: -Z**b2-
    )DIAGRAM");
}

TEST(CircuitBuilder, cccz_bqcqa) {
    CircuitBuilder builder;
    builder.cccz(BitId{1}, QubitId{2}, true, QubitId{4}, std::vector<QubitId>{QubitId{5}});
    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
        q0: -------

        q1: -------

        q2: -@-----
             |
        q3: -|-----
             |
        q4: -Z**b1-
    )DIAGRAM");
}

TEST(CircuitBuilder, cccz_bqfqa) {
    CircuitBuilder builder;
    builder.cccz(BitId{1}, QubitId{2}, false, QubitId{4}, std::vector<QubitId>{QubitId{5}});
    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
    )DIAGRAM");
}

TEST(CircuitBuilder, cccz_qqqq) {
    CircuitBuilder builder;
    builder.cccz(QubitId{1}, QubitId{2}, QubitId{3}, QubitId{4});
    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
        q0: -X-@-X-@-
             | | | |
        q1: -@-|-@-|-
             | | | |
        q2: -@-|-@-|-
               |   |
        q3: ---@---@-
               |   |
        q4: ---Z---Z-
    )DIAGRAM");
}

TEST(CircuitBuilder, broadcast_ccx_aqq_1) {
    CircuitBuilder builder;
    builder.broadcast_ccx(
        QubitOrBitOrBool(true),
        array_z::copy_of(std::vector<QubitId>{QubitId{2}, QubitId{3}, QubitId{4}}),
        std::vector<QubitId>{QubitId{5}, QubitId{6}, QubitId{7}});
    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
        q0: -------

        q1: -------

        q2: -@-----
             |
        q3: -|-@---
             | |
        q4: -|-|-@-
             | | |
        q5: -X-|-|-
               | |
        q6: ---X-|-
                 |
        q7: -----X-
    )DIAGRAM");
}

TEST(CircuitBuilder, broadcast_ccx_aqq_2) {
    CircuitBuilder builder;
    builder.broadcast_ccx(
        QubitOrBitOrBool(false),
        array_z::copy_of(std::vector<QubitId>{QubitId{2}, QubitId{3}, QubitId{4}}),
        std::vector<QubitId>{QubitId{5}, QubitId{6}, QubitId{7}});
    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
    )DIAGRAM");
}

TEST(CircuitBuilder, broadcast_ccx_aqq_3) {
    CircuitBuilder builder;
    builder.broadcast_ccx(
        QubitOrBitOrBool(BitId{1}),
        array_z::copy_of(std::vector<QubitId>{QubitId{2}, QubitId{3}, QubitId{4}}),
        std::vector<QubitId>{QubitId{5}, QubitId{6}, QubitId{7}});
    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
              push_cond if b1       pop_cond
        q0: ---------------------------------

        q1: ---------------------------------

        q2: ------------------@--------------
                              |
        q3: ------------------|-@------------
                              | |
        q4: ------------------|-|-@----------
                              | | |
        q5: ------------------X-|-|----------
                                | |
        q6: --------------------X-|----------
                                  |
        q7: ----------------------X----------
    )DIAGRAM");
}
TEST(CircuitBuilder, broadcast_ccx_aqq_4) {
    CircuitBuilder builder;
    builder.broadcast_ccx(
        QubitOrBitOrBool(QubitId{1}),
        array_z::copy_of(std::vector<QubitId>{QubitId{2}, QubitId{3}, QubitId{4}}),
        std::vector<QubitId>{QubitId{5}, QubitId{6}, QubitId{7}});
    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
        q0: -------

        q1: -@-@-@-
             | | |
        q2: -@-|-|-
             | | |
        q3: -|-@-|-
             | | |
        q4: -|-|-@-
             | | |
        q5: -X-|-|-
               | |
        q6: ---X-|-
                 |
        q7: -----X-
    )DIAGRAM");
}

TEST(CircuitBuilder, broadcast_ccx_acq_1) {
    CircuitBuilder builder;
    builder.broadcast_ccx(
        QubitOrBitOrBool(true),
        array_z::copy_of(std::vector<BitId>{BitId{2}, BitId{3}, BitId{4}}),
        std::vector<QubitId>{QubitId{5}, QubitId{6}, QubitId{7}});
    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
        q0: -------

        q1: -------

        q2: -------

        q3: -------

        q4: -------

        q5: -X**b2-

        q6: -X**b3-

        q7: -X**b4-
    )DIAGRAM");
}

TEST(CircuitBuilder, broadcast_ccx_acq_2) {
    CircuitBuilder builder;
    builder.broadcast_ccx(
        QubitOrBitOrBool(false),
        array_z::copy_of(std::vector<BitId>{BitId{2}, BitId{3}, BitId{4}}),
        std::vector<QubitId>{QubitId{5}, QubitId{6}, QubitId{7}});
    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
    )DIAGRAM");
}

TEST(CircuitBuilder, broadcast_ccx_acq_3) {
    CircuitBuilder builder;
    builder.broadcast_ccx(
        QubitOrBitOrBool(BitId{1}),
        array_z::copy_of(std::vector<BitId>{BitId{2}, BitId{3}, BitId{4}}),
        std::vector<QubitId>{QubitId{5}, QubitId{6}, QubitId{7}});
    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
              push_cond if b1       pop_cond
        q0: ---------------------------------

        q1: ---------------------------------

        q2: ---------------------------------

        q3: ---------------------------------

        q4: ---------------------------------

        q5: ------------------X**b2----------

        q6: ------------------X**b3----------

        q7: ------------------X**b4----------
    )DIAGRAM");
}

TEST(CircuitBuilder, broadcast_ccx_acq_4) {
    CircuitBuilder builder;
    builder.broadcast_ccx(
        QubitOrBitOrBool(QubitId{1}),
        array_z::copy_of(std::vector<BitId>{BitId{2}, BitId{3}, BitId{4}}),
        std::vector<QubitId>{QubitId{5}, QubitId{6}, QubitId{7}});
    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
        q0: -------

        q1: -@-----
             |
        q2: -|-----
             |
        q3: -|-----
             |
        q4: -|-----
             |
        q5: -X**b2-
             |
        q6: -X**b3-
             |
        q7: -X**b4-
    )DIAGRAM");
}

TEST(CircuitBuilder, broadcast_ccx_abq) {
    {
        CircuitBuilder builder;
        builder.broadcast_ccx(
            QubitOrBitOrBool(true),
            array_z::copy_of(std::array<bool, 3>{false, true, true}),
            std::vector<QubitId>{QubitId{5}, QubitId{6}, QubitId{7}});
        expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
            q0: ---

            q1: ---

            q2: ---

            q3: ---

            q4: ---

            q5: ---

            q6: -X-

            q7: -X-
        )DIAGRAM");
    }

    {
        CircuitBuilder builder;
        builder.broadcast_ccx(
            QubitOrBitOrBool(false),
            array_z::copy_of(std::array<bool, 3>{false, true, true}),
            std::vector<QubitId>{QubitId{5}, QubitId{6}, QubitId{7}});
        expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
        )DIAGRAM");
    }
    {
        CircuitBuilder builder;
        builder.broadcast_ccx(
            QubitOrBitOrBool(BitId{1}),
            array_z::copy_of(std::array<bool, 3>{false, true, true}),
            std::vector<QubitId>{QubitId{5}, QubitId{6}, QubitId{7}});
        expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
                  push_cond if b1   pop_cond
            q0: -----------------------------

            q1: -----------------------------

            q2: -----------------------------

            q3: -----------------------------

            q4: -----------------------------

            q5: -----------------------------

            q6: ------------------X----------

            q7: ------------------X----------
        )DIAGRAM");
    }
    {
        CircuitBuilder builder;
        builder.broadcast_ccx(
            QubitOrBitOrBool(QubitId{1}),
            array_z::copy_of(std::array<bool, 3>{false, true, true}),
            std::vector<QubitId>{QubitId{5}, QubitId{6}, QubitId{7}});
        expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
            q0: ---

            q1: -@-
                 |
            q2: -|-
                 |
            q3: -|-
                 |
            q4: -|-
                 |
            q5: -|-
                 |
            q6: -X-
                 |
            q7: -X-
        )DIAGRAM");
    }
}

TEST(CircuitBuilder, broadcast_ccx_amq) {
    {
        CircuitBuilder builder;
        builder.broadcast_ccx(
            QubitOrBitOrBool(true),
            array_z::copy_of(std::vector<QubitOrBitOrBool>{false, true, BitId{1}, QubitId{2}}),
            std::vector<QubitId>{QubitId{4}, QubitId{5}, QubitId{6}, QubitId{7}});
        expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
            q0: ---------

            q1: ---------

            q2: -------@-
                       |
            q3: -------|-
                       |
            q4: -------|-
                       |
            q5: -X-----|-
                       |
            q6: -X**b1-|-
                       |
            q7: -------X-
        )DIAGRAM");
    }

    {
        CircuitBuilder builder;
        builder.broadcast_ccx(
            QubitOrBitOrBool(false),
            array_z::copy_of(std::vector<QubitOrBitOrBool>{false, true, BitId{1}, QubitId{2}}),
            std::vector<QubitId>{QubitId{4}, QubitId{5}, QubitId{6}, QubitId{7}});
        expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
        )DIAGRAM");
    }
    {
        CircuitBuilder builder;
        builder.broadcast_ccx(
            QubitOrBitOrBool(BitId{1}),
            array_z::copy_of(std::vector<QubitOrBitOrBool>{false, true, BitId{1}, QubitId{2}}),
            std::vector<QubitId>{QubitId{4}, QubitId{5}, QubitId{6}, QubitId{7}});
        expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
                  push_cond if b1         pop_cond
            q0: -----------------------------------

            q1: -----------------------------------

            q2: ------------------------@----------
                                        |
            q3: ------------------------|----------
                                        |
            q4: ------------------------|----------
                                        |
            q5: ------------------X-----|----------
                                        |
            q6: ------------------X**b1-|----------
                                        |
            q7: ------------------------X----------
        )DIAGRAM");
    }
    {
        CircuitBuilder builder;
        builder.broadcast_ccx(
            QubitOrBitOrBool(QubitId{1}),
            array_z::copy_of(std::vector<QubitOrBitOrBool>{false, true, BitId{1}, QubitId{2}}),
            std::vector<QubitId>{QubitId{4}, QubitId{5}, QubitId{6}, QubitId{7}});
        expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
            q0: ---------

            q1: -@-----@-
                 |     |
            q2: -|-----@-
                 |     |
            q3: -|-----|-
                 |     |
            q4: -|-----|-
                 |     |
            q5: -X-----|-
                 |     |
            q6: -X**b1-|-
                       |
            q7: -------X-
        )DIAGRAM");
    }
}

constexpr uint32_t BF = 0b00000000000000000000000000000000;
constexpr uint32_t BT = 0b11111111111111111111111111111111;
constexpr uint32_t B0 = 0b11111111111111110000000000000000;
constexpr uint32_t B1 = 0b11111111000000001111111100000000;
constexpr uint32_t B2 = 0b11110000111100001111000011110000;
constexpr uint32_t B3 = 0b11001100110011001100110011001100;
constexpr uint32_t B4 = 0b10101010101010101010101010101010;

struct Key {
    uint32_t tagged_id;
    Key(QubitOrBitOrBool q) : tagged_id(q.tagged_id) {
    }
    Key(QubitOrXBitOrXBool q) : tagged_id(q.tagged_id) {
    }
    Key(QubitOrMinusState q) : tagged_id(q.tagged_id) {
    }
    Key(QubitId q) : tagged_id(q.tagged_id) {
    }
    Key(BitId q) : tagged_id(q.tagged_id) {
    }
    Key(XBool q) : tagged_id(q.tagged_id) {
    }
    Key(bool q) : tagged_id((uint32_t)q) {
    }
    bool is_bit_or_x_bit() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::BIT_ID ||
               (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::XBIT_ID;
    }
    bool is_qubit() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::QUBIT_ID;
    }
    constexpr inline uint32_t untagged_id() const {
        return tagged_id & UNTAGGED_MASK;
    }
    bool operator==(const Key &other) const = default;
    bool operator<(const Key &other) const {
        return tagged_id < other.tagged_id;
    }
};

void apply_to(const Circuit &circuit, std::map<Key, uint32_t> &vals, uint32_t &phase) {
    Sim<b64, false> sim(std::mt19937_64{0});
    sim.configure_for(circuit);
    sim.clear_for_shot();
    for (const auto &kv : vals) {
        if (kv.first.is_qubit()) {
            sim.qubit_span()[kv.first.untagged_id()] = decltype(sim)::ubits_t::from_u32_broadcast(kv.second);
        } else if (kv.first.is_bit_or_x_bit()) {
            sim.bit_span()[kv.first.untagged_id()] = decltype(sim)::ubits_t::from_u32_broadcast(kv.second);
        }
    }
    sim.apply(circuit);
    for (auto &kv : vals) {
        if (kv.first.is_qubit()) {
            kv.second = sim.qubit_span()[kv.first.untagged_id()].u32(0);
        } else if (kv.first.is_bit_or_x_bit()) {
            kv.second = sim.bit_span()[kv.first.untagged_id()].u32(0);
        }
    }
    vals[true] = BT;
    vals[false] = BF;
    vals[MINUS_KET] = BT;
    vals[PLUS_KET] = BF;
    phase = sim.global_phase_ref().u32(0);
}

static inline std::array<QubitOrBitOrBool, 4> qz_cases(uint32_t index) {
    return std::array<QubitOrBitOrBool, 4>{false, true, QubitId{index}, BitId{index}};
}

static inline std::array<QubitOrXBitOrXBool, 4> qx_cases(uint32_t index) {
    return std::array<QubitOrXBitOrXBool, 4>{PLUS_KET, MINUS_KET, QubitId{index}, XBitId{index}};
}

TEST(CircuitBuilder, cx) {
    for (QubitOrBitOrBool c : qz_cases(0)) {
        for (QubitOrXBitOrXBool t : qx_cases(1)) {
            CircuitBuilder builder;
            builder.cx(c, t);
            builder.z(QubitId{10});
            builder.z(BitId{10});

            std::map<Key, uint32_t> vals;
            uint32_t phase = 0;
            vals[c] = B0;
            vals[t] = B1;
            apply_to(builder.finish_circuit(), vals, phase);
            ASSERT_EQ(phase, t.is_qubit() ? 0 : vals[t] & vals[c]) << c << ", " << t;
            ASSERT_TRUE(!t.is_qubit() || vals[t] == (B1 ^ vals[c])) << c << ", " << t;
            ASSERT_TRUE(c.is_bool() || vals[c] == B0);
        }
    }
}

TEST(CircuitBuilder, cz) {
    for (QubitOrBitOrBool c1 : qz_cases(0)) {
        for (QubitOrBitOrBool c2 : qz_cases(1)) {
            CircuitBuilder builder;
            builder.cz(c1, c2);
            builder.z(QubitId{10});
            builder.z(BitId{10});

            std::map<Key, uint32_t> vals;
            uint32_t phase = 0;
            vals[c1] = B0;
            vals[c2] = B1;
            apply_to(builder.finish_circuit(), vals, phase);
            ASSERT_EQ(phase, vals[c1] & vals[c2]) << c1 << ", " << c2;
            ASSERT_TRUE(c1.is_bool() || vals[c1] == B0);
            ASSERT_TRUE(c2.is_bool() || vals[c2] == B1);
        }
    }
}

TEST(CircuitBuilder, ccx) {
    for (QubitOrBitOrBool c1 : qz_cases(0)) {
        for (QubitOrBitOrBool c2 : qz_cases(1)) {
            for (QubitOrXBitOrXBool t : qx_cases(2)) {
                CircuitBuilder builder;
                builder.ccx(c1, c2, t);
                builder.z(QubitId{10});
                builder.z(BitId{10});

                std::map<Key, uint32_t> vals;
                uint32_t phase = 0;
                vals[c1] = B0;
                vals[c2] = B1;
                vals[t] = B2;
                apply_to(builder.finish_circuit(), vals, phase);
                ASSERT_EQ(phase, t.is_qubit() ? 0 : vals[t] & vals[c1] & vals[c2]) << c1 << ", " << c2 << ", " << t;
                ASSERT_TRUE(!t.is_qubit() || vals[t] == (B2 ^ (vals[c1] & vals[c2]))) << c1 << ", " << c2 << ", " << t;
                ASSERT_TRUE(c1.is_bool() || vals[c1] == B0);
                ASSERT_TRUE(c2.is_bool() || vals[c2] == B1);
            }
        }
    }
}

TEST(CircuitBuilder, ccz) {
    for (QubitOrBitOrBool c1 : qz_cases(0)) {
        for (QubitOrBitOrBool c2 : qz_cases(1)) {
            for (QubitOrBitOrBool c3 : qz_cases(2)) {
                CircuitBuilder builder;
                builder.ccz(c1, c2, c3);
                builder.z(QubitId{10});
                builder.z(BitId{10});

                std::map<Key, uint32_t> vals;
                uint32_t phase = 0;
                vals[c1] = B0;
                vals[c2] = B1;
                vals[c3] = B2;
                apply_to(builder.finish_circuit(), vals, phase);
                ASSERT_EQ(phase, vals[c1] & vals[c2] & vals[c3]) << c1 << ", " << c2 << ", " << c3;
                ASSERT_TRUE(c1.is_bool() || vals[c1] == B0);
                ASSERT_TRUE(c2.is_bool() || vals[c2] == B1);
                ASSERT_TRUE(c3.is_bool() || vals[c3] == B2);
            }
        }
    }
}

TEST(CircuitBuilder, parity_ccx) {
    for (QubitOrBitOrBool p1 : qz_cases(0)) {
        for (QubitOrBitOrBool p2 : qz_cases(1)) {
            for (QubitOrBitOrBool c : qz_cases(2)) {
                for (QubitId t : std::array<QubitId, 1>{QubitId{3}}) {
                    CircuitBuilder builder;
                    builder.parity_ccx({p1, p2}, c, t);
                    builder.z(QubitId{10});
                    builder.z(BitId{10});

                    std::map<Key, uint32_t> vals;
                    uint32_t phase = 0;
                    vals[p1] = B0;
                    vals[p2] = B1;
                    vals[c] = B2;
                    vals[t] = B3;
                    apply_to(builder.finish_circuit(), vals, phase);
                    ASSERT_EQ(phase, 0);
                    ASSERT_EQ(vals[t], B3 ^ ((vals[p1] ^ vals[p2]) & vals[c]));
                    ASSERT_TRUE(p1.is_bool() || vals[p1] == B0);
                    ASSERT_TRUE(p2.is_bool() || vals[p2] == B1);
                    ASSERT_TRUE(c.is_bool() || vals[c] == B2);
                }
            }
        }
    }
}

TEST(CircuitBuilder, parity_cccx) {
    for (QubitOrBitOrBool p1 : qz_cases(0)) {
        for (QubitOrBitOrBool p2 : qz_cases(1)) {
            for (QubitOrBitOrBool c1 : qz_cases(2)) {
                for (QubitOrBitOrBool c2 : qz_cases(3)) {
                    for (QubitId t : std::array<QubitId, 1>{QubitId{4}}) {
                        CircuitBuilder builder;
                        builder.parity_cccx({p1, p2}, c1, c2, t, {});
                        builder.z(QubitId{10});
                        builder.z(BitId{10});

                        std::map<Key, uint32_t> vals;
                        uint32_t phase = 0;
                        vals[p1] = B0;
                        vals[p2] = B1;
                        vals[c1] = B2;
                        vals[c2] = B3;
                        vals[t] = B4;
                        apply_to(builder.finish_circuit(), vals, phase);
                        ASSERT_TRUE(p1.is_bool() || vals[p1] == B0);
                        ASSERT_TRUE(p2.is_bool() || vals[p2] == B1);
                        ASSERT_TRUE(c1.is_bool() || vals[c1] == B2);
                        ASSERT_TRUE(c2.is_bool() || vals[c2] == B3);
                        ASSERT_EQ(vals[t], B4 ^ ((vals[p1] ^ vals[p2]) & vals[c1] & vals[c2]));
                        ASSERT_EQ(phase, 0);
                    }
                }
            }
        }
    }
}

TEST(CircuitBuilder, parity_ccz) {
    for (QubitOrBitOrBool p1 : qz_cases(0)) {
        for (QubitOrBitOrBool p2 : qz_cases(1)) {
            for (QubitOrBitOrBool c1 : qz_cases(2)) {
                for (QubitOrBitOrBool c2 : qz_cases(3)) {
                    CircuitBuilder builder;
                    builder.parity_ccz({p1, p2}, c1, c2);
                    builder.z(QubitId{10});
                    builder.z(BitId{10});

                    std::map<Key, uint32_t> vals;
                    uint32_t phase = 0;
                    vals[p1] = B0;
                    vals[p2] = B1;
                    vals[c1] = B2;
                    vals[c2] = B3;
                    apply_to(builder.finish_circuit(), vals, phase);
                    ASSERT_TRUE(p1.is_bool() || vals[p1] == B0);
                    ASSERT_TRUE(p2.is_bool() || vals[p2] == B1);
                    ASSERT_TRUE(c1.is_bool() || vals[c1] == B2);
                    ASSERT_TRUE(c2.is_bool() || vals[c2] == B3);
                    ASSERT_EQ(phase, (vals[p1] ^ vals[p2]) & vals[c1] & vals[c2]);
                }
            }
        }
    }
}

TEST(CircuitBuilder, parity_cccz) {
    for (QubitOrBitOrBool p1 : qz_cases(0)) {
        for (QubitOrBitOrBool p2 : qz_cases(1)) {
            for (QubitOrBitOrBool c1 : qz_cases(2)) {
                for (QubitOrBitOrBool c2 : qz_cases(3)) {
                    for (QubitOrBitOrBool c3 : qz_cases(4)) {
                        CircuitBuilder builder;
                        builder.parity_cccz({p1, p2}, c1, c2, c3, {});
                        builder.z(QubitId{10});
                        builder.z(BitId{10});

                        std::map<Key, uint32_t> vals;
                        uint32_t phase = 0;
                        vals[p1] = B0;
                        vals[p2] = B1;
                        vals[c1] = B2;
                        vals[c2] = B3;
                        vals[c3] = B4;
                        apply_to(builder.finish_circuit(), vals, phase);
                        ASSERT_TRUE(p1.is_bool() || vals[p1] == B0);
                        ASSERT_TRUE(p2.is_bool() || vals[p2] == B1);
                        ASSERT_TRUE(c1.is_bool() || vals[c1] == B2);
                        ASSERT_TRUE(c2.is_bool() || vals[c2] == B3);
                        ASSERT_TRUE(c3.is_bool() || vals[c3] == B4);
                        ASSERT_EQ(phase, (vals[p1] ^ vals[p2]) & vals[c1] & vals[c2] & vals[c3]);
                    }
                }
            }
        }
    }
}

TEST(CircuitBuilder, write_analysis_svg_to_every_gate) {
    CircuitBuilder builder;
    {
        auto mark = builder.raii_mark_block_entry("every <gate & op>");
        builder.mut.append(circuit_with_every_operation());
    }
    std::stringstream ss;
    builder.write_analysis_svg_to(ss, 10);
    std::string svg = ss.str();
    ASSERT_TRUE(!svg.empty());
    ASSERT_EQ(svg.find("entire circuit"), std::string::npos);
    ASSERT_EQ(svg.find("<gate & op>"), std::string::npos);
    ASSERT_NE(svg.find(">every &lt;gate &amp; op&gt;</text>"), std::string::npos);
}
