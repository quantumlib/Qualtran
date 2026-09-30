#include "gen_cmp_low_space.h"

#include "gtest/gtest.h"

#include "gen_cmp.h"
#include "kickmix/sim/fuzzer.h"
#include "kickmix/util/circuit_testing.test.h"
#include "test.test.h"

using namespace kickmix;

TEST(masked_phase_by_prefix_using_dirty_workspace, diagram_dirty) {
    CircuitBuilder builder;
    auto target = builder.append_register(10);
    auto masks = builder.append_classical_register_mixed_result(10);
    auto clean = builder.reserve_qubits(0);
    CircuitGenCtx ctx{.clean_workspace = clean};
    auto dirty = builder.reserve_qubits(7);
    ctx = ctx.with_more_dirty_qubits(dirty);
    masked_phase_by_prefix_using_dirty_workspace(builder, ctx, target, masks);
    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
                      reg1[0]=b0
         q0: -reg0[0]------------Z**b0-----------@-------------------------------------------@-------------------------------
                      reg1[1]=b1                 |                                           |
         q1: -reg0[1]----------------------------@-------------------------------------------@-------------------------------
                      reg1[2]=b2                 |                                           |
         q2: -reg0[2]--------------------------@-|-@---------------------------------------@-|-@-----------------------------
                      reg1[3]=b3               | | |                                       | | |
         q3: -reg0[3]------------------------@-|-|-|-@-----------------------------------@-|-|-|-@---------------------------
                      reg1[4]=b4             | | | | |                                   | | | | |
         q4: -reg0[4]----------------------@-|-|-|-|-|-@-------------------------------@-|-|-|-|-|-@-------------------------
                      reg1[5]=b5           | | | | | | |                               | | | | | | |
         q5: -reg0[5]--------------------@-|-|-|-|-|-|-|-@---------------------------@-|-|-|-|-|-|-|-@-----------------------
                      reg1[6]=b6         | | | | | | | | |                           | | | | | | | | |
         q6: -reg0[6]------------------@-|-|-|-|-|-|-|-|-|-@-----------------------@-|-|-|-|-|-|-|-|-|-@---------------------
                      reg1[7]=b7       | | | | | | | | | | |                       | | | | | | | | | | |
         q7: -reg0[7]------------@-----|-|-|-|-|-|-|-|-|-|-|-@-------------------@-|-|-|-|-|-|-|-|-|-|-|-@-------------------
                      reg1[8]=b8 |     | | | | | | | | | | | |                   | | | | | | | | | | | | |
         q8: -reg0[8]------------|-----|-|-|-|-|-|-|-|-|-|-|-|-------@-----@-----|-|-|-|-|-|-|-|-|-|-|-|-|-------@-----@-----
                      reg1[9]=b9 |     | | | | | | | | | | | |       |     |     | | | | | | | | | | | | |       |     |
         q9: -reg0[9]------------|-----|-|-|-|-|-|-|-|-|-|-|-|-------|-----@-----|-|-|-|-|-|-|-|-|-|-|-|-|-------|-----@-----
                                 |     | | | | | | | | | | | |       |     |     | | | | | | | | | | | | |       |     |
        q10: --------------------|-----|-|-|-|-@-X-@-|-|-|-|-|-Z**b1-|-----|-----|-|-|-|-|-@-X-@-|-|-|-|-|-Z**b1-|-----|-----
                                 |     | | | | |   | | | | | |       |     |     | | | | | |   | | | | | |       |     |
        q11: --------------------|-----|-|-|-@-X---X-@-|-|-|-|-Z**b2-|-----|-----|-|-|-|-@-X---X-@-|-|-|-|-Z**b2-|-----|-----
                                 |     | | | |       | | | | |       |     |     | | | | |       | | | | |       |     |
        q12: --------------------|-----|-|-@-X-------X-@-|-|-|-Z**b3-|-----|-----|-|-|-@-X-------X-@-|-|-|-Z**b3-|-----|-----
                                 |     | | |           | | | |       |     |     | | | |           | | | |       |     |
        q13: --------------------|-----|-@-X-----------X-@-|-|-Z**b4-|-----|-----|-|-@-X-----------X-@-|-|-Z**b4-|-----|-----
                                 |     | |               | | |       |     |     | | |               | | |       |     |
        q14: --------------------|-----@-X---------------X-@-|-Z**b5-|-----|-----|-@-X---------------X-@-|-Z**b5-|-----|-----
                                 |     |                   | |       |     |     | |                   | |       |     |
        q15: --------------------@-----X-------------------X-@-Z**b6-|-----|-----@-X-------------------X-@-Z**b6-|-----|-----
                                 |                           |       |     |     |                       |       |     |
        q16: --------------------X---------------------------X-Z**b7-Z**b8-Z**b9-X-----------------------X-Z**b7-Z**b8-Z**b9-
    )DIAGRAM");
}

TEST(masked_phase_by_prefix_using_dirty_workspace, diagram_partial_dirty) {
    CircuitBuilder builder;
    auto target = builder.append_register(10);
    auto masks = builder.append_classical_register_qcarray_result(10);
    auto clean = builder.reserve_qubits(2);
    CircuitGenCtx ctx{.clean_workspace = clean};
    auto dirty = builder.reserve_qubits(5);
    ctx = ctx.with_more_dirty_qubits(dirty);
    masked_phase_by_prefix_using_dirty_workspace(builder, ctx, target, masks);
    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
                      reg1[0]=b0
         q0: -reg0[0]------------Z**b0---------@----------------------------------------------------------------@------------------------------
                      reg1[1]=b1               |                                                                |
         q1: -reg0[1]--------------------------@----------------------------------------------------------------Z**b10-------------------------
                      reg1[2]=b2               |
         q2: -reg0[2]--------------------------|-@-----------------------------------------------@---------------------------------------------
                      reg1[3]=b3               | |                                               |
         q3: -reg0[3]--------------------------|-|-@-----------------------------------@---------|---------------------------------------------
                      reg1[4]=b4               | | |                                   |         |
         q4: -reg0[4]------------------------@-|-|-|-@-------------------------------@-|---------|--------------@------------------------------
                      reg1[5]=b5             | | | | |                               | |         |              |
         q5: -reg0[5]----------------------@-|-|-|-|-|-@---------------------------@-|-|---------|--------------|------@-----------------------
                      reg1[6]=b6           | | | | | | |                           | | |         |              |      |
         q6: -reg0[6]--------------------@-|-|-|-|-|-|-|-@-----------------------@-|-|-|---------|--------------|------|-@---------------------
                      reg1[7]=b7         | | | | | | | | |                       | | | |         |              |      | |
         q7: -reg0[7]------------------@-|-|-|-|-|-|-|-|-|-@-------------------@-|-|-|-|---------|--------------|------|-|-@-------------------
                      reg1[8]=b8       | | | | | | | | | | |                   | | | | |         |              |      | | |
         q8: -reg0[8]------------------|-|-|-|-|-|-|-|-|-|-|-------@-----@-----|-|-|-|-|---------|--------------|------|-|-|-------@-----@-----
                      reg1[9]=b9       | | | | | | | | | | |       |     |     | | | | |         |              |      | | |       |     |
         q9: -reg0[9]------------------|-|-|-|-|-|-|-|-|-|-|-------|-----@-----|-|-|-|-|---------|--------------|------|-|-|-------|-----@-----
                                       | | | | | | | | | | |       |     |     | | | | |         |              |      | | |       |     |
        q10:                     |0>---|-|-|-|-X-@-|-|-|-|-|-Z**b1-|-----|-----|-|-|-|-|---------Z**b10-HMR=b10 |      | | |       |     |
                                       | | | |   | | | | | |       |     |     | | | | |                        |      | | |       |     |
        q11:                     |0>---|-|-|-|---X-@-|-|-|-|-Z**b2-|-----|-----|-|-|-|-@-HMR=b10                |      | | |       |     |
                                       | | | |     | | | | |       |     |     | | | | |                        |      | | |       |     |
        q12: --------------------------|-|-|-@-----X-@-|-|-|-Z**b3-|-----|-----|-|-|-@-X------------------------@------|-|-|-Z**b3-|-----|-----
                                       | | | |       | | | |       |     |     | | | |                          |      | | |       |     |
        q13: --------------------------|-|-@-X-------X-@-|-|-Z**b4-|-----|-----|-|-@-X--------------------------X------@-|-|-Z**b4-|-----|-----
                                       | | |           | | |       |     |     | | |                                   | | |       |     |
        q14: --------------------------|-@-X-----------X-@-|-Z**b5-|-----|-----|-@-X-----------------------------------X-@-|-Z**b5-|-----|-----
                                       | |               | |       |     |     | |                                       | |       |     |
        q15: --------------------------@-X---------------X-@-Z**b6-|-----|-----@-X---------------------------------------X-@-Z**b6-|-----|-----
                                       |                   |       |     |     |                                           |       |     |
        q16: --------------------------X-------------------X-Z**b7-Z**b8-Z**b9-X-------------------------------------------X-Z**b7-Z**b8-Z**b9-
    )DIAGRAM");
}

TEST(masked_phase_by_prefix_using_dirty_workspace, fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 20;
        size_t nc = rng() % 5;
        auto target = builder.append_register(n, "target");
        auto masks = builder.append_classical_register_mixed_result(n, "masks");
        auto clean = builder.append_register(nc, "@clean");
        auto dirty = builder.append_register(n, "@dirty");
        CircuitGenCtx ctx{.clean_workspace = clean};
        ctx = ctx.with_more_dirty_qubits(dirty);
        masked_phase_by_prefix_using_dirty_workspace(builder, ctx, target, masks);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        for (size_t k = 0; k < sample["target"].num_bits; k++) {
            if (!sample["target"][k]) {
                return;
            }
            sample.phase_half_turns += sample["masks"][k];
        }
    });

    fuzzer.fuzz(100, 128);
}

TEST(icmp, gen_xif_less_than_3anc_2ntof) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 30;
        auto lhs = builder.append_register(n, "lhs");
        auto rhs = builder.append_classical_register(n, "rhs");
        auto target = builder.append_register(1, "target")[0];
        QubitOrBitOrBool or_equal = builder.append_classical_register(1, "or_equal")[0];
        QubitOrTrue control = builder.append_register(1, "control")[0];
        auto clean = builder.append_register(3, "@clean");
        CircuitGenCtx ctx{.clean_workspace = clean};
        gen_xif_less_than_3anc_2ntof(
            builder, ctx, stride_span<const QubitId>{lhs}, rhs, target, or_equal, false, control);
    });

    fuzzer.use_input_sampler([](InputSample &sample, std::mt19937_64 &rng) {
        sample["lhs"].randomize(rng);
        sample["or_equal"].randomize(rng);
        sample["target"].randomize(rng);
        sample["control"].randomize(rng);
        if (rng() % 2) {
            // Give a near-value case to check <= vs < more thoroughly at large sizes.
            sample["rhs"] = sample["lhs"];
            sample["rhs"] += (int)(rng() % 20) - 10;
        } else {
            sample["rhs"].randomize(rng);
        }
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if (sample["control"]) {
            if (sample["or_equal"]) {
                sample["target"] ^= sample["lhs"] <= sample["rhs"];
            } else {
                sample["target"] ^= sample["lhs"] < sample["rhs"];
            }
        }
    });

    fuzzer.fuzz(100, 256);
}
