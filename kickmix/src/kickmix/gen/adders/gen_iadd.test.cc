#include "gen_iadd.h"

#include "gtest/gtest.h"

#include "kickmix/sim/fuzzer.h"
#include "kickmix/util/circuit_testing.test.h"
#include "test.test.h"

using namespace kickmix;

TEST(gen_iadd, zero_workspace_diagram) {
    CircuitBuilder builder;
    auto control = builder.append_register(1)[0];
    auto target = builder.append_register(6);
    auto offset = builder.append_register(6);
    gen_iadd(builder, CircuitGenCtx{}, target, offset, control);
    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
         q0: -reg0[0]-@-------------------@-@---@---@-----@-----@-----@-----@---@-
                      |                   | |   |   |     |     |     |     |   |
         q1: -reg1[0]-|---@---------------|-X-@-X-@-|-----|-----|-----|---@-X---|-
                      |   |               | | | | | |     |     |     |   | |   |
         q2: -reg1[1]-|-X-|-X-@-----------|-|-|-|-|-|-----|-----|---@-X-X-|-|-X-|-
                      | | | | |           | | | | | |     |     |   | | | | | | |
         q3: -reg1[2]-|-X-|-|-|-X-@-------|-|-|-|-|-|-----|---@-X-X-|-|-|-|-|-X-|-
                      | | | | | | |       | | | | | |     |   | | | | | | | | | |
         q4: -reg1[3]-|-X-|-|-|-|-|-X-@---|-|-|-|-|-|---@-X-X-|-|-|-|-|-|-|-|-X-|-
                      | | | | | | | | |   | | | | | |   | | | | | | | | | | | | |
         q5: -reg1[4]-|-X-|-|-|-|-|-|-|-X-|-@-|-@-|-X-X-|-|-|-|-|-|-|-|-|-|-|-X-|-
                      | | | | | | | | | | |   |   | | | | | | | | | | | | | | | |
         q6: -reg1[5]-X-X-|-|-|-|-|-|-|-|-X---X---X-|-|-|-|-|-|-|-|-|-|-|-|-|-X-X-
                      | | | | | | | | | | |   |   | | | | | | | | | | | | | | | |
         q7: -reg2[0]-|-|-@-|-|-|-|-|-|-|-|---|---|-|-|-|-|-|-|-|-|-|-|-|-@-@-|-|-
                      | | | | | | | | | | |   |   | | | | | | | | | | | | |   | |
         q8: -reg2[1]-|-X-|-X-@-|-|-|-|-|-|---|---|-|-|-|-|-|-|-|-|-@-@-X-|---X-|-
                      | | | | | | | | | | |   |   | | | | | | | | | |   | |   | |
         q9: -reg2[2]-|-X-|-|-|-X-@-|-|-|-|---|---|-|-|-|-|-|-@-@-X-|---|-|---X-|-
                      | | | | | | | | | | |   |   | | | | | | |   | |   | |   | |
        q10: -reg2[3]-|-X-|-|-|-|-|-X-@-|-|---|---|-|-|-@-@-X-|---|-|---|-|---X-|-
                      | | | | | | | | | | |   |   | | | |   | |   | |   | |   | |
        q11: -reg2[4]-|-X-|-|-|-|-|-|-|-X-|---@---@-@-X-|---|-|---|-|---|-|---X-|-
                      | | | | | | | | | | |           | |   | |   | |   | |   | |
        q12: -reg2[5]-@-@-X-@-X-@-X-@-X-@-@-----------@-X---@-X---@-X---@-X---@-@-
    )DIAGRAM");
}

TEST(gen_iadd, clean_workspace_diagram) {
    CircuitBuilder builder;
    auto control = builder.append_register(1)[0];
    auto target = builder.append_register(6);
    auto offset = builder.append_register(6);
    auto clean = builder.reserve_qubits(3);
    gen_iadd(builder, CircuitGenCtx{.clean_workspace = clean}, target, offset, control);
    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
         q0: -reg0[0]-@---------------------------@-@---@---@-----@------------------@------------------@----------------@-
                      |                           | |   |   |     |                  |                  |                |
         q1: -reg1[0]-|---@-----------------------|-X-@-X-@-|-----|------------------|------------------|----------@-----X-
                      |   |                       | | | | | |     |                  |                  |          |     |
         q2: -reg1[1]-|---|-X---@-----------------|-|-|-|-|-|-----|------------------|------------@-----X-X--------|-----|-
                      |   | |   |                 | | | | | |     |                  |            |     | |        |     |
         q3: -reg1[2]-|---|-|---|---X---@---------|-|-|-|-|-|-----|------------@-----X-X----------|-----|-|--------|-----|-
                      |   | |   |   |   |         | | | | | |     |            |     | |          |     | |        |     |
         q4: -reg1[3]-|---|-|---|---|---|---X-@---|-|-|-|-|-|---@-X-X----------|-----|-|----------|-----|-|--------|-----|-
                      |   | |   |   |   |   | |   | | | | | |   | | |          |     | |          |     | |        |     |
         q5: -reg1[4]-|---|-|---|---|---|---|-|-X-|-@-|-@-|-X-X-|-|-|----------|-----|-|----------|-----|-|--------|-----|-
                      |   | |   |   |   |   | | | |   |   | | | | | |          |     | |          |     | |        |     |
         q6: -reg1[5]-X---|-|---|---|---|---|-|-|-X---X---X-|-|-|-|-|----------|-----|-|----------|-----|-|--------|-----|-
                      |   | |   |   |   |   | | | |   |   | | | | | |          |     | |          |     | |        |     |
         q7: -reg2[0]-|---@-|---|---|---|---|-|-|-|---|---|-|-|-|-|-|----------|-----|-|----------|-----|-|--------Z**b0-@-
                      |   | |   |   |   |   | | | |   |   | | | | | |          |     | |          |     | |
         q8: -reg2[1]-|---|-X---@---|---|---|-|-|-|---|---|-|-|-|-|-|----------|-----|-|----------Z**b0-@-X----------------
                      |   | |   |   |   |   | | | |   |   | | | | | |          |     | |                  |
         q9: -reg2[2]-|---|-|---|---X---@---|-|-|-|---|---|-|-|-|-|-|----------Z**b0-@-X------------------|----------------
                      |   | |   |   |   |   | | | |   |   | | | | | |                  |                  |
        q10: -reg2[3]-|---|-|---|---|---|---X-@-|-|---|---|-|-|-@-@-X------------------|------------------|----------------
                      |   | |   |   |   |   | | | |   |   | | | |   |                  |                  |
        q11: -reg2[4]-|---|-|---|---|---|---|-|-X-|---@---@-@-X-|---|------------------|------------------|----------------
                      |   | |   |   |   |   | | | |           | |   |                  |                  |
        q12: -reg2[5]-@---|-|---|---|---|---|-|-|-|-----------|-|---|------------------|------------------|----------------
                          | |   |   |   |   | | | |           | |   |                  |                  |
        q13:          |0>-X-@---|-X-@---|-X-@-X-@-@-----------@-X---@-X----------------@-X----------------@-HMR=b0
                                | |     | |                           |                  |
        q14:                |0>-X-@-----|-|---------------------------|------------------@-HMR=b0
                                        | |                           |
        q15:                        |0>-X-@---------------------------@-HMR=b0
    )DIAGRAM");
}

TEST(gen_add, fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t num_target = rng() % 8;
        size_t num_offset = rng() % 8;
        size_t num_dirty = rng() % 2;
        int min_clean = std::max(0, (int)num_target - (int)num_offset - (int)(num_dirty != 0));
        size_t num_clean = rng() % 7 + min_clean;

        auto target = builder.append_register(num_target, "target");
        auto offset = builder.append_register(num_offset, "offset");
        auto control = builder.append_register(1, "control")[0];
        auto clean = builder.append_register(num_clean, "@clean");
        auto dirty = builder.append_register(num_dirty, "@dirty");
        gen_iadd(
            builder, CircuitGenCtx{.clean_workspace = clean}.with_more_dirty_qubits(dirty), target, offset, control);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if (sample["control"]) {
            sample["target"] += sample["offset"];
        }
    });

    fuzzer.fuzz(100, 256);
}

TEST(gen_sub, fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t num_target = rng() % 8;
        size_t num_offset = rng() % 8;
        size_t num_dirty = rng() % 2;
        int min_clean = std::max(0, (int)num_target - (int)num_offset - (int)(num_dirty != 0));
        size_t num_clean = rng() % 7 + min_clean;

        auto target = builder.append_register(num_target, "target");
        auto offset = builder.append_register(num_offset, "offset");
        auto control = builder.append_register(1, "control")[0];
        auto clean = builder.append_register(num_clean, "@clean");
        auto dirty = builder.append_register(num_dirty, "@dirty");
        gen_isub(
            builder, CircuitGenCtx{.clean_workspace = clean}.with_more_dirty_qubits(dirty), target, offset, control);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if (sample["control"]) {
            sample["target"] -= sample["offset"];
        }
    });

    fuzzer.fuzz(100, 256);
}
