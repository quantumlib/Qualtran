#include "gen_lookup.h"

#include "gtest/gtest.h"

#include "kickmix/sim/fuzzer.h"
#include "kickmix/util/circuit_testing.test.h"
#include "test.test.h"

using namespace kickmix;

TEST(gen_lookup, diagram) {
    CircuitBuilder builder;
    auto address = builder.append_register(3);
    auto output = builder.append_register(2);
    auto data = builder.append_classical_register_qcarray_result((2 << 3) - 4);
    auto clean = builder.reserve_qubits(address.size());
    gen_lookup(builder, CircuitGenCtx{.clean_workspace = clean}, data, address, output, 'X', true);

    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
                     reg2[0]=b0 reg2[9]=b9
        q0: -reg0[0]-------------------------------------X---@-X---------------------Z**b12-X---@-X---------------------Z**b12----------------------X---@-X----------------------Z**b12-------------------------------
                     reg2[1]=b1 reg2[10]=b10                 |                       |          |                       |                               |                        |
        q1: -reg0[1]-------------------------------X---@-X---|-----------------------|----------|-----------------------|--------------Z**b12-X---@-X---|------------------------|--------------Z**b12----------------
                     reg2[2]=b2 reg2[11]=b11           |     |                       |          |                       |              |          |     |                        |              |
        q2: -reg0[2]-------------------------X---@-X---|-----|-----------------------|----------|-----------------------|--------------|----------|-----|------------------------|--------------|--------------Z**b12-
                     reg2[3]=b3                  |     |     |                       |          |                       |              |          |     |                        |              |
        q3: -reg1[0]-----------------------------|-----|-----|-X**b0---X**b2---------|----------|-X**b4---X**b6---------|--------------|----------|-----|-X**b8---X**b10---------|--------------|---------------------
                     reg2[4]=b4                  |     |     | |       |             |          | |       |             |              |          |     | |       |              |              |
        q4: -reg1[1]-----------------------------|-----|-----|-X**b1---X**b3---------|----------|-X**b5---X**b7---------|--------------|----------|-----|-X**b9---X**b11---------|--------------|---------------------
                     reg2[5]=b5                  |     |     | |       |             |          | |       |             |              |          |     | |       |              |              |
        q5:                                  |0>-X-----@-----|-|-------|-------------|------@---|-|-------|-------------|--------------@------X---@-----|-|-------|--------------|--------------@------HMR=b12
                     reg2[6]=b6                        |     | |       |             |      |   | |       |             |                         |     | |       |              |
        q6:                                        |0>-X-----@-|-----@-|-------------@------X---@-|-----@-|-------------@------HMR=b12        |0>-X-----@-|-----@-|--------------@------HMR=b12
                     reg2[7]=b7                              | |     | |                        | |     | |                                             | |     | |
        q7:                                              |0>-X-@-----X-@-----HMR=b12        |0>-X-@-----X-@-----HMR=b12                             |0>-X-@-----X-@------HMR=b12
                     reg2[8]=b8
    )DIAGRAM");
}

TEST(gen_unlookup, diagram) {
    CircuitBuilder builder;
    auto address = builder.append_register(3);
    auto output = builder.append_register(2);
    auto data = builder.append_classical_register_qcarray_result((2 << 3) - 4);
    auto clean = builder.reserve_qubits(address.size());
    gen_unlookup(builder, CircuitGenCtx{.clean_workspace = clean}, data, address, output);

    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
                     reg2[0]=b0 reg2[8]=b8   b16=0          push_cond if b20          push_cond if b20                                                                                                                                 neg if b20
        q0: -reg0[0]---------------------------------------------------------------------------------------SWAP-X-X---------------Z**b12---Z**b14------------------------Z**b16---Z**b18-------------------------------SWAP-@---------------------
                     reg2[1]=b1 reg2[9]=b9   b17=0          b12^=1 if b0              b12^=1 if b1         |      |               |        |                             |        |                                    |    |
        q1: -reg0[1]---------------------------------------------------------------------------------------|------|-------X---@-X-|--------|--------------Z**b20-X---@-X-|--------|--------------Z**b20----------------|----|---------------------
                     reg2[2]=b2 reg2[10]=b10 b18=0          b13^=1 if b2              b13^=1 if b3         |      |           |   |        |              |          |   |        |              |                     |    |
        q2: -reg0[2]---------------------------------------------------------------------------------------|------|-X---@-X---|---|--------|--------------|----------|---|--------|--------------|--------------Z**b20-|----|---------------------
                     reg2[3]=b3 reg2[11]=b11 b19=0          b14^=1 if b4              b14^=1 if b5         |      |     |     |   |        |              |          |   |        |              |                     |    |
        q3: -reg1[0]-------------------------------HMR=b20                                             |0>-SWAP---@-----|-----|---Z**b13---Z**b15---------|----------|---Z**b17---Z**b19---------|---------------------SWAP-X-HMR=b20
                     reg2[4]=b4 b12=0                       b15^=1 if b6              b15^=1 if b7                      |     |   |        |              |          |   |        |              |
        q4: -reg1[1]---------------------------------------------------------HMR=b20                                    |     |   |        |              |          |   |        |              |
                     reg2[5]=b5 b13=0                       b16^=1 if b8              b16^=1 if b9                      |     |   |        |              |          |   |        |              |
        q5:                                                                                                         |0>-X-----@---|------@-|--------------@------X---@---|------@-|--------------@------HMR=b20
                     reg2[6]=b6 b14=0                       b17^=1 if b10             b17^=1 if b11                           |   |      | |                         |   |      | |
        q6:                                                                                                               |0>-X---@------X-@------HMR=b20        |0>-X---@------X-@------HMR=b20
                     reg2[7]=b7 b15=0                       pop_cond                  pop_cond
    )DIAGRAM");
}

TEST(gen_binary_to_unary, diagram) {
    CircuitBuilder builder;
    auto target = builder.append_register(8);
    gen_binary_to_unary(builder, CircuitGenCtx{.clean_workspace = {}}, target);

    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
        q0: -reg0[0]---------------SWAP-X-X-@-X-@-----@-X-@---------------------
                                   |      | | | |     | | |
        q1: -reg0[1]----------SWAP-SWAP---@-|-|-|---X-|-|-|-@-X-@---------------
                              |             | | |   | | | | | | |
        q2: -reg0[2]-----SWAP-|-------------X-@-X-@-|-|-|-|-|-|-|-@-X-@---------
                         |    |               |   | | | | | | | | | | |
        q3:  reg0[3] |0>-|----SWAP------------@---X-@-|-|-|-|-|-|-|-|-|-------X-
                         |                            | | | | | | | | |       |
        q4:  reg0[4] |0>-|----------------------------X-@-X-|-|-|-|-|-|-@-----|-
                         |                              |   | | | | | | |     |
        q5:  reg0[5] |0>-|------------------------------|---X-@-X-|-|-|-|-@---|-
                         |                              |     |   | | | | |   |
        q6:  reg0[6] |0>-|------------------------------|-----|---X-@-X-|-|-@-|-
                         |                              |     |     |   | | | |
        q7:  reg0[7] |0>-SWAP---------------------------@-----@-----@---X-X-X-@-
    )DIAGRAM");
}

TEST(gen_unary_to_binary, diagram) {
    CircuitBuilder builder;
    auto target = builder.append_register(8);
    gen_unary_to_binary(builder, CircuitGenCtx{.clean_workspace = {}}, target);

    expect_circuit_has_text_diagram(builder.finish_circuit(), R"DIAGRAM(
                                                                                                              neg if b0
        q0: -reg0[0]-SWAP--------------------------------X--------Z**b0--------X--------------------@-------------------
                     |                                   |        |            |                    |
        q1: -reg0[1]-SWAP-SWAP------X--------------------|--------|------------@-----X--------@-----|-------------------
                          |         |                    |        |                  |        |     |
        q2: -reg0[2]------|----SWAP-@-X-X-X--------@-----|--------@------------@-----|--------|-----|-------------------
                          |    |      | | |        |     |                     |     |        |     |
        q3: -reg0[3]------SWAP-|------|-|-|--------|-----|--------X------------Z**b0-X--------Z**b0-X-HMR=b0
                               |      | | |        |     |        |                  |
        q4: -reg0[4]-----------|------|-|-@--------|-----|--------@-----HMR=b0       |
                               |      | |          |     |                           |
        q5: -reg0[5]-----------|------|-@----------|-----@-HMR=b0                    |
                               |      |            |                                 |
        q6: -reg0[6]-----------|------@---@-HMR=b0 |                                 |
                               |          |        |                                 |
        q7: -reg0[7]-----------SWAP-------X--------Z**b0-----------------------------@-HMR=b0
    )DIAGRAM");
}

TEST(gen_binary_to_unary, fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 6;
        auto target = builder.append_register(1 << n, "target");
        gen_binary_to_unary(builder, CircuitGenCtx{.clean_workspace = {}}, target);
    });

    fuzzer.use_input_sampler([](InputSample &sample, std::mt19937_64 &rng) {
        sample["target"].words[0] = rng() % sample["target"].num_bits;
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        uint64_t k = sample["target"].words[0];
        sample["target"].clear_to_zero();
        sample["target"].bit_ref(k) = true;
    });

    fuzzer.fuzz(10, 64);
}

TEST(gen_unary_to_binary, fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 6;
        auto target = builder.append_register(1 << n, "target");
        gen_binary_to_unary(builder, CircuitGenCtx{.clean_workspace = {}}, target);
        gen_unary_to_binary(builder, CircuitGenCtx{.clean_workspace = {}}, target);
    });

    fuzzer.use_input_sampler([](InputSample &sample, std::mt19937_64 &rng) {
        sample["target"].words[0] = rng() % sample["target"].num_bits;
    });

    fuzzer.fuzz(10, 64);
}

TEST(gen_lookup, fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t w = rng() % 5;
        size_t n = rng() % 5;
        auto data = builder.append_classical_register_qcarray_result(w << n, "data");
        auto address = builder.append_register(n, "address");
        auto output = builder.append_register(w, "output");
        auto clean = builder.append_register(address.size(), "@clean");
        gen_lookup(builder, CircuitGenCtx{.clean_workspace = clean}, data, address, output, 'X', true);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        size_t n = sample["address"].num_bits;
        size_t w = sample["data"].num_bits >> n;
        size_t a = (uint64_t)sample["address"];
        for (size_t k = 0; k < w; k++) {
            sample["output"].bit_ref(k) ^= sample["data"][a * w + k];
        }
    });

    fuzzer.fuzz(10, 64);
}

TEST(gen_lookup, fuzz_z) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t w = rng() % 5;
        size_t n = rng() % 5;
        auto data = builder.append_classical_register_qcarray_result(w << n, "data");
        auto address = builder.append_register(n, "address");
        auto output = builder.append_register(w, "output");
        auto clean = builder.append_register(address.size(), "@clean");
        gen_lookup(builder, CircuitGenCtx{.clean_workspace = clean}, data, address, output, 'Z', true);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        size_t n = sample["address"].num_bits;
        size_t w = sample["data"].num_bits >> n;
        size_t a = (uint64_t)sample["address"];
        for (size_t k = 0; k < w; k++) {
            sample.phase_half_turns += sample["output"][k] && sample["data"][a * w + k];
        }
    });

    fuzzer.fuzz(10, 64);
}

TEST(gen_unlookup, fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t w = rng() % 5;
        size_t n = rng() % 5;
        auto data = builder.append_classical_register_qcarray_result(w << n, "data");
        auto address = builder.append_register(n, "address");
        auto output = builder.append_register(w, "output");
        auto clean = builder.append_register(address.size(), "@clean");
        gen_lookup(builder, CircuitGenCtx{.clean_workspace = clean}, data, address, output, 'X', true);
        gen_unlookup(builder, CircuitGenCtx{.clean_workspace = clean}, data, address, output);
    });

    fuzzer.use_input_sampler([&](InputSample &sample, std::mt19937_64 &rng) {
        sample["address"].randomize(rng);
        sample["data"].randomize(rng);
        sample["output"].clear_to_zero();
    });
    fuzzer.use_output_sampler([](OutputSample &sample) {
    });

    fuzzer.fuzz(10, 64);
}
