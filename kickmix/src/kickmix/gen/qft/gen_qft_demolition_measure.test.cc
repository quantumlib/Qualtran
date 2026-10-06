#include "gen_qft_demolition_measure.h"

#include "gtest/gtest.h"

#include "kickmix/sim/sim.h"
#include "kickmix/simd/simd.h"
#include "kickmix/util/circuit_testing.test.h"
#include "test.test.h"

using namespace kickmix;

TEST(gen_qft_demolition_measure, diagram) {
    CircuitBuilder builder;
    size_t n = 4;
    auto target = builder.append_register(n, "target");
    auto output = builder.append_classical_register(n, "output");
    CircuitGenCtx ctx{};
    gen_qft_demolition_measure(builder, ctx, target, output, false);
    auto c = builder.finish_circuit();
    expect_circuit_has_text_diagram(c, R"DIAGRAM(
                       output[0]=b0
        q0: -target[0]--------------SWAP-HMR=b0
                       output[1]=b1 |           if(b0)
        q1: -target[1]--------------|----SWAP---Z^0.5--HMR=b1
                       output[2]=b2 |    |                    if(b0) if(b1)
        q2: -target[2]--------------|----SWAP-----------------Z^0.25-Z^0.5--HMR=b2
                       output[3]=b3 |                                              if(b0)  if(b1) if(b2)
        q3: -target[3]--------------SWAP-------------------------------------------Z^0.125-Z^0.25-Z^0.5--HMR=b3
    )DIAGRAM");
}

TEST(gen_qft_demolition_measure, diagram_inverse) {
    CircuitBuilder builder;
    size_t n = 4;
    auto target = builder.append_register(n, "target");
    auto output = builder.append_classical_register(n, "output");
    CircuitGenCtx ctx{};
    gen_qft_demolition_measure(builder, ctx, target, output, true);
    auto circuit = builder.finish_circuit();
    expect_circuit_has_text_diagram(circuit, R"DIAGRAM(
                       output[0]=b0
        q0: -target[0]--------------SWAP-HMR=b0
                       output[1]=b1 |           if(b0)
        q1: -target[1]--------------|----SWAP---Z^1.5--HMR=b1
                       output[2]=b2 |    |                    if(b0) if(b1)
        q2: -target[2]--------------|----SWAP-----------------Z^1.75-Z^1.5--HMR=b2
                       output[3]=b3 |                                              if(b0)  if(b1) if(b2)
        q3: -target[3]--------------SWAP-------------------------------------------Z^1.875-Z^1.75-Z^1.5--HMR=b3
    )DIAGRAM");
}

TEST(gen_qft_demolition_measure, behavior) {
    for (size_t n = 0; n < 25; n++) {
        CircuitBuilder builder;
        auto target = builder.append_register(n, "target");
        auto output = builder.append_classical_register(n, "output");
        CircuitGenCtx ctx{};
        gen_qft_demolition_measure(builder, ctx, target, output, false);
        auto circuit = builder.finish_circuit();

        Sim<b64, false> sim(INDEPENDENT_TEST_RNG());
        sim.configure_for(circuit);
        sim.clear_for_shot();
        for (size_t k = 0; k < sim.BATCH_SIZE; k++) {
            sim.register_buffers[k][0].randomize(sim.rng);
            sim.register_buffers[k][1].clear_to_zero();
        }
        sim.copy_register_buffer_into_bit_packed_state();
        std::swap(sim.register_buffers, sim.register_buffers2);
        sim.apply(circuit);
        sim.copy_bit_packed_state_into_register_buffer();
        for (size_t k = 0; k < sim.BATCH_SIZE; k++) {
            const auto &old_target = sim.register_buffers2[k][0];
            const auto &new_target = sim.register_buffers[k][0];
            const auto &old_output = sim.register_buffers2[k][1];
            const auto &new_output = sim.register_buffers[k][1];
            auto actual_phase = sim.read_shot_phase(k);
            auto expected_phase = FixedPrecisionAngle128::from_power_of_2_half_turns(1 - (int)n);
            if (n > 0) {
                expected_phase *= old_target.words[0];
                expected_phase *= new_output.words[0];
            }

            ASSERT_FALSE(new_target.non_zero());
            ASSERT_FALSE(old_output.non_zero());
            EXPECT_EQ(actual_phase, expected_phase) << k;
        }
    }
}
