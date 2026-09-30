#include "kickmix/sim/sim.h"

#include "gtest/gtest.h"

#include "kickmix/circuit/circuit.h"
#include "kickmix/simd/simd.h"
#include "kickmix/util/circuit_testing.test.h"
#include "test.test.h"

using namespace kickmix;

TEST(sim, populate_register_buffer) {
    Circuit circuit(R"CIRCUIT(
        APPEND_TO_REGISTER q0 r0
        APPEND_TO_REGISTER q1 r0
        APPEND_TO_REGISTER q2 r0
        APPEND_TO_REGISTER q3 r0
        APPEND_TO_REGISTER q4 r0
        APPEND_TO_REGISTER q5 r1
        APPEND_TO_REGISTER q6 r0
        APPEND_TO_REGISTER q7 r0
    )CIRCUIT");
    Sim<b64, false> sim(externally_seeded_rng());
    sim.configure_for(circuit);
    ASSERT_EQ(sim.registers.size(), 2);
    ASSERT_EQ(sim.registers[0].contents.size(), 7);
    ASSERT_EQ(sim.registers[1].contents.size(), 1);
    ASSERT_EQ(sim.register_buffers.size(), 64);
    ASSERT_EQ(sim.register_buffers[0][0].num_bits, 7);
    ASSERT_EQ(sim.register_buffers[0][1].num_bits, 1);
    ASSERT_EQ(sim.register_buffers[63][0].num_bits, 7);
    ASSERT_EQ(sim.register_buffers[63][1].num_bits, 1);

    sim.copy_bit_packed_state_into_register_buffer();
    for (size_t shot_idx = 0; shot_idx < sim.BATCH_SIZE; shot_idx++) {
        EXPECT_EQ(sim.register_buffers[shot_idx][0], 0);
        EXPECT_EQ(sim.register_buffers[shot_idx][1], 0);
    }

    sim.qubit_span()[6].v[0] = 4;
    sim.copy_bit_packed_state_into_register_buffer();
    for (size_t shot_idx = 0; shot_idx < sim.BATCH_SIZE; shot_idx++) {
        EXPECT_EQ(sim.register_buffers[shot_idx][0], shot_idx == 2 ? (1 << 5) : 0);
        EXPECT_EQ(sim.register_buffers[shot_idx][1], 0);
    }
}

TEST(sim, populate_register_buffer_mixed) {
    Circuit circuit(R"CIRCUIT(
        APPEND_TO_REGISTER q0 r0
        APPEND_TO_REGISTER q1 r0
        APPEND_TO_REGISTER q2 r0
        APPEND_TO_REGISTER b0 r0
        APPEND_TO_REGISTER q4 r0
        APPEND_TO_REGISTER q5 r1
        APPEND_TO_REGISTER q6 r0
        APPEND_TO_REGISTER q7 r0
    )CIRCUIT");
    Sim<b64, false> sim(externally_seeded_rng());
    sim.configure_for(circuit);
    ASSERT_EQ(sim.registers.size(), 2);
    ASSERT_EQ(sim.registers[0].contents.size(), 7);
    ASSERT_EQ(sim.registers[1].contents.size(), 1);
    ASSERT_EQ(sim.register_buffers.size(), 64);
    ASSERT_EQ(sim.register_buffers[0][0].num_bits, 7);
    ASSERT_EQ(sim.register_buffers[0][1].num_bits, 1);
    ASSERT_EQ(sim.register_buffers[63][0].num_bits, 7);
    ASSERT_EQ(sim.register_buffers[63][1].num_bits, 1);

    sim.copy_bit_packed_state_into_register_buffer();
    for (size_t shot_idx = 0; shot_idx < sim.BATCH_SIZE; shot_idx++) {
        EXPECT_EQ(sim.register_buffers[shot_idx][0], 0);
        EXPECT_EQ(sim.register_buffers[shot_idx][1], 0);
    }

    sim.qubit_span()[6].v[0] = 4;
    sim.copy_bit_packed_state_into_register_buffer();
    for (size_t shot_idx = 0; shot_idx < sim.BATCH_SIZE; shot_idx++) {
        EXPECT_EQ(sim.register_buffers[shot_idx][0], shot_idx == 2 ? (1 << 5) : 0);
        EXPECT_EQ(sim.register_buffers[shot_idx][1], 0);
    }
}

TEST(sim, populate_register_buffer_128) {
    Circuit circuit(R"CIRCUIT(
        APPEND_TO_REGISTER q0 r0
        APPEND_TO_REGISTER q1 r0
        APPEND_TO_REGISTER q2 r0
        APPEND_TO_REGISTER q3 r0
        APPEND_TO_REGISTER q4 r0
        APPEND_TO_REGISTER q5 r1
        APPEND_TO_REGISTER q6 r0
        APPEND_TO_REGISTER q7 r0
    )CIRCUIT");
    Sim<b128, false> sim(externally_seeded_rng());
    sim.configure_for(circuit);
    ASSERT_EQ(sim.registers.size(), 2);
    ASSERT_EQ(sim.registers[0].contents.size(), 7);
    ASSERT_EQ(sim.registers[1].contents.size(), 1);
    ASSERT_EQ(sim.register_buffers.size(), 128);
    ASSERT_EQ(sim.register_buffers[0][0].num_bits, 7);
    ASSERT_EQ(sim.register_buffers[0][1].num_bits, 1);
    ASSERT_EQ(sim.register_buffers[63][0].num_bits, 7);
    ASSERT_EQ(sim.register_buffers[63][1].num_bits, 1);

    sim.copy_bit_packed_state_into_register_buffer();
    for (size_t shot_idx = 0; shot_idx < sim.BATCH_SIZE; shot_idx++) {
        EXPECT_EQ(sim.register_buffers[shot_idx][0], 0);
        EXPECT_EQ(sim.register_buffers[shot_idx][1], 0);
    }

    sim.qubit_span()[6].v[0] = 1 << 2;
    sim.qubit_span()[1].v[1] = 1 << (67 - 64);
    sim.copy_bit_packed_state_into_register_buffer();
    for (size_t shot_idx = 0; shot_idx < sim.BATCH_SIZE; shot_idx++) {
        EXPECT_EQ(sim.register_buffers[shot_idx][0], shot_idx == 2 ? (1 << 5) : shot_idx == 67 ? (1 << 1) : 0);
        EXPECT_EQ(sim.register_buffers[shot_idx][1], 0);
    }
}

TEST(sim, supports_all_gates) {
    Circuit circuit = Circuit(circuit_with_every_operation());
    Sim<b64, false> sim(std::mt19937_64{0});
    sim.configure_for(circuit);

    testing::internal::CaptureStderr();  // The debug instructions can produce stderr output.
    sim.apply(circuit);
}

TEST(sim, SimInitInstruction_from_str_many) {
    ASSERT_THROW({ SimInitInstruction::from_str_many("rr"); }, std::invalid_argument);
    ASSERT_THROW({ SimInitInstruction::from_str_many("r="); }, std::invalid_argument);
    ASSERT_THROW({ SimInitInstruction::from_str_many("r<"); }, std::invalid_argument);
    ASSERT_THROW({ SimInitInstruction::from_str_many("r0="); }, std::invalid_argument);
    ASSERT_THROW({ SimInitInstruction::from_str_many("r0<"); }, std::invalid_argument);
    ASSERT_EQ(
        SimInitInstruction::from_str("r1=2"),
        (SimInitInstruction{
            .value = FixedWidthInt("2"),
            .randomize = false,
            .target_register = RegisterId(1),
        }));
}

TEST(sim, count_instructions) {
    Sim<b64, true> sim(INDEPENDENT_TEST_RNG());
    Circuit circuit(R"CIRCUIT(
        HMR q0 b0
        HMR q0 b1
        PUSH_CONDITION if b0
        CCX q1 q2 q3
        CCZ q1 q2 q3 if b1
        POP_CONDITION
        CX q1 q2 if b1
    )CIRCUIT");
    sim.configure_for(circuit);
    for (size_t k = 0; k < 100; k++) {
        sim.clear_for_shot();
        sim.apply(circuit);
    }
    size_t ccx = 0;
    size_t ccz = 0;
    size_t cx = 0;
    for (size_t k = 0; k < 64; k++) {
        ccx += sim.new_op_counters[(size_t)OpType::CCX].compute_total(k);
        ccx += sim.new_op_counters[(size_t)OpType::CCX_IF].compute_total(k);
        ccz += sim.new_op_counters[(size_t)OpType::CCZ].compute_total(k);
        ccz += sim.new_op_counters[(size_t)OpType::CCZ_IF].compute_total(k);
        cx += sim.new_op_counters[(size_t)OpType::CX].compute_total(k);
        cx += sim.new_op_counters[(size_t)OpType::CX_IF].compute_total(k);
    }
    EXPECT_TRUE(3200 - 200 < ccx && ccx < 3200 + 200) << "ccx=" << ccx;
    EXPECT_TRUE(1600 - 200 < ccz && ccz < 1600 + 200) << "ccz=" << ccz;
    EXPECT_TRUE(3200 - 200 < cx && cx < 3200 + 200) << "cx=" << cx;
}

TEST(sim, hmr_condition) {
    Sim<b64, false> sim(INDEPENDENT_TEST_RNG());
    Circuit circuit(R"CIRCUIT(
        BIT_STORE0 b0
        X q0
        BIT_STORE1 b1
        HMR q0 b1 if b0
        X q0
    )CIRCUIT");
    sim.configure_for(circuit);
    sim.clear_for_shot();
    sim.apply(circuit);
    ASSERT_EQ(sim.qubit_span()[0], b64{});
    ASSERT_EQ(sim.bit_span()[0], b64{});
    ASSERT_EQ(sim.bit_span()[1], ~b64{});
    ASSERT_EQ(sim.global_phase_ref(), b64{});
}

TEST(sim, reset_condition) {
    Sim<b64, false> sim(INDEPENDENT_TEST_RNG());
    Circuit circuit(R"CIRCUIT(
        BIT_STORE0 b0
        X q0
        R q0 if b0
        X q0
    )CIRCUIT");
    sim.configure_for(circuit);
    sim.clear_for_shot();
    sim.apply(circuit);
    ASSERT_EQ(sim.qubit_span()[0], b64{});
    ASSERT_EQ(sim.bit_span()[0], b64{});
    ASSERT_EQ(sim.global_phase_ref(), b64{});
}

TEST(sim, safe_read_val_bounds) {
    Sim<b64, false> sim(INDEPENDENT_TEST_RNG());
    sim.num_qubits = 2;
    sim.num_bits = 2;
    sim.state_block.assign(5 + 4, ~b64{});
    ASSERT_EQ(sim.safe_read_val(QubitId(0)), ~b64{});
    ASSERT_EQ(sim.safe_read_val(QubitId(1)), ~b64{});
    ASSERT_EQ(sim.safe_read_val(QubitId(2)), b64{});
    ASSERT_EQ(sim.safe_read_val(BitId(0)), ~b64{});
    ASSERT_EQ(sim.safe_read_val(BitId(1)), ~b64{});
    ASSERT_EQ(sim.safe_read_val(BitId(2)), b64{});
}

TEST(sim, z_pow) {
    Sim<b64, false> sim(INDEPENDENT_TEST_RNG());
    Circuit circuit(R"CIRCUIT(
        Z_POW q0 0.125
    )CIRCUIT");
    sim.configure_for(circuit);
    sim.clear_for_shot();
    sim.qubit_span()[0].set_bit(0, false);
    sim.qubit_span()[0].set_bit(1, true);
    sim.apply(circuit);
    ASSERT_EQ(sim.angles[0], FixedPrecisionAngle128::from_half_turns_exact_double(0.0));
    ASSERT_EQ(sim.angles[1], FixedPrecisionAngle128::from_half_turns_exact_double(0.125));
}

TEST(sim, z_pow_if) {
    Sim<b64, false> sim(INDEPENDENT_TEST_RNG());
    Circuit circuit(R"CIRCUIT(
        Z_POW q0 0.125 if b0
    )CIRCUIT");
    sim.configure_for(circuit);
    sim.clear_for_shot();
    sim.qubit_span()[0] = b64::from_u32_broadcast(0b0101);
    sim.bit_span()[0] = b64::from_u32_broadcast(0b0011);
    sim.apply(circuit);
    ASSERT_EQ(sim.angles[0], FixedPrecisionAngle128::from_half_turns_exact_double(0.125));
    ASSERT_EQ(sim.angles[1], FixedPrecisionAngle128::from_half_turns_exact_double(0.0));
    ASSERT_EQ(sim.angles[2], FixedPrecisionAngle128::from_half_turns_exact_double(0.0));
    ASSERT_EQ(sim.angles[3], FixedPrecisionAngle128::from_half_turns_exact_double(0.0));
}
