#ifndef KICKMIX_SIM_H
#define KICKMIX_SIM_H

#include <iostream>
#include <kickmix/id/qubit_or_bit_or_bool.h>
#include <random>
#include <span>
#include <vector>

#include "kickmix/circuit/circuit.h"
#include "kickmix/id/register_id.h"
#include "kickmix/util/fixed_width_int.h"
#include "kickmix/util/pop_counter.h"
#include "kickmix/util/xoshiro_256_plusplus.h"

namespace kickmix {

template <uint64_t mask, uint64_t shift>
inline void inplace_transpose_64x64_pass(uint64_t *data) {
    for (size_t k = 0; k < 64; k++) {
        if (k & shift) {
            continue;
        }
        uint64_t &x = data[k];
        uint64_t &y = data[k + shift];
        uint64_t a = x & mask;
        uint64_t b = x & ~mask;
        uint64_t c = y & mask;
        uint64_t d = y & ~mask;
        x = a | (c << shift);
        y = (b >> shift) | d;
    }
}

inline void inplace_transpose_64x64(uint64_t *data) {
    inplace_transpose_64x64_pass<0x5555555555555555UL, 1>(data);
    inplace_transpose_64x64_pass<0x3333333333333333UL, 2>(data);
    inplace_transpose_64x64_pass<0x0F0F0F0F0F0F0F0FUL, 4>(data);
    inplace_transpose_64x64_pass<0x00FF00FF00FF00FFUL, 8>(data);
    inplace_transpose_64x64_pass<0x0000FFFF0000FFFFUL, 16>(data);
    inplace_transpose_64x64_pass<0x00000000FFFFFFFFUL, 32>(data);
}

struct SimInitInstruction {
    FixedWidthInt value;
    bool randomize;
    RegisterId target_register;

    bool operator==(const SimInitInstruction &other) const {
        return value == other.value && randomize == other.randomize && target_register == other.target_register;
    }
    static std::vector<SimInitInstruction> from_str_many(std::string_view text);
    static SimInitInstruction from_str(std::string_view text);
};
std::ostream &operator<<(std::ostream &out, const SimInitInstruction &rhs);

template <typename TWord>
struct EmptyCounters {
    PopCounter<TWord> *begin() const {
        return nullptr;
    }
    PopCounter<TWord> *end() const {
        return nullptr;
    }
};

template <typename TWord, bool count_shots>
struct Sim {
    using ubits_t = TWord;
    constexpr static size_t BATCH_SIZE = sizeof(TWord) * 8;
    constexpr static size_t SHOT_WORD_COUNT = BATCH_SIZE / 64;

    std::vector<TWord> state_block;
    std::vector<TWord> condition_stack;
    size_t num_qubits;
    size_t num_bits;
    std::mt19937_64 rng;
    std::array<uint64_t, 64> transpose_buffer;
    std::vector<FixedPrecisionAngle128> angles;
    std::vector<RegisterData> registers;
    bool ignore_debug_print_operations = false;

    // shot_index -> register_index -> word_index
    std::array<std::vector<FixedWidthInt>, BATCH_SIZE> register_buffers;
    std::array<std::vector<FixedWidthInt>, BATCH_SIZE> register_buffers2;

    std::conditional_t<count_shots, std::array<PopCounter<TWord>, 256>, EmptyCounters<TWord>> new_op_counters{};

    inline std::span<TWord> qubit_span() {
        return {state_block.data() + 5, num_qubits};
    }
    inline std::span<TWord> bit_span() {
        return {state_block.data() + 5 + num_qubits, num_bits};
    }
    inline TWord &global_phase_ref() {
        return state_block[0];
    }
    inline std::span<TWord> rng_state_span() {
        return {&state_block[1], 4};
    }

    inline std::span<const TWord> qubit_span() const {
        return {state_block.data() + 5, num_qubits};
    }
    inline std::span<const TWord> bit_span() const {
        return {state_block.data() + 5 + num_qubits, num_bits};
    }
    inline const TWord &global_phase_ref() const {
        return state_block[0];
    }
    inline std::span<const TWord> rng_state_span() const {
        return {&state_block[1], 4};
    }

    FixedPrecisionAngle128 read_shot_phase(size_t shot_index) const {
        auto result = angles[shot_index];
        if (global_phase_ref().bit(shot_index)) {
            result = result.rotated180();
        }
        return result;
    }

    explicit Sim(std::mt19937_64 &&rng)
        : state_block(5), num_qubits(0), num_bits(0), rng(rng), transpose_buffer{}, angles(BATCH_SIZE) {
        state_block[1].randomize(rng);
        state_block[2].randomize(rng);
        state_block[3].randomize(rng);
        state_block[4].randomize(rng);
    }
    Sim(const Sim &sim) = default;
    Sim(Sim &&sim) = default;
    Sim &operator=(Sim &&sim) = default;
    Sim &operator=(const Sim &sim) = default;

    inline TWord safe_read_val(QubitOrBitOrBool id) const {
        if (id.is_qubit()) {
            if (id.untagged_id() >= qubit_span().size()) {
                return TWord{};
            }
            return qubit_span()[id.untagged_id()];
        } else if (id.is_bit()) {
            if (id.untagged_id() >= bit_span().size()) {
                return TWord{};
            }
            return bit_span()[id.untagged_id()];
        } else if ((bool)id) {
            return ~TWord{};
        } else {
            return TWord{};
        }
    }

    inline TWord &val_for(QubitId id) {
        return qubit_span()[id.untagged_id()];
    }
    inline TWord &val_for(BitId id) {
        return bit_span()[id.untagged_id()];
    }
    inline TWord &val_for(QubitOrBit id) {
        if (id.is_qubit()) {
            return val_for((QubitId)id);
        } else {
            return val_for((BitId)id);
        }
    }
    inline TWord &val_for(RegisterId reg, size_t offset) {
        return val_for(registers[reg.id].contents[offset]);
    }

    inline const TWord &val_for(QubitOrBit id) const {
        if (id.is_qubit()) {
            return qubit_span()[id.untagged_id()];
        } else {
            return bit_span()[id.untagged_id()];
        }
    }

    void copy_single_bit_packed_state_into_register_buffer(RegisterId register_id) {
        size_t reg_idx = register_id.id;
        const auto &reg = registers[reg_idx];
        for (size_t word_idx = 0; word_idx < reg.contents.size(); word_idx += 64) {
            size_t w = word_idx / 64;
            size_t n = std::min(static_cast<size_t>(64), reg.contents.size() - word_idx);

            for (size_t shot_block = 0; shot_block < SHOT_WORD_COUNT; shot_block++) {
                for (size_t i = 0; i < n; i++) {
                    transpose_buffer[i] = val_for(reg.contents[word_idx + i]).v[shot_block];
                }
                for (size_t i = n; i < 64; i++) {
                    transpose_buffer[i] = 0;
                }
                inplace_transpose_64x64(transpose_buffer.data());
                for (size_t shot_idx = 0; shot_idx < 64; shot_idx++) {
                    register_buffers[shot_idx + shot_block * 64][reg_idx].words[w] = transpose_buffer[shot_idx];
                }
            }
        }
    }

    void copy_bit_packed_state_into_register_buffer() {
        for (size_t reg_idx = 0; reg_idx < registers.size(); reg_idx++) {
            copy_single_bit_packed_state_into_register_buffer(RegisterId(reg_idx));
        }
    }

    void copy_single_register_buffer_into_bit_packed_state(RegisterId register_id) {
        size_t reg_idx = register_id.id;
        const auto &reg = registers[reg_idx];
        for (size_t word_idx = 0; word_idx < reg.contents.size(); word_idx += 64) {
            size_t w = word_idx / 64;
            size_t n = std::min(static_cast<size_t>(64), reg.contents.size() - word_idx);

            for (size_t shot_block = 0; shot_block < SHOT_WORD_COUNT; shot_block++) {
                for (size_t shot_idx = 0; shot_idx < 64; shot_idx++) {
                    transpose_buffer[shot_idx] = register_buffers[shot_idx + shot_block * 64][reg_idx].words[w];
                }
                inplace_transpose_64x64(transpose_buffer.data());
                for (size_t i = 0; i < n; i++) {
                    val_for(reg.contents[word_idx + i]).v[shot_block] = transpose_buffer[i];
                }
            }
        }
    }

    void copy_register_buffer_into_bit_packed_state() {
        for (size_t reg_idx = 0; reg_idx < registers.size(); reg_idx++) {
            copy_single_register_buffer_into_bit_packed_state(RegisterId(reg_idx));
        }
    }

    void ensure_big_enough_state_for(size_t min_num_qubits, size_t min_num_bits) {
        if (min_num_qubits <= num_qubits && min_num_bits <= num_bits) {
            return;
        }

        // Multiplicative resizing ensures no quadratic overhead.
        min_num_qubits = std::max(num_qubits * 2, min_num_qubits);
        min_num_bits = std::max(num_bits * 2, min_num_bits);

        std::span<TWord> old_bits = bit_span();
        std::span<TWord> old_qubits = qubit_span();
        std::vector<TWord> old_state = std::move(state_block);
        state_block = std::vector<TWord>();
        state_block.resize(min_num_qubits + min_num_bits + 5);
        size_t old_num_qubits = num_qubits;
        size_t old_num_bits = num_bits;
        num_qubits = min_num_qubits;
        num_bits = min_num_bits;

        memcpy(state_block.data(), old_state.data(), 5 * sizeof(TWord));
        if (old_num_bits) {
            memcpy(bit_span().data(), old_bits.data(), old_num_bits * sizeof(TWord));
        }
        if (old_num_qubits) {
            memcpy(qubit_span().data(), old_qubits.data(), old_num_qubits * sizeof(TWord));
        }
    }

    void configure_for(const Circuit &circuit) {
        registers = circuit.register_data;
        for (size_t shot_idx = 0; shot_idx < BATCH_SIZE; shot_idx++) {
            register_buffers[shot_idx].resize(registers.size());
            for (size_t reg_idx = 0; reg_idx < registers.size(); reg_idx++) {
                register_buffers[shot_idx][reg_idx].resize_clear(registers[reg_idx].contents.size());
            }
        }
        register_buffers2 = register_buffers;

        num_qubits = circuit.num_qubits;
        num_bits = circuit.num_bits;
        state_block.resize(num_qubits + num_bits + 5);
        clear_for_shot();
    }

    void apply(const Circuit &circuit) {
        Xoshiro256PlusPlus<TWord> rng_state;
        memcpy(&rng_state, rng_state_span().data(), sizeof(rng_state));
        const auto qubits = qubit_span();
        const auto bits = bit_span();
        TWord current_base_condition = condition_stack.empty() ? ~TWord{} : condition_stack.back();

        auto *data_qqq0 = circuit.qqq0;
        auto *data_qqq1 = circuit.qqq1;
        auto *data_qqq2 = circuit.qqq2;
        auto *data_qq0 = circuit.qq0;
        auto *data_qq1 = circuit.qq1;
        auto *data_q0 = circuit.q0;
        auto *data_bit_cond = circuit.bc;
        auto *data_bit_targ = circuit.b0;
        auto *data_angles = circuit.angles;
        TWord phase = global_phase_ref();
        for (size_t k = 0; k < circuit.num_ops; k++) {
            auto kind = circuit.op_types[k];

            TWord cond = current_base_condition;
            switch (kind) {
                case OpType::CCX_IF:
                    cond &= bits[*data_bit_cond++];
                case OpType::CCX:
                    qubits[*data_qqq2++] ^= cond & qubits[*data_qqq1++] & qubits[*data_qqq0++];
                    break;

                case OpType::CX_IF:
                    cond &= bits[*data_bit_cond++];
                case OpType::CX:
                    qubits[*data_qq1++] ^= cond & qubits[*data_qq0++];
                    break;

                case OpType::SWAP_IF:
                    cond &= bits[*data_bit_cond++];
                case OpType::SWAP: {
                    auto q0 = *data_qq0++;
                    auto q1 = *data_qq1++;
                    auto a = qubits[q0];
                    auto b = qubits[q1];
                    a ^= b;
                    b ^= a & cond;
                    a ^= b;
                    qubits[q0] = a;
                    qubits[q1] = b;
                    break;
                }

                case OpType::X_IF:
                    cond &= bits[*data_bit_cond++];
                case OpType::X:
                    qubits[*data_q0++] ^= cond;
                    break;

                case OpType::CCZ_IF:
                    cond &= bits[*data_bit_cond++];
                case OpType::CCZ:
                    phase ^= cond & qubits[*data_qqq0++] & qubits[*data_qqq1++] & qubits[*data_qqq2++];
                    break;

                case OpType::CZ_IF:
                    cond &= bits[*data_bit_cond++];
                case OpType::CZ:
                    phase ^= cond & qubits[*data_qq0++] & qubits[*data_qq1++];
                    break;

                case OpType::Z_POW_IF:
                    cond &= bits[*data_bit_cond++];
                case OpType::Z_POW: {
                    auto angle = *data_angles++;
                    auto mask = qubits[*data_q0++] & cond;
                    for (size_t b = 0; b < BATCH_SIZE; b++) {
                        if (mask.bit(b)) {
                            angles[b] += angle;
                        }
                    }
                    break;
                }

                case OpType::Z_IF:
                    cond &= bits[*data_bit_cond++];
                case OpType::Z:
                    phase ^= cond & qubits[*data_q0++];
                    break;

                case OpType::NEG_IF:
                    cond &= bits[*data_bit_cond++];
                case OpType::NEG:
                    phase ^= cond;
                    break;

                case OpType::HMR_IF:
                    cond &= bits[*data_bit_cond++];
                case OpType::HMR: {
                    auto target_q = *data_q0++;
                    auto target_b = *data_bit_targ++;
                    auto q = qubits[target_q];
                    auto b = bits[target_b];
                    b &= ~cond;
                    b ^= rng_state.next() & cond;
                    phase ^= q & b & cond;
                    q &= ~cond;
                    qubits[target_q] = q;
                    bits[target_b] = b;
                    break;
                }

                case OpType::R_IF:
                    cond &= bits[*data_bit_cond++];
                case OpType::R: {
                    // Dephase as if measuring in X basis and ignoring the result.
                    auto tq = *data_q0++;
                    phase ^= qubits[tq] & rng_state.next() & cond;
                    qubits[tq] &= ~cond;
                    break;
                }

                case OpType::BIT_INVERT_IF:
                    cond &= bits[*data_bit_cond++];
                case OpType::BIT_INVERT:
                    bits[*data_bit_targ++] ^= cond;
                    break;

                case OpType::BIT_STORE0_IF:
                    cond &= bits[*data_bit_cond++];
                case OpType::BIT_STORE0:
                    bits[*data_bit_targ++] &= ~cond;
                    break;

                case OpType::BIT_STORE1_IF:
                    cond &= bits[*data_bit_cond++];
                case OpType::BIT_STORE1:
                    bits[*data_bit_targ++] |= cond;
                    break;

                case OpType::DEBUG_PRINT_C_IF:
                    cond &= bits[*data_bit_cond++];
                case OpType::DEBUG_PRINT_C: {
                    auto b = *data_bit_targ++;
                    if (cond.bit(0) && !ignore_debug_print_operations) {
                        std::cerr << (bits[b].bit(0) ? '1' : '0');
                    }
                    break;
                }
                case OpType::DEBUG_PRINT_Q_IF:
                    cond &= bits[*data_bit_cond++];
                case OpType::DEBUG_PRINT_Q: {
                    auto q = *data_bit_targ++;
                    if (cond.bit(0) && !ignore_debug_print_operations) {
                        std::cerr << (qubits[q].bit(0) ? '1' : '0');
                    }
                    break;
                }
                case OpType::DEBUG_PRINT_EMPTY_IF:
                    cond &= bits[*data_bit_cond++];
                case OpType::DEBUG_PRINT_EMPTY:
                    if (cond.bit(0) && !ignore_debug_print_operations) {
                        std::cerr << " (phase=" << read_shot_phase(0) << ")\n";
                    }
                    break;

                case OpType::PUSH_CONDITION:
                    current_base_condition &= bits[*data_bit_cond++];
                    condition_stack.push_back(current_base_condition);
                    break;

                case OpType::POP_CONDITION:
                    condition_stack.pop_back();
                    current_base_condition = condition_stack.empty() ? ~TWord{} : condition_stack.back();
                    break;

                default: {
                    std::stringstream ss;
                    ss << "sim.apply: operation type not implemented: " << kind;
                    throw std::invalid_argument(ss.str());
                }
            }
            if constexpr (count_shots) {
                new_op_counters[(uint8_t)kind].masked_increments(cond);
            }
        }

        global_phase_ref() = phase;
        memcpy(rng_state_span().data(), &rng_state, sizeof(rng_state));
    }

    void clear_for_shot() {
        for (auto &e : qubit_span()) {
            e.clear_to_zero();
        }
        for (auto &e : bit_span()) {
            e.clear_to_zero();
        }
        global_phase_ref().clear_to_zero();
        if (!angles.empty()) {
            memset(angles.data(), 0, sizeof(FixedPrecisionAngle128) * angles.size());
        }
        condition_stack.clear();
    }
};

}  // namespace kickmix

#endif
