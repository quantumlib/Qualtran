#ifndef KICKMIX_CIRCUIT_BUILDER_H
#define KICKMIX_CIRCUIT_BUILDER_H

#include <functional>
#include <span>

#include "circuit_gen_ctx.h"
#include "circuit_pattern.h"
#include "kickmix/circuit/circuit.h"
#include "kickmix/id/bit_or_bool.h"
#include "kickmix/id/qubit_or_true.h"
#include "kickmix/id/qubit_or_xbit_or_xbool.h"
#include "kickmix/id/qubit_or_xzbit_or_xzbool.h"
#include "kickmix/id/register_id.h"
#include "kickmix/id/stride_span_x.h"
#include "kickmix/id/stride_span_xz.h"
#include "kickmix/mem/monotonic_arena.h"

namespace kickmix {

struct CircuitBuilder;

struct RaiiCircuitBuilderPopPushedCondition {
    CircuitBuilder *builder;
    RaiiCircuitBuilderPopPushedCondition(CircuitBuilder *builder);
    ~RaiiCircuitBuilderPopPushedCondition();
    RaiiCircuitBuilderPopPushedCondition(RaiiCircuitBuilderPopPushedCondition &&) noexcept;
    RaiiCircuitBuilderPopPushedCondition &operator=(RaiiCircuitBuilderPopPushedCondition &&) noexcept;
    RaiiCircuitBuilderPopPushedCondition(const RaiiCircuitBuilderPopPushedCondition &) = delete;
    RaiiCircuitBuilderPopPushedCondition &operator=(const RaiiCircuitBuilderPopPushedCondition &) = delete;
};
struct CircuitBuilderRaiiBit {
    CircuitBuilder *builder;
    BitId bit;
    bool is_pushed = false;
    CircuitBuilderRaiiBit(const CircuitBuilderRaiiBit &) = delete;
    CircuitBuilderRaiiBit &operator=(const CircuitBuilderRaiiBit &) = delete;
    CircuitBuilderRaiiBit &operator=(CircuitBuilderRaiiBit &&) = delete;
    CircuitBuilderRaiiBit(CircuitBuilder *builder, BitId bit, bool push);
    CircuitBuilderRaiiBit(CircuitBuilderRaiiBit &&other) noexcept
        : builder(other.builder), bit(other.bit), is_pushed(other.is_pushed) {
        other.builder = nullptr;
        other.is_pushed = false;
    }
    ~CircuitBuilderRaiiBit();
    /// Returns the stored bit and disables the RAII mechanism that returns it to the builder.
    /// The caller takes ownership of the result, and becomes responsible for pushing it back
    /// onto the builder's raii_bits field when they are done with the result.
    BitId disarm() {
        BitId result = bit;
        builder = nullptr;
        is_pushed = false;
        return result;
    }
};

struct CircuitBuilderRaiiXBit {
    CircuitBuilder *builder;
    BitId bit;
    bool is_pushed = false;
    CircuitBuilderRaiiXBit() = delete;
    CircuitBuilderRaiiXBit(const CircuitBuilderRaiiXBit &) = delete;
    CircuitBuilderRaiiXBit &operator=(const CircuitBuilderRaiiXBit &) = delete;
    CircuitBuilderRaiiXBit &operator=(CircuitBuilderRaiiXBit &&) = delete;
    CircuitBuilderRaiiXBit(CircuitBuilder *builder, BitId bit, bool push);
    CircuitBuilderRaiiXBit(CircuitBuilderRaiiXBit &&other) noexcept
        : builder(other.builder), bit(other.bit), is_pushed(other.is_pushed) {
        other.builder = nullptr;
        other.is_pushed = false;
    }
    CircuitBuilderRaiiXBit(CircuitBuilderRaiiBit &&other) noexcept
        : builder(other.builder), bit(other.bit), is_pushed(other.is_pushed) {
        other.builder = nullptr;
        other.is_pushed = false;
    }
    ~CircuitBuilderRaiiXBit();
    operator QubitOrXBitOrXBool() const {
        return QubitOrXBitOrXBool(bit.conjugated_by_h());
    }
};

template <class T>
concept ControlLike =
    std::is_same<T, bool>::value || std::is_same<T, BitId>::value || std::is_same<T, QubitOrTrue>::value ||
    std::is_same<T, QubitId>::value || std::is_same<T, BitOrBool>::value || std::is_same<T, QubitOrBitOrBool>::value;

template <class T>
concept QuantumTargetLike = std::is_same<T, QubitId>::value || std::is_same<T, QubitOrXBitOrXBool>::value ||
                            std::is_same<T, QubitOrMinusState>::value || std::is_same<T, CircuitBuilderRaiiXBit>::value;

template <class T>
concept ClassicalControlLike =
    std::is_same<T, bool>::value || std::is_same<T, BitId>::value || std::is_same<T, BitOrBool>::value;

struct Mark {
    bool enter_else_exit;
    std::string_view name;
    size_t offset;
};
struct RaiiDeferredMark {
    CircuitBuilder *builder;
    Mark deferred;
    RaiiDeferredMark() = delete;
    RaiiDeferredMark(CircuitBuilder *builder, Mark deferred) : builder(builder), deferred(deferred) {
    }
    RaiiDeferredMark(const RaiiDeferredMark &) = delete;
    RaiiDeferredMark(RaiiDeferredMark &&other) noexcept : builder(other.builder), deferred(other.deferred) {
        other.builder = nullptr;
    }
    RaiiDeferredMark &operator=(const RaiiDeferredMark &) = delete;
    RaiiDeferredMark &operator=(RaiiDeferredMark &&) = delete;
    ~RaiiDeferredMark();
};

/// A helper class for building kickmix circuits.
///
/// This class handles the following otherwise-annoying things:
/// 1. Downgrading. For example, ccz(bit, qubit, qubit) becoming a classically-controlled CZ.
/// 2. Broadcasting. For example, cx(qubit, list) becoming a list of CX gates.
/// 3. Allocating. For example, hmr(qubit) returning a different bit each time.
struct CircuitBuilder {
    // Methods for creating register and allocating objects use these values (incrementing them as needed).
    uint32_t next_qubit_id = 0;
    uint32_t next_bit_id = 0;
    uint32_t next_register_id = 0;
    // Available temporarily allocated bits.
    std::vector<BitId> raii_bits;
    // Fields for c_push and c_resolve.
    std::vector<QubitId> c_resolve_qubit_controls;
    std::vector<BitId> c_resolve_bit_controls;
    bool c_resolve_has_false_control = false;
    std::vector<Mark> marks;
    // Determines if validation is skipped, to reduce runtime cost.
    bool skip_validation = false;
    CircuitPattern pattern;
    MutableCircuit mut;

    RaiiDeferredMark raii_mark_block_entry(std::string_view name);
    Circuit finish_circuit() const;
    void write_analysis_svg_to(std::ostream &out, size_t reference_qubit_count);

    void for_each(size_t start, size_t end, const std::function<void(LoopBuilder &out, iota k)> &func);
    void for_each_reversed(size_t start, size_t end, const std::function<void(LoopBuilder &out, iota k)> &func);

    void neg();
    void neg_if(BitId cond);
    void z(QubitId v1);
    void z_if(QubitId v1, BitId cond);
    void cz_if(QubitId v1, QubitId v2, BitId cond);
    void ccz(QubitId control2, QubitId control1, QubitId target);
    void ccz_if(QubitId control2, QubitId control1, QubitId target, BitId cond);
    /// Emits an 'X' instruction.
    ///
    /// An X instruction performs a NOT gate.
    ///
    /// Args:
    ///     target: The qubit to apply the X gate to.
    void x(QubitId target);
    /// Emits an 'X_IF' instruction.
    ///
    /// An X instruction performs a NOT gate.
    ///
    /// Args:
    ///     target: The qubit to apply the X gate to.
    ///     cond: A classical condition bit that determines if the operation happens.
    void x_if(QubitId target, BitId cond);
    /// Emits a 'CX_IF' instruction.
    ///
    /// An CX_IF instruction performs a controlled-not gate conditioned on a classical bit.
    ///
    /// Args:
    ///     control: The control of the controlled-not gate. The condition qubit.
    ///     target: The target of the controlled-not gate. The qubit that gets flipped.
    ///     cond: A classical condition bit that determines if the operation happens.
    void cx_if(QubitId control, QubitId target, BitId cond);
    /// Emits a 'CCX' instruction.
    ///
    /// An CCX instruction performs a controlled-controlled-not gate.
    ///
    /// Args:
    ///     control1: One control of the gate. One of the condition qubits.
    ///     control2: The other control of the gate. One of the condition qubits.
    ///     target: The target of the gate. The qubit that gets flipped.
    void ccx(QubitId control2, QubitId control1, QubitId target);
    /// Emits a 'CCX_IF' instruction.
    ///
    /// An CCX_IF instruction performs a conditioned controlled-controlled-not gate.
    ///
    /// Args:
    ///     control1: One control of the gate. One of the condition qubits.
    ///     control2: The other control of the gate. One of the condition qubits.
    ///     target: The target of the gate. The qubit that gets flipped.
    ///     cond: A classical condition bit that determines if the operation happens.
    void ccx_if(QubitId control2, QubitId control1, QubitId target, BitId cond);
    void reset(QubitId target);
    void reset_if(QubitId target, BitId cond);
    /// Emits an 'HMR' instruction.
    ///
    /// An HMR is an X-basis measurement fused with a Z-basis reset.
    ///
    /// Args:
    ///     target: The qubit to X-measure and Z-reset.
    ///     out: The bit to store the measurement result into.
    void hmr(QubitId target, BitId out);
    /// Emits an 'HMR_IF' instruction.
    ///
    /// An HMR is an X-basis measurement fused with a Z-basis reset.
    ///
    /// Args:
    ///     target: The qubit to X-measure and Z-reset.
    ///     out: The bit to store the measurement result into.
    ///     cond: A classical condition bit that determines if the operation happens.
    void hmr_if(QubitId target, BitId out, BitId cond);
    void push_condition(BitId bit);
    void pop_condition();

    void z_pow(QubitId v1, FixedPrecisionAngle128 exponent);
    void z_pow_if(QubitId v1, FixedPrecisionAngle128 exponent, BitId cond);

    // === Building block operations ===.

    void append_qubit_to_register(std::span<const QubitId> qubits, RegisterId reg);
    void append_bit_to_register(std::span<const BitId> bits, RegisterId reg);
    /// Temporarily allocates a bit id.
    ///
    /// The allocated id is returned to the allocation pool when the result is destructed.
    CircuitBuilderRaiiBit alloc_clean_raii_bit();
    CircuitBuilderRaiiBit alloc_dirty_raii_bit();
    /// Adds a quantum or classical control to be used by c_resolve.
    void c_push(bool control);
    void c_push(BitId control);
    void c_push(QubitId control);
    void c_push(BitOrBool control);
    void c_push(QubitOrTrue control);
    void c_push(QubitOrBitOrBool control);
    /// Performs an X gate or global phase negation controlled by the controls given to c_push.
    ///
    /// Clears the list of pushed controls as part of resolving the operation.
    void c_resolve_exact(QubitOrMinusState target, std::span<const QubitId> clean);
    void c_resolve(const CircuitBuilderRaiiXBit &target, std::span<const QubitId> clean);
    void c_resolve(QubitOrXBitOrXBool target, std::span<const QubitId> clean);
    void c_resolve(QubitId target, std::span<const QubitId> clean);
    std::invalid_argument c_resolve_explain_failure(std::string_view msg) const;

    // === Derived operations ===.
    std::vector<QubitId> append_register(size_t length, std::string_view name = "");
    std::vector<QubitId> reserve_qubits(size_t length);
    std::vector<BitId> reserve_bits(size_t length);
    RegisterId reserve_register();
    std::vector<BitId> append_classical_register(size_t length, std::string_view name = "");
    RegisterId append_classical_register(std::span<const BitId> bits, std::string_view name = "");
    std::vector<QubitOrBitOrBool> append_classical_register_mixed_result(size_t length, std::string_view name = "");
    std::vector<QubitOrBitOrBool> append_register_mixed_result(size_t length, std::string_view name = "");
    array_z append_classical_register_qcarray_result(size_t length, std::string_view name = "");
    array_z append_register_qcarray_result(size_t length, std::string_view name = "");

    /// Applies HMR to the qubit (X-basis measurement + Z-basis reset) and
    /// returns the result as a Raii XBit.
    ///
    /// The 'raii' refers to the fact that the bit's index will not be reused
    /// until the caller allows the return value to be destructed.
    ///
    /// The 'X' in 'XBit' means that the returned value can be passed into most
    /// methods as if it were a qubit as long as that method acts on the X operator
    /// of the qubit. When used that way, the bit is treated as a control for the
    /// operation. For example, "CX(q, xbit)" means "apply a Z gate to q if the xbit
    /// is True" and "flip_target_if_lhs_less_than_rhs(target_xbit, lhs, rhs)"
    /// means "if target_xbit is True then negate the amplitudes of states where lhs < rhs".
    ///
    /// Args:
    ///     target: The qubit to X-measure and Z-reset.
    ///
    /// Returns:
    ///     The allocated Bit that the result will be stored into.
    CircuitBuilderRaiiXBit hmr_raii_xbit(QubitId target);
    CircuitBuilderRaiiXBit hmr_raii_push_condition(QubitId target);

    void debug_print();
    void debug_print_if(QubitOrBit qb, BitId cond);
    void debug_print_if(RegisterId r, BitId cond);
    void debug_print_if(BitId b, BitId cond);
    void debug_print_if(QubitId q, BitId cond);
    void debug_print_if(BitId cond);
    void debug_print(std::span<const QubitId> q);
    void debug_print(std::span<const QubitOrBitOrBool> q);
    void debug_print(std::span<const BitOrBool> q);
    void debug_print(QubitId q);
    void debug_print(BitId b);
    void debug_print(QubitOrXZBitOrXZBool q);
    void debug_print(BitOrBool q);
    void debug_print(bool q);
    void broadcast_reset(stride_span<const QubitId> targets);

    void x(QubitOrXBitOrXBool target);
    void broadcast_x(stride_span<const QubitId> targets);

    void z(QubitOrBitOrBool v);
    void z(BitOrBool v);
    void z(BitId v);
    void z(bool v);

    /// Edits each entry of the target span to be its opposite.
    ///
    /// Emits an X instruction for each Qubit.
    /// Emits a BIT_INVERT instruction for each Bit.
    /// For build time boolean values, mutates the boolean value
    ///     to be its opposite. (That's why the type is
    ///     span<QubitOrBitOrBool> instead of
    ///     span<const QubitOrBitOrBool>.)
    void inplace_invert(QubitOrBitOrBool &target);
    void inplace_invert(stride_span<QubitOrBitOrBool> target);
    void inplace_invert(array_z &target);

    void cccx(
        QubitId control3, QubitId control2, QubitId control1, QubitId target, std::span<const QubitId> clean = {});
    void cccz(
        QubitId control3, QubitId control2, QubitId control1, QubitId control0, std::span<const QubitId> clean = {});

    void bit_invert(BitId v);
    void bit_store0(BitId v);
    void bit_store1(BitId v);
    void bit_invert_if(BitId v, BitId condition);
    void bit_store0_if(BitId v, BitId condition);
    void bit_store1_if(BitId v, BitId condition);
    void bit_store_and(BitId c1, BitId c2, BitId target);
    void broadcast_bit_store(
        const stride_span<const BitId> &targets, bool value, const stride_span<const BitId> &controls);
    void broadcast_bit_store(
        const stride_span<const BitId> &targets, bool value, const stride_span<const bool> &controls);
    void broadcast_bit_store(const stride_span<const BitId> &targets, bool value, const stride_span_z &controls);

    void cswap(BitId control, QubitId q1, QubitId q2);
    void cswap(QubitOrBitOrBool control, QubitId q1, QubitId q2);
    void broadcast_cswap(QubitOrTrue control, stride_span<const QubitId> q1, stride_span<const QubitId> q2);
    void broadcast_hmr(stride_span<const QubitId> target, stride_span<const BitId> output);

    RaiiCircuitBuilderPopPushedCondition raii_push_condition(BitId control);

    void left_rotate(stride_span<const QubitId> reg);
    void right_rotate(stride_span<const QubitId> reg);
    void cleft_rotate(QubitOrTrue control, stride_span<const QubitId> reg);
    void cright_rotate(QubitOrTrue control, stride_span<const QubitId> reg);

    void reset_and(QubitOrBitOrBool control1, QubitOrBitOrBool control2, QubitId target);
    void del_zero(QubitId target);
    void broadcast_del_zero(stride_span<const QubitId> target);
    void del_and(QubitOrBitOrBool control1, QubitOrBitOrBool control2, QubitId target);
    void broadcast_ccx(QubitOrBitOrBool control1, const stride_span_z &controls2, stride_span<const QubitId> targets);
    void broadcast_ccz(QubitOrBitOrBool control1, const stride_span_z &controls2, stride_span<const QubitId> targets);

    template <ControlLike TControl3, ControlLike TControl2, ControlLike TControl1, QuantumTargetLike TTarget>
    void cccx(
        const TControl3 &control3,
        const TControl2 &control2,
        const TControl1 &control1,
        const TTarget &target,
        std::span<const QubitId> clean = {}) {
        c_push(control3);
        c_push(control2);
        c_push(control1);
        c_resolve(target, clean);
    }
    template <ControlLike TControl3, ControlLike TControl2, ControlLike TControl1, ControlLike TControl0>
    void cccz(
        const TControl3 &control3,
        const TControl2 &control2,
        const TControl1 &control1,
        const TControl0 &control0,
        std::span<const QubitId> clean = {}) {
        c_push(control3);
        c_push(control2);
        c_push(control1);
        c_push(control0);
        c_resolve(MINUS_KET, clean);
    }
    void parity_ccz(
        const std::array<QubitOrBitOrBool, 2> &c_parity, QubitOrBitOrBool c_single, QubitOrBitOrBool target);
    void parity_ccx(
        const std::array<QubitOrBitOrBool, 2> &c_parity, QubitOrBitOrBool c_single, QubitOrXBitOrXBool target);
    void parity_cccx(
        const std::array<QubitOrBitOrBool, 2> &c_parity,
        QubitOrBitOrBool c2,
        QubitOrBitOrBool c3,
        QubitOrXBitOrXBool target,
        std::span<const QubitId> clean);
    void parity_cccz(
        const std::array<QubitOrBitOrBool, 2> &c_parity,
        QubitOrBitOrBool c2,
        QubitOrBitOrBool c3,
        QubitOrBitOrBool target,
        std::span<const QubitId> clean);

    void ccx(QubitOrBitOrBool control2, QubitOrBitOrBool control1, QubitOrXBitOrXBool target);
    void ccz(QubitOrBitOrBool c1, QubitOrBitOrBool c2, QubitOrBitOrBool c3);

    void swap(QubitId q1, QubitId q2);
    void swap(BitId q1, BitId q2);
    void swap(XBitId q1, XBitId q2);
    void broadcast_swap(const stride_span<const QubitId> &targets1, const stride_span<const QubitId> &targets2);
    void broadcast_swap(const stride_span<const BitId> &targets1, const stride_span<const BitId> &targets2);
    void broadcast_swap(const stride_span<const XBitId> &targets1, const stride_span<const XBitId> &targets2);
    void mux_swap(QubitOrXZBitOrXZBool q1, QubitOrXZBitOrXZBool q2);
    void mux_broadcast_swap(const stride_span_xz &targets1, const stride_span_xz &targets2);
    void mux_broadcast_swap(
        const stride_span<const QubitOrXZBitOrXZBool> &targets1,
        const stride_span<const QubitOrXZBitOrXZBool> &targets2);

    void cz(QubitId v1, QubitId v2);
    void cz(QubitOrBitOrBool c1, QubitOrBitOrBool c2);
    void mux_cz(QubitOrXZBitOrXZBool c1, QubitOrXZBitOrXZBool c2);
    void broadcast_cz(const stride_span<const bool> &controls1, const stride_span<const bool> &controls2);
    void broadcast_cz(const stride_span<const bool> &controls1, const stride_span<const QubitId> &controls2);
    void broadcast_cz(const stride_span<const bool> &controls1, const stride_span<const BitId> &controls2);
    void broadcast_cz(const stride_span<const BitId> &controls1, const stride_span<const BitId> &controls2);
    void broadcast_cz(const stride_span<const BitId> &controls1, const stride_span<const QubitId> &controls2);
    void broadcast_cz(const stride_span<const QubitId> &controls1, const stride_span<const QubitId> &controls2);
    void broadcast_cz(
        const stride_span<const QubitOrBitOrBool> &controls1, const stride_span<const QubitOrBitOrBool> &controls2);
    void broadcast_cz(const stride_span_z &controls1, const stride_span_z &controls2);
    void mux_broadcast_cz(
        const stride_span<const QubitOrXZBitOrXZBool> &controls1,
        const stride_span<const QubitOrXZBitOrXZBool> &controls2);
    void mux_broadcast_cz(const stride_span_xz &controls1, const stride_span_xz &controls2);

    void broadcast_cx(const stride_span<const bool> &controls, const stride_span<const QubitId> &targets);
    void broadcast_cx(const stride_span<const bool> &controls, const stride_span<const XBitId> &targets);
    void broadcast_cx(const stride_span<const bool> &controls, const stride_span<const XBool> &targets);
    void broadcast_cx(const stride_span<const BitId> &controls, const stride_span<const QubitId> &targets);
    void broadcast_cx(const stride_span<const BitId> &controls, const stride_span<const XBitId> &targets);
    void broadcast_cx(const stride_span<const BitId> &controls, const stride_span<const XBool> &targets);
    void broadcast_cx(const stride_span<const QubitId> &controls, const stride_span<const QubitId> &targets);
    void broadcast_cx(const stride_span<const QubitId> &controls, const stride_span<const XBitId> &targets);
    void broadcast_cx(const stride_span<const QubitId> &controls, const stride_span<const XBool> &targets);
    void broadcast_cx(const stride_span<const BitId> &controls, const stride_span<const BitId> &targets);
    void broadcast_cx(const stride_span<const bool> &controls, const stride_span<const BitId> &targets);
    void broadcast_cx(const stride_span_z &controls, const stride_span_x &targets);
    void broadcast_cx(QubitId control, const stride_span<const QubitId> &targets);
    void broadcast_cx(BitId control, const stride_span<const QubitId> &targets);
    void mux_cx(
        const stride_span<const QubitOrXZBitOrXZBool> &controls,
        const stride_span<const QubitOrXZBitOrXZBool> &targets);
    void mux_broadcast_cx(
        const stride_span<const QubitOrXZBitOrXZBool> &controls,
        const stride_span<const QubitOrXZBitOrXZBool> &targets);
    void mux_broadcast_cx(const stride_span_xz &controls, const stride_span_xz &targets);
    void mux_cx(QubitOrXZBitOrXZBool control, QubitOrXZBitOrXZBool target);
    void cx(QubitId control, QubitId target);
    void cx(BitId control, BitId target);
    void cx(bool control, BitId target);
    void cx(BitOrBool control, BitId target);
    void cx(QubitOrBitOrBool control, QubitOrXBitOrXBool target);
};

}  // namespace kickmix

#endif
