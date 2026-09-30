#ifndef KICKGEN_GEN_UNARY_ITERATION_H
#define KICKGEN_GEN_UNARY_ITERATION_H

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "kickmix/build/circuit_builder.h"

namespace kickmix {

/// Steps through integer values of a quantum address register, maintaining a
/// qubit that marks when `address` matches the current value.
///
/// Requires `max(1, address.size())` clean workspace qubits in `ctx`. Construction
/// emits no gates and starts the cursor at `address_space_size()`, where
/// `match_qubit()` is `|0>`. Calling `move_to(v)` updates `match_qubit()` to be ON
/// in the parts of the superposition where `control` is active and `address == v`
/// for `v < address_space_size()`, or resets it to `|0>` for `v >= address_space_size()`.
///
/// You must call `close()` before dropping an active cursor. The destructor never
/// emits gates, because a cursor may be owned by a garbage-collected Python object
/// whose destruction time is unpredictable.
///
/// Do not modify `address`, `control`, or `match_qubit()` while the cursor is
/// positioned at a value below `address_space_size()`.
struct UnaryIterationCursor {
    /// Reserves `max(1, address.size())` clean qubits from `ctx` without emitting gates.
    ///
    /// `address` is little-endian and may have at most 63 qubits. Pass `control = true`
    /// for an uncontrolled cursor. To control the pass with a classical bit, wrap the
    /// cursor's lifetime in `builder.raii_push_condition(bit)`.
    UnaryIterationCursor(
        CircuitBuilder &builder, CircuitGenCtx ctx, stride_span<const QubitId> address, QubitOrTrue control = true);

    /// Destroys the cursor without emitting gates; call `close()` before destruction.
    ~UnaryIterationCursor() = default;

    UnaryIterationCursor(const UnaryIterationCursor &) = delete;
    UnaryIterationCursor &operator=(const UnaryIterationCursor &) = delete;
    UnaryIterationCursor(UnaryIterationCursor &&) = delete;
    UnaryIterationCursor &operator=(UnaryIterationCursor &&) = delete;

    /// Returns the qubit that is ON in the parts of the superposition where
    /// `control` is active and `address == cur_address_value()`.
    QubitId match_qubit() const;

    /// Returns the cursor's current address value (initially `address_space_size()`).
    uint64_t cur_address_value() const {
        return cursor_;
    }

    /// Returns the number of qubits in the address register.
    size_t num_address_bits() const {
        return address_.size();
    }

    /// Returns the number of representable address values (`1 << num_address_bits()`).
    uint64_t address_space_size() const {
        return address_space_size_;
    }

    /// Moves the cursor to `new_address_value` and updates `match_qubit()`.
    ///
    /// Moving between distinct in-range values `a` and `b` currently costs at most
    /// `bit_width(a ^ b) - 1` Toffoli gates. Entering the in-range values from
    /// `>= address_space_size()` costs at most `num_address_bits()` Toffoli gates (or
    /// `num_address_bits() - 1` when `control` is `true`, and `0` when `address` is
    /// empty). Moving to any value `>= address_space_size()` resets `match_qubit()`
    /// to `|0>` with zero Toffoli gates.
    void move_to(uint64_t new_address_value);

    /// Resets temporary workspace qubits to `|0>` and parks the cursor at `address_space_size()`.
    ///
    /// Equivalent to `move_to(address_space_size())`. Idempotent and costs zero Toffoli gates.
    void close();

   private:
    bool is_live(uint64_t address_value) const {
        return address_value < address_space_size_;
    }

    void compute_level(size_t level, uint64_t address_value);
    void uncompute_level(size_t level, uint64_t address_value);
    void build_levels(uint64_t address_value, size_t num_levels);
    void teardown_levels(uint64_t address_value, size_t num_levels);
    void build_full(uint64_t address_value);
    void teardown_full(uint64_t address_value);

    CircuitBuilder &builder_;
    CircuitGenCtx ctx_;
    std::vector<QubitId> address_;
    QubitOrTrue control_;
    std::vector<QubitId> workspace_;
    uint64_t address_space_size_;
    uint64_t cursor_;
};

/// Visits each value in `address_values` in order, calling `body(address_value, match_qubit)`,
/// and then closes the cursor.
void gen_unary_iteration(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    stride_span<const QubitId> address,
    QubitOrTrue control,
    std::span<const uint64_t> address_values,
    const std::function<void(uint64_t address_value, QubitId match_qubit)> &body);

}  // namespace kickmix

#endif
