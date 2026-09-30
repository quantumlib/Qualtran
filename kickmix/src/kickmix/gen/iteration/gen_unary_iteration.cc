#include "gen_unary_iteration.h"

#include <algorithm>
#include <bit>
#include <stdexcept>

using namespace kickmix;

UnaryIterationCursor::UnaryIterationCursor(
    CircuitBuilder &builder, CircuitGenCtx ctx, stride_span<const QubitId> address, QubitOrTrue control)
    : builder_(builder), ctx_(ctx), control_(control), address_space_size_(0), cursor_(0) {
    if (address.size() > 63) {
        throw std::invalid_argument(
            "UnaryIterationCursor: address registers wider than 63 qubits aren't supported "
            "(address values must fit in a uint64_t).");
    }
    address_space_size_ = uint64_t{1} << address.size();
    // Park one past the last address value, which the address register can never hold.
    cursor_ = address_space_size_;
    address_.reserve(address.size());
    for (QubitId q : address) {
        address_.push_back(q);
    }

    // Reserve one clean qubit per address bit, or one qubit when `address` is empty
    // so that `match_qubit()` always refers to a valid qubit that resets to |0>
    // when parked.
    size_t num_workspace_qubits = std::max(address_.size(), size_t{1});
    auto workspace = ctx_.take_clean(num_workspace_qubits, "UnaryIterationCursor");
    workspace_.assign(workspace.begin(), workspace.end());
}

QubitId UnaryIterationCursor::match_qubit() const {
    return workspace_[0];
}

// Level k corresponds to `address_[k]` (little-endian), so `workspace_[n - 1]`
// combines `control_` with the most significant address bit and `workspace_[0]`
// (`match_qubit()`) combines in `address_[0]`.
void UnaryIterationCursor::compute_level(size_t level, uint64_t address_value) {
    size_t top = address_.size() - 1;
    bool bit = (address_value >> level) & 1;
    if (!bit) {
        builder_.x(address_[level]);
    }
    QubitOrTrue parent = level == top ? control_ : QubitOrTrue(workspace_[level + 1]);
    builder_.reset_and(parent, address_[level], workspace_[level]);
    if (!bit) {
        builder_.x(address_[level]);
    }
}

void UnaryIterationCursor::uncompute_level(size_t level, uint64_t address_value) {
    size_t top = address_.size() - 1;
    bool bit = (address_value >> level) & 1;
    if (!bit) {
        builder_.x(address_[level]);
    }
    QubitOrTrue parent = level == top ? control_ : QubitOrTrue(workspace_[level + 1]);
    builder_.del_and(parent, address_[level], workspace_[level]);
    if (!bit) {
        builder_.x(address_[level]);
    }
}

void UnaryIterationCursor::build_levels(uint64_t address_value, size_t num_levels) {
    for (size_t k = num_levels; k-- > 0;) {
        compute_level(k, address_value);
    }
}

void UnaryIterationCursor::teardown_levels(uint64_t address_value, size_t num_levels) {
    for (size_t k = 0; k < num_levels; k++) {
        uncompute_level(k, address_value);
    }
}

void UnaryIterationCursor::build_full(uint64_t address_value) {
    if (address_.empty()) {
        builder_.reset_and(control_, true, workspace_[0]);
        return;
    }
    build_levels(address_value, address_.size());
}

void UnaryIterationCursor::teardown_full(uint64_t address_value) {
    if (address_.empty()) {
        builder_.del_and(control_, true, workspace_[0]);
        return;
    }
    teardown_levels(address_value, address_.size());
}

void UnaryIterationCursor::move_to(uint64_t new_address_value) {
    bool was_live = is_live(cursor_);
    bool now_live = is_live(new_address_value);
    if (new_address_value == cursor_ || (!was_live && !now_live)) {
        cursor_ = new_address_value;
        return;
    }

    auto mark = builder_.raii_mark_block_entry("unary_iteration");
    if (!was_live) {
        build_full(new_address_value);
    } else if (!now_live) {
        teardown_full(cursor_);
    } else {
        // Only recompute levels at and below the highest bit where the two values differ.
        size_t level = std::bit_width(cursor_ ^ new_address_value) - 1;
        teardown_levels(cursor_, level);
        QubitOrTrue parent = level == address_.size() - 1 ? control_ : QubitOrTrue(workspace_[level + 1]);
        builder_.cx(parent, workspace_[level]);
        build_levels(new_address_value, level);
    }
    cursor_ = new_address_value;
}

void UnaryIterationCursor::close() {
    move_to(address_space_size_);
}

void kickmix::gen_unary_iteration(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    stride_span<const QubitId> address,
    QubitOrTrue control,
    std::span<const uint64_t> address_values,
    const std::function<void(uint64_t address_value, QubitId match_qubit)> &body) {
    if (address_values.empty()) {
        return;
    }
    UnaryIterationCursor cursor(builder, ctx, address, control);
    for (uint64_t address_value : address_values) {
        cursor.move_to(address_value);
        body(address_value, cursor.match_qubit());
    }
    cursor.close();
}
