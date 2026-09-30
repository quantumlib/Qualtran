#ifndef KICKMIX_CIRCUIT_GEN_CTX_H
#define KICKMIX_CIRCUIT_GEN_CTX_H

#include <span>
#include <sstream>

#include "kickmix/id/qubit_id.h"
#include "kickmix/mem/stride_span.h"
#include "kickmix/util/fixed_width_int.h"

namespace kickmix {

inline void throw_unless(bool b) {
    if (!b) {
        throw std::invalid_argument("unlabelled throw_unless");
    }
}
inline void throw_unless(bool b, std::string_view text) {
    if (!b) {
        throw std::invalid_argument(std::string(text));
    }
}

template <typename T>
struct LinkedListStack {
    stride_span<const T> items;
    const LinkedListStack<T> *next;
    size_t items_left = 0;

    const T &back() const {
        if (items.empty()) {
            throw std::invalid_argument("Empty stack.");
        }
        return items.back();
    }
    bool empty() const {
        return items_left == 0;
    }
    size_t count() const {
        return items_left;
    }
    LinkedListStack<T> after_push(stride_span<const T> new_items) const {
        if (new_items.empty()) {
            return *this;
        }
        if (items.empty() && next == nullptr) {
            return LinkedListStack<T>{.items = new_items, .next = nullptr, .items_left = new_items.size()};
        }
        return LinkedListStack<T>{.items = new_items, .next = this, .items_left = items_left + new_items.size()};
    }
    void copy_n_items_into(size_t n, std::vector<T> &out) const {
        const LinkedListStack *cur = this;
        while (n > 0) {
            if (cur == nullptr) {
                throw std::invalid_argument("Not enough items.");
            }
            size_t m = std::min(n, cur->items.size());
            for (size_t k = 0; k < m; k++) {
                out.push_back(cur->items[k]);
            }
            n -= m;
            cur = cur->next;
        }
    }
};

struct CircuitGenCtx {
    /// Clean qubits are qubits guaranteed to be unused (i.e. in the zero state).
    /// Although not required, clean qubits should be reset before use (just to more clearly indicate where the usage
    /// is).
    std::span<const QubitId> clean_workspace{};

    /// Dirty qubits are qubits being used to store data, but they are currently at rest. In
    /// some cases this is a useful resource.
    LinkedListStack<kickmix::QubitId> dirty_workspace{};
    bool minimize_qubits = false;

    bool has_no_workspace() const {
        return clean_workspace.empty() && dirty_workspace.empty();
    }
    std::span<const kickmix::QubitId> clean_workspace_if_not_minimizing() const {
        if (minimize_qubits) {
            return {};
        }
        return clean_workspace;
    }
    CircuitGenCtx with_more_dirty_qubits(stride_span<const QubitId> dirty) const {
        return CircuitGenCtx{clean_workspace, dirty_workspace.after_push(dirty), minimize_qubits};
    }
    CircuitGenCtx with_clean_subspan(size_t offset) const {
        if (offset >= clean_workspace.size()) {
            return CircuitGenCtx{{}, dirty_workspace, minimize_qubits};
        }
        return CircuitGenCtx{clean_workspace.subspan(offset), dirty_workspace, minimize_qubits};
    }
    CircuitGenCtx with_minimize_qubits(bool minimize_qubits_val) const {
        return CircuitGenCtx{clean_workspace, dirty_workspace, minimize_qubits_val};
    }
    std::span<const kickmix::QubitId> take_clean(
        size_t num_clean, std::string_view location_context = "A method that didn't specify its name") {
        if (clean_workspace.size() < num_clean) {
            std::stringstream ss;
            ss << location_context;
            ss << " tried to take ";
            ss << num_clean << " clean qubits, ";
            ss << "but only " << clean_workspace.size() << " were available.";
            throw std::invalid_argument(ss.str());
        }
        auto result = clean_workspace.subspan(0, num_clean);
        clean_workspace = clean_workspace.subspan(num_clean);
        return result;
    }
};

}  // namespace kickmix

#endif
