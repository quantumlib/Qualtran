#ifndef KICKGEN_PYBIND_UNARY_ITERATION_H
#define KICKGEN_PYBIND_UNARY_ITERATION_H

#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

#include "kickmix/gen/iteration/gen_unary_iteration.h"
#include "kickmix/py/circuit/circuit_builder.pybind.h"

namespace kickmix_py {

/// Python context-manager and iterator wrapper around `kickmix::UnaryIterationCursor`.
///
/// Lowers Python-only control types before delegating to `UnaryIterationCursor`:
/// a classical bit control pushes a condition around the active pass, and `False`
/// skips cursor construction and leaves `match_qubit` fixed at `|0>`.
struct PyUnaryIterationCursor {
    PyCircuitBuilder *py_builder = nullptr;
    std::vector<kickmix::QubitId> address_storage;
    pybind11::int_ address_values_begin{0};
    pybind11::int_ address_values_end{0};
    kickmix::QubitOrBitOrBool control = true;

    /// Temporary qubits allocated from `py_builder` while a pass is open.
    std::vector<kickmix::QubitId> workspace_qubits;
    /// True if opening the pass pushed a classical condition that `close_session` must pop.
    bool pushed_condition = false;
    /// Underlying C++ cursor for the active pass, or null when `control` is `False`.
    std::unique_ptr<kickmix::UnaryIterationCursor> cursor;
    /// Current address value of the open pass.
    pybind11::int_ position{0};
    /// Next address value to yield from `__next__`.
    pybind11::int_ next_address_value{0};
    bool is_iterating = false;

    PyUnaryIterationCursor() = default;
    /// Destroys the wrapper without emitting circuit operations.
    ~PyUnaryIterationCursor() = default;

    PyUnaryIterationCursor(const PyUnaryIterationCursor &) = delete;
    PyUnaryIterationCursor &operator=(const PyUnaryIterationCursor &) = delete;
    PyUnaryIterationCursor(PyUnaryIterationCursor &&) = delete;
    PyUnaryIterationCursor &operator=(PyUnaryIterationCursor &&) = delete;

    /// Returns the number of representable address values (`1 << len(address)`).
    uint64_t address_space_size() const {
        return uint64_t{1} << address_storage.size();
    }
    /// Returns true if a pass is currently open.
    bool is_open() const {
        return !workspace_qubits.empty();
    }
    void open_session();
    void close_session();
    pybind11::object next();
    void check_open() const;
    kickmix::QubitId match_qubit() const;
    pybind11::int_ cur_address_value() const;
    void move_to(const pybind11::int_ &address_value);
    std::string repr() const;
};

std::unique_ptr<PyUnaryIterationCursor> make_unary_iteration(
    PyCircuitBuilder &self,
    const pybind11::object &address_obj,
    const pybind11::object &address_values_obj,
    const pybind11::object &control_obj);

pybind11::class_<PyUnaryIterationCursor> register_unary_iteration_class(pybind11::module &m);
void register_unary_iteration_methods(pybind11::class_<PyUnaryIterationCursor> &c);

}  // namespace kickmix_py

#endif
