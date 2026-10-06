#include "kickmix/py/circuit/subs/unary_iteration.pybind.h"

#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "kickmix/py/util.pybind.h"
#include "kickmix/py/val/converted_array.pybind.h"
#include "kickmix/py/val/id.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

namespace {

void parse_address_values(
    const pybind11::object &address_values_obj,
    size_t num_address_bits,
    pybind11::int_ *out_begin,
    pybind11::int_ *out_end) {
    uint64_t limit = uint64_t{1} << num_address_bits;
    if (address_values_obj.is_none()) {
        *out_begin = pybind11::int_(0);
        *out_end = pybind11::int_(limit);
    } else if (pybind11::isinstance(address_values_obj, pybind11::module_::import("builtins").attr("range"))) {
        pybind11::int_ step = address_values_obj.attr("step").cast<pybind11::int_>();
        if (!step.equal(pybind11::int_(1))) {
            throw std::invalid_argument("address_values must have step 1; visit other patterns with move_to.");
        }
        pybind11::int_ start = address_values_obj.attr("start").cast<pybind11::int_>();
        pybind11::int_ stop = address_values_obj.attr("stop").cast<pybind11::int_>();
        *out_begin = start;
        *out_end = stop < start ? start : stop;
    } else {
        throw std::invalid_argument("address_values must be None or a range.");
    }
}

}  // namespace

void PyUnaryIterationCursor::check_open() const {
    if (!is_open()) {
        throw std::invalid_argument("The unary iteration cursor is not open. Use 'with cursor as it:' to open a pass.");
    }
}

void PyUnaryIterationCursor::open_session() {
    if (is_open()) {
        throw std::invalid_argument(
            "The unary iteration cursor is already open. "
            "Leave the enclosing 'with' block before starting another pass.");
    }

    // Reserve one qubit when `control` is `False` so `match_qubit` remains a
    // valid `|0>` qubit; otherwise reserve `max(1, len(address))` workspace qubits.
    size_t num_workspace_qubits = control.is_false() ? 1 : std::max<size_t>(1, address_storage.size());
    if (py_builder->allocated_qubits + num_workspace_qubits > py_builder->max_qubits) {
        std::stringstream ss;
        ss << "Not enough free qubits for unary iteration.";
        ss << "\n    allocated: " << py_builder->allocated_qubits;
        ss << "\n    needed: " << num_workspace_qubits;
        ss << "\n    max: " << py_builder->max_qubits;
        throw std::invalid_argument(ss.str());
    }

    position = pybind11::int_(address_space_size());
    next_address_value = address_values_begin;
    workspace_qubits.reserve(num_workspace_qubits);
    for (size_t k = 0; k < num_workspace_qubits; k++) {
        workspace_qubits.push_back(py_builder->alloc_qubit());
    }

    if (control.is_bit()) {
        py_builder->builder.push_condition((BitId)control);
        pushed_condition = true;
    }
    if (!control.is_false()) {
        QubitOrTrue cursor_control = true;
        if (control.is_qubit()) {
            cursor_control = (QubitId)control;
        }
        std::span<const QubitId> workspace_span{workspace_qubits.data(), num_workspace_qubits};
        cursor = std::make_unique<UnaryIterationCursor>(
            py_builder->builder,
            CircuitGenCtx{workspace_span},
            stride_span<const QubitId>(address_storage),
            cursor_control);
    }
}

void PyUnaryIterationCursor::close_session() {
    is_iterating = false;
    if (cursor != nullptr) {
        cursor->close();
        cursor.reset();
    }
    if (pushed_condition) {
        py_builder->builder.pop_condition();
        pushed_condition = false;
    }
    while (!workspace_qubits.empty()) {
        QubitId q = workspace_qubits.back();
        workspace_qubits.pop_back();
        py_builder->free_qubit(q);
    }
}

QubitId PyUnaryIterationCursor::match_qubit() const {
    check_open();
    return workspace_qubits[0];
}

pybind11::int_ PyUnaryIterationCursor::cur_address_value() const {
    check_open();
    return position;
}

void PyUnaryIterationCursor::move_to(const pybind11::int_ &address_value) {
    check_open();
    position = address_value;
    if (cursor != nullptr) {
        uint64_t limit = address_space_size();
        if (address_value >= pybind11::int_(0) && address_value < pybind11::int_(limit)) {
            cursor->move_to(address_value.cast<uint64_t>());
        } else {
            cursor->move_to(limit);
        }
    }
}

pybind11::object PyUnaryIterationCursor::next() {
    check_open();
    if (cursor == nullptr || next_address_value >= address_values_end) {
        is_iterating = false;
        throw pybind11::stop_iteration();
    }
    is_iterating = true;
    pybind11::int_ address_value = next_address_value;
    next_address_value = next_address_value + pybind11::int_(1);
    move_to(address_value);
    return pybind11::make_tuple(address_value, match_qubit());
}

std::string PyUnaryIterationCursor::repr() const {
    std::stringstream ss;
    ss << "<km.UnaryIterationCursor address=km.array([";
    for (size_t k = 0; k < address_storage.size(); k++) {
        if (k > 0) {
            ss << ", ";
        }
        ss << "km.q(" << address_storage[k].untagged_id() << ")";
    }
    ss << "])";

    if (!address_values_begin.equal(pybind11::int_(0)) ||
        !address_values_end.equal(pybind11::int_(address_space_size()))) {
        ss << ", address_values=range(" << pybind11::str(pybind11::handle(address_values_begin)).cast<std::string>()
           << ", " << pybind11::str(pybind11::handle(address_values_end)).cast<std::string>() << ")";
    }

    if (!control.is_true()) {
        ss << ", control=";
        if (control.is_qubit()) {
            ss << "km.q(" << control.untagged_id() << ")";
        } else if (control.is_bit()) {
            ss << "km.b(" << control.untagged_id() << ")";
        } else if (control.is_bool()) {
            ss << (control.is_true() ? "True" : "False");
        }
    }

    ss << ">";
    return ss.str();
}

std::unique_ptr<PyUnaryIterationCursor> kickmix_py::make_unary_iteration(
    PyCircuitBuilder &self,
    const pybind11::object &address_obj,
    const pybind11::object &address_values_obj,
    const pybind11::object &control_obj) {
    ConvertedArrayXZ address_conv = ConvertedArrayXZ::from_obj(address_obj, "address");
    auto address = address_conv.span.checked_cast_to_qubit_ids("address");
    if (address.size() > 63) {
        throw std::invalid_argument("len(address) > 63");
    }

    auto result = std::make_unique<PyUnaryIterationCursor>();
    result->py_builder = &self;
    result->address_storage.resize(address.size());
    for (size_t k = 0; k < address.size(); k++) {
        result->address_storage[k] = address[k];
    }

    parse_address_values(
        address_values_obj, address.size(), &result->address_values_begin, &result->address_values_end);

    result->control = obj_to_qubit_or_bit_or_bool(control_obj, "control");

    return result;
}

pybind11::class_<PyUnaryIterationCursor> kickmix_py::register_unary_iteration_class(pybind11::module &m) {
    return pybind11::class_<PyUnaryIterationCursor>(
        m,
        "UnaryIterationCursor",
        clean_doc_string(R"DOC(
            A cursor that steps through values of an address register.

            Create a cursor with `builder.unary_iteration(address, ...)` and open a pass
            using a `with` block. Inside the block, iterate with a `for` loop to visit
            `(address_value, match_qubit)` pairs in order, or call `move_to` to jump to
            any value.

            Entering the `with` block allocates `max(1, len(address))` temporary qubits;
            leaving the block uncomputes and frees them. A cursor that is never opened
            adds no operations to the circuit, and the same cursor can be reused across
            multiple `with` blocks.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> address = builder.create_quantum_register(2, name='address')
                >>> target = builder.create_quantum_register(1, name='target')
                >>> cursor = builder.unary_iteration(address)
                >>> with cursor:
                ...     for address_value, match_qubit in cursor:
                ...         if address_value == 3:
                ...             builder.cx(match_qubit, target[0])
                >>> # Reusing the same cursor to uncompute:
                >>> with cursor:
                ...     for address_value, match_qubit in cursor:
                ...         if address_value == 3:
                ...             builder.cx(match_qubit, target[0])
                >>> circuit = builder.finish_circuit()
                >>> sim = km.Simulator(batch_size=4)
                >>> sim.use_same_registers_as(circuit)
                >>> for a in range(4):
                ...     sim.write_within_shot('address', a, a)
                >>> sim.do(circuit)
                >>> for a in range(4):
                ...     print(f"{a}: {sim.read_within_shot('target', a, out=int)}")
                0: 0
                1: 0
                2: 0
                3: 0
        )DOC")
            .data());
}

void kickmix_py::register_unary_iteration_methods(pybind11::class_<PyUnaryIterationCursor> &c) {
    c.def(
        pybind11::init([](PyCircuitBuilder &builder,
                          const pybind11::object &address,
                          const pybind11::object &address_values,
                          const pybind11::object &control) {
            return make_unary_iteration(builder, address, address_values, control);
        }),
        pybind11::arg("builder"),
        pybind11::arg("address"),
        pybind11::arg("address_values") = pybind11::none(),
        pybind11::kw_only(),
        pybind11::arg("control") = true,
        pybind11::keep_alive<1, 2>(),
        clean_doc_string(R"DOC(
            @signature def __init__(self, builder: km.CircuitBuilder, address: km.array, address_values: range | None = None, *, control: q | b | bool = True) -> None:
            Initializes a reusable unary iteration cursor over `address`.

            Args:
                builder: Where circuit operations and temporary qubit allocations are
                    recorded.
                address: The little-endian address to match against (at most 63 qubits).
                address_values: Defaults to `range(1 << len(address))`. The step-1 `range`
                    of values visited by `for` loops.
                control: Gates the iteration when provided. If `False`, no operations are
                    emitted and `for` loops visit zero values. Defaults to `True`.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> address = builder.create_quantum_register(2, name='address')
                >>> target = builder.create_quantum_register(1, name='target')
                >>> cursor = km.UnaryIterationCursor(builder, address)
                >>> with cursor:
                ...     cursor.move_to(2)
                ...     builder.cx(cursor.match_qubit, target[0])
                >>> builder.finish_circuit().max_magic()
                1
        )DOC")
            .data());

    c.def(
        "__enter__",
        [](PyUnaryIterationCursor &self) -> PyUnaryIterationCursor & {
            if (self.is_open()) {
                throw std::invalid_argument("The unary iteration cursor is already open.");
            }
            self.open_session();
            return self;
        },
        clean_doc_string(R"DOC(
            @signature def __enter__(self) -> km.UnaryIterationCursor:
            Opens a unary iteration pass and allocates temporary qubits.

            No gates are emitted on entry. The cursor starts at `1 << len(address)` with
            `match_qubit` set to `|0>` until the first `for` step or `move_to` call.

            Returns:
                The open cursor (`self`).

            Raises:
                ValueError: If the cursor is already open.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> address = builder.create_quantum_register(2, name='address')
                >>> with builder.unary_iteration(address) as it:
                ...     it.cur_address_value
                4
        )DOC")
            .data());

    c.def(
        "__exit__",
        [](PyUnaryIterationCursor &self, const pybind11::object &, const pybind11::object &, const pybind11::object &)
            -> bool {
            self.close_session();
            return false;
        },
        clean_doc_string(R"DOC(
            @signature def __exit__(self, exc_type, exc_value, traceback) -> bool:
            Closes the active pass, uncomputing and freeing its temporary qubits.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> address = builder.create_quantum_register(2, name='address')
                >>> with builder.unary_iteration(address) as it:
                ...     it.move_to(1)
                ...     print(builder.num_allocated_qubits)
                4
                >>> builder.num_allocated_qubits
                2
        )DOC")
            .data());

    c.def(
        "__iter__",
        [](PyUnaryIterationCursor &self) -> PyUnaryIterationCursor & {
            if (!self.is_open()) {
                throw std::invalid_argument(
                    "The unary iteration cursor is not open. Open it with a 'with' block before iterating:\n"
                    "    with builder.unary_iteration(address) as it:\n"
                    "        for address_value, match_qubit in it:\n"
                    "            ...");
            }
            if (self.is_iterating) {
                throw std::invalid_argument("The unary iteration cursor is already being iterated.");
            }
            self.next_address_value = self.address_values_begin;
            return self;
        },
        clean_doc_string(R"DOC(
            @signature def __iter__(self) -> Iterator[tuple[int, q]]:
            Yields `(address_value, match_qubit)` pairs for each configured value.

            The cursor must already be open inside a `with` block. Finishing or breaking
            out of the loop leaves the pass open until the `with` block exits.

            Raises:
                ValueError: If the cursor is not open or is already iterating.
        )DOC")
            .data());

    c.def(
        "__next__",
        [](PyUnaryIterationCursor &self) -> pybind11::object {
            return self.next();
        },
        clean_doc_string(R"DOC(
            @signature def __next__(self) -> tuple[int, q]:
            Advances to the next value and returns `(address_value, match_qubit)`.

            Raises:
                StopIteration: When all values in `address_values` have been visited.
                ValueError: If the cursor is not open.
        )DOC")
            .data());

    c.def_property_readonly(
        "match_qubit",
        [](const PyUnaryIterationCursor &self) -> QubitId {
            return self.match_qubit();
        },
        clean_doc_string(R"DOC(
            @signature def match_qubit(self) -> q:
            Returns the qubit that is ON in the parts of the superposition where `control`
            is active and `address == cur_address_value`.

            Use this qubit only as a control. When `cur_address_value` is outside
            `range(1 << len(address))`, this qubit is `|0>`.

            Raises:
                ValueError: If the cursor is not open.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> address = builder.create_quantum_register(2, name='address')
                >>> target = builder.create_quantum_register(1, name='target')
                >>> with builder.unary_iteration(address) as it:
                ...     it.move_to(2)
                ...     builder.cx(it.match_qubit, target[0])
        )DOC")
            .data());

    c.def_property_readonly(
        "cur_address_value",
        [](const PyUnaryIterationCursor &self) -> pybind11::int_ {
            return self.cur_address_value();
        },
        clean_doc_string(R"DOC(
            @signature def cur_address_value(self) -> int:
            Returns the address value the cursor is currently positioned at.

            When a pass is first opened, `cur_address_value` is `1 << len(address)` and
            `match_qubit` is `|0>`.

            Raises:
                ValueError: If the cursor is not open.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> address = builder.create_quantum_register(2, name='address')
                >>> with builder.unary_iteration(address) as it:
                ...     it.move_to(3)
                ...     it.cur_address_value
                3
        )DOC")
            .data());

    c.def(
        "move_to",
        [](PyUnaryIterationCursor &self, const pybind11::object &address_value) {
            self.move_to(address_value.attr("__index__")().cast<pybind11::int_>());
        },
        pybind11::arg("address_value"),
        clean_doc_string(R"DOC(
            @signature def move_to(self, address_value: int) -> None:
            Moves the cursor to `address_value` and updates `match_qubit`.

            For a non-empty `address`, moving from out of range to an in-range value
            currently costs at most `len(address)` Toffoli gates when `control` is a
            qubit, or `len(address) - 1` when `control` is `True` or a classical bit.
            Moving between distinct in-range values `a` and `b` costs at most
            `(a ^ b).bit_length() - 1` Toffoli gates. Moving to any value outside
            `range(1 << len(address))` sets `match_qubit` to `|0>` with zero Toffoli
            gates while recording `cur_address_value == address_value`.

            Args:
                address_value: The target value to position the cursor at.

            Raises:
                ValueError: If the cursor is not open.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> address = builder.create_quantum_register(2, name='address')
                >>> with builder.unary_iteration(address) as it:
                ...     it.move_to(2)
                ...     it.cur_address_value
                2
        )DOC")
            .data());

    c.def(
        "__repr__",
        [](const PyUnaryIterationCursor &self) -> std::string {
            return self.repr();
        },
        clean_doc_string(R"DOC(
            @signature def __repr__(self) -> str:
            Returns a string describing the cursor.
        )DOC")
            .data());

    c.def(
        "close",
        [](PyUnaryIterationCursor &self) {
            self.close_session();
        },
        clean_doc_string(R"DOC(
            @signature def close(self) -> None:
            Closes the active pass, uncomputing and freeing its temporary qubits.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> address = builder.create_quantum_register(2, name='address')
                >>> with builder.unary_iteration(address) as it:
                ...     it.move_to(1)
                ...     print(builder.num_allocated_qubits)
                ...     it.close()
                ...     print(builder.num_allocated_qubits)
                4
                2
        )DOC")
            .data());
}
