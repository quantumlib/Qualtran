#include "kickmix/py/circuit/circuit_builder.pybind.h"

#include <kickmix/py/val/converted_array.pybind.h>

#include "circuit.pybind.h"
#include "kickmix/py/circuit/ops/bit_store.pybind.h"
#include "kickmix/py/circuit/ops/ccx.pybind.h"
#include "kickmix/py/circuit/ops/cswap.pybind.h"
#include "kickmix/py/circuit/ops/cx.pybind.h"
#include "kickmix/py/circuit/ops/cz.pybind.h"
#include "kickmix/py/circuit/ops/swap.pybind.h"
#include "kickmix/py/circuit/ops/x.pybind.h"
#include "kickmix/py/circuit/subs/bit_rotate.pybind.h"
#include "kickmix/py/circuit/subs/flip_if_cmp.pybind.h"
#include "kickmix/py/circuit/subs/gf_arithmetic.pybind.h"
#include "kickmix/py/circuit/subs/iadd.pybind.h"
#include "kickmix/py/circuit/subs/multi_control_x.pybind.h"
#include "kickmix/py/circuit/subs/unary_iteration.pybind.h"
#include "kickmix/py/util.pybind.h"
#include "kickmix/py/val/angle.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

void broadcast_cz_struct(PyCircuitBuilder &self, const stride_span_z &c1, const stride_span_z &c2) {
    if (c1.count != c2.count) {
        throw std::invalid_argument("Incompatible lengths.");
    }
    for (size_t k = 0; k < c1.count; k++) {
        // TODO: fancier type-specific broadcasts
        self.builder.cz(c1[k], c2[k]);
    }
}

void append_z_obj(PyCircuitBuilder &self, pybind11::object target_obj) {
    QubitOrBitOrBool t = obj_to_qubit_or_bit_or_bool(target_obj, "target");
    self.builder.z(t);
}

static void z_pow_obj(PyCircuitBuilder &self, const pybind11::object &target_obj, const pybind11::object &angle_obj) {
    auto angle = fixed_precision_angle_from_half_turns_obj(angle_obj);
    auto target_conv = ConvertedArrayXZ::from_obj(target_obj, "target");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    for (auto e : target) {
        self.builder.z_pow(e, angle);
    }
}

void append_neg_obj(PyCircuitBuilder &self) {
    self.builder.neg();
}

void append_neg_if_obj(PyCircuitBuilder &self, pybind11::object condition_obj) {
    BitId condition = obj_to_bit_id(condition_obj, "condition");
    self.builder.neg_if(condition);
}

void append_push_condition_obj(PyCircuitBuilder &self, pybind11::object condition_obj) {
    BitId condition = obj_to_bit_id(condition_obj, "condition");
    self.builder.push_condition(condition);
}

void append_pop_condition_obj(PyCircuitBuilder &self) {
    self.builder.pop_condition();
}

void append_debug_print_obj(PyCircuitBuilder &self, const pybind11::object &arg_obj) {
    ConvertedArrayXZ v = ConvertedArrayXZ::from_obj(arg_obj, "target");
    for (size_t k = 0; k < v.span.size(); k++) {
        self.builder.debug_print(v.span[k]);
    }
    self.builder.debug_print();
}

void broadcast_hmr_obj(PyCircuitBuilder &self, pybind11::object target_obj, pybind11::object output_obj) {
    auto v_target = ConvertedArrayXZ::from_obj(target_obj, "target");
    stride_span<const QubitId> target = v_target.span.checked_cast_to_qubit_ids("target");
    auto v_output = ConvertedArrayXZ::from_obj(output_obj, "output");
    stride_span<const BitId> output = v_output.span.checked_cast_to_bit_ids("output");
    if (target.size() != output.size()) {
        throw std::invalid_argument("len(target) != len(output)");
    }

    self.builder.broadcast_hmr(target, output);
}

void broadcast_reset_obj(PyCircuitBuilder &self, pybind11::object target_obj) {
    auto v_target = ConvertedArrayXZ::from_obj(target_obj, "target");
    stride_span<const QubitId> target = v_target.span.checked_cast_to_qubit_ids("target");
    self.builder.broadcast_reset(target);
}

void append_reset_obj(PyCircuitBuilder &self, pybind11::object target_obj) {
    QubitId t = obj_to_qubit_id(target_obj, "target");
    self.builder.reset(t);
}

void append_hmr_obj(PyCircuitBuilder &self, pybind11::object target_obj, pybind11::object output_obj) {
    QubitId t = obj_to_qubit_id(target_obj, "target");
    BitId o = obj_to_bit_id(output_obj, "output");
    self.builder.hmr(t, o);
}

void kickmix_py::register_circuit_builder_methods(pybind11::class_<PyCircuitBuilder> &c_circuit_builder) {
    c_circuit_builder.def(
        pybind11::init([]() -> PyCircuitBuilder {
            return PyCircuitBuilder();
        }),
        clean_doc_string(R"DOC(
            Initializes a new circuit builder.
        )DOC")
            .data());

    c_circuit_builder.def(
        "start_auto_free_scope",
        [](PyCircuitBuilder &self) {
            self.start_auto_free_scope();
        },
        clean_doc_string(R"DOC(
            Starts a scope that automatically frees allocations.

            Call builder.end_auto_free_scope() to free allocations performed during the
            scope. If multiple scopes are being used, they operate in LIFO order (i.e. like
            a stack).
        )DOC")
            .data());

    c_circuit_builder.def(
        "end_auto_free_scope",
        [](PyCircuitBuilder &self) {
            self.end_auto_free_scope();
        },
        clean_doc_string(R"DOC(
            Ends a scope that automatically frees allocations.

            Allocations since the corresponding builder.start_auto_free_scope() will be
            freed. If multiple scopes are being used, they operate in LIFO order (i.e. like
            a stack).
        )DOC")
            .data());

    c_circuit_builder.def_property_readonly(
        "max_qubits",
        [](PyCircuitBuilder &self) -> pybind11::object {
            if (self.max_qubits == SIZE_MAX) {
                return pybind11::cast(INFINITY);
            }
            return pybind11::cast(self.max_qubits);
        },
        clean_doc_string(R"DOC(
            Returns the maximum number of qubits allowed when building circuits.
        )DOC")
            .data());

    c_circuit_builder.def_property_readonly(
        "num_free_qubits",
        [](PyCircuitBuilder &self) -> pybind11::object {
            if (self.max_qubits == SIZE_MAX) {
                return pybind11::cast(INFINITY);
            }
            return pybind11::cast((int64_t)self.max_qubits - (int64_t)self.allocated_qubits);
        },
        clean_doc_string(R"DOC(
            Returns the maximum number of qubits allowed when building circuits.
        )DOC")
            .data());

    c_circuit_builder.def_property_readonly(
        "num_allocated_qubits",
        [](PyCircuitBuilder &self) {
            return self.allocated_qubits;
        },
        clean_doc_string(R"DOC(
            Returns the current number of allocated qubits.
        )DOC")
            .data());

    c_circuit_builder.def(
        "set_max_qubits",
        [](PyCircuitBuilder &self, pybind11::object limit) {
            size_t limit_val;
            if (pybind11::isinstance<pybind11::float_>(limit) && pybind11::cast<float>(limit) == INFINITY) {
                limit_val = SIZE_MAX;
            } else if (pybind11::isinstance<pybind11::int_>(limit)) {
                limit_val = pybind11::cast<pybind11::size_t>(limit);
            } else {
                throw std::invalid_argument("Expected 'limit' to be a non-negative integer or float('infinity').");
            }
            if (limit_val < self.allocated_qubits) {
                std::stringstream ss;
                ss << "Attempted to set max_qubits lower than the number of currently allocated qubits.";
                ss << "\n    allocated: " << self.allocated_qubits;
                ss << "\n    requested_max: ";
                if (limit_val == SIZE_MAX) {
                    ss << "inf";
                } else {
                    ss << limit_val;
                }
                ss << "\n    current_max: ";
                if (self.max_qubits == SIZE_MAX) {
                    ss << "inf";
                } else {
                    ss << self.max_qubits;
                }
                throw std::invalid_argument(ss.str());
            }
            self.max_qubits = limit_val;
        },
        pybind11::arg("limit"),
        clean_doc_string(R"DOC(
            @signature def set_max_qubits(limit: int | float) -> None:

            Sets the maximum number of qubits allowed when building circuits.

            Attempting to allocate more qubits than the maximum will cause the allocation
            to raise an exception. This includes both qubits returned by
            `CircuitBuilder.alloc_qubits` and by `CircuitBuilder.create_quantum_register`.

            Args:
                limit: The maximum number of qubits that can be allocated.
                    Set to `float('inf')` to disable the limit.
        )DOC")
            .data());

    c_circuit_builder.def(
        "alloc_qubits",
        [](PyCircuitBuilder &self, size_t count) -> PyArrayXZ {
            if (self.allocated_qubits + count > self.max_qubits) {
                std::stringstream ss;
                ss << "Attempted to allocate more qubits than the maximum.";
                ss << "\n    allocated: " << self.allocated_qubits;
                ss << "\n    requested: " << count;
                ss << "\n    allocated+requested: " << (self.allocated_qubits + count);
                ss << "\n    max: " << self.max_qubits;
                throw std::invalid_argument(ss.str());
            }

            array_xz result = array_xz::alloc_noinit(count, QXZTypeTag8::QUBIT_ID);
            for (size_t k = 0; k < count; k++) {
                result[k] = self.alloc_qubit();
            }
            PyArrayXZ wrapped;
            wrapped.owner = std::make_shared<array_xz>(std::move(result));
            wrapped.view = *wrapped.owner;
            return wrapped;
        },
        pybind11::arg("count"),
        R"DOC(
            Allocates clean qubits to use as workspace in a circuit.

            Allocated qubits can be deallocated by using the `CircuitBuilder.free` method.
            Qubits allocated within a `with builder.scope(auto_free=True):` block are
            deallocated automatically upon execution exiting the block.

            Although the qubits returned by this method *should* be clean, nothing *enforces*
            this. If a prior user of the allocated qubit failed to correctly clean some qubits
            before freeing them, the returned qubits may contain their buggy garbage. As such,
            it's recommended that you reset the qubits before using them. Reset gates also
            have the nice side effect of clearly indicating in circuit diagrams when a new
            usage of a qubit has begun.

            Args:
                length: The number of qubits to allocate.

            Returns:
                An array containing the ids of the allocated qubits. Until these ids are
                freed, they won't be returned by any other allocation calls.
        )DOC");

    c_circuit_builder.def(
        "alloc_bits",
        [](PyCircuitBuilder &self, size_t count) -> PyArrayXZ {
            array_xz result = array_xz::alloc_noinit(count, QXZTypeTag8::BIT_ID);
            for (size_t k = 0; k < count; k++) {
                result[k] = self.alloc_bit();
            }
            PyArrayXZ wrapped;
            wrapped.owner = std::make_shared<array_xz>(std::move(result));
            wrapped.view = *wrapped.owner;
            return wrapped;
        },
        pybind11::arg("count"),
        R"DOC(
        )DOC");

    c_circuit_builder.def(
        "free",
        [](PyCircuitBuilder &self, const pybind11::object &obj) {
            auto x = ConvertedArrayXZ::from_obj(obj, "arg");
            for (auto e : x.span) {
                if (e.is_qubit()) {
                    self.free_qubit((QubitId)e);
                } else if (e.is_bit()) {
                    self.free_bit((BitId)e);
                } else {
                    if (x.was_singleton) {
                        std::stringstream ss;
                        ss << "Don't know how to free ";
                        ss << pybind11::repr(obj);
                        ss << " (expected a qubit id or bit id).";
                        throw std::invalid_argument(ss.str());
                    } else {
                        std::stringstream ss;
                        ss << "Don't know how to free ";
                        ss << e;
                        ss << " from ";
                        ss << pybind11::repr(obj);
                        ss << " (expected qubit ids or bit ids).";
                        throw std::invalid_argument(ss.str());
                    }
                }
            }
        },
        pybind11::arg("arg"),
        R"DOC(
            Frees allocated qubits or bits, so that they can be allocated again.
        )DOC");

    c_circuit_builder.def(
        "create_quantum_register",
        [](PyCircuitBuilder &self, size_t length, std::string_view name) -> PyArrayXZ {
            if (self.allocated_qubits + length > self.max_qubits) {
                std::stringstream ss;
                ss << "Attempted to use more qubits than the maximum.";
                ss << "\n    allocated: " << self.allocated_qubits;
                ss << "\n    requested: " << length;
                ss << "\n    allocated+requested: " << (self.allocated_qubits + length);
                ss << "\n    max: " << self.max_qubits;
                throw std::invalid_argument(ss.str());
            }
            self.allocated_qubits += length;

            auto r = self.builder.append_register(length, name);
            array_xz result = array_xz::copy_of(r);
            PyArrayXZ wrapped;
            wrapped.owner = std::make_shared<array_xz>(std::move(result));
            wrapped.view = *wrapped.owner;
            return wrapped;
        },
        pybind11::arg("length"),
        pybind11::arg("name"),
        R"DOC(
            Creates a quantum register.

            Reserves the given number of qubits for the register, and emits instructions
            defining the register.

            Args:
                name: A name for the register, which can be used to refer to it.
                length: The number of qubits in the register.

            Returns:
                A kickmix.CircuitRegister containing the allocated qubits.
        )DOC");

    c_circuit_builder.def(
        "create_classical_register",
        [](PyCircuitBuilder &self, size_t length, std::string_view name) -> PyArrayXZ {
            auto r = self.builder.append_classical_register_qcarray_result(length, name);
            PyArrayXZ wrapped;
            wrapped.owner = std::make_shared<array_xz>(std::move(r));
            wrapped.view = *wrapped.owner;
            return wrapped;
        },
        pybind11::arg("length"),
        pybind11::arg("name"),
        R"DOC(
            Creates a classical register.

            Reserves the given number of bits for the register, and emits instructions
            defining the register.

            Args:
                name: A name for the register, which can be used to refer to it.
                length: The number of bits in the register.

            Returns:
                A kickmix.CircuitRegister containing the allocated qubits.
        )DOC");

    c_circuit_builder.def(
        "finish_circuit",
        [](PyCircuitBuilder &self) -> Circuit {
            return self.builder.finish_circuit();
        },
        R"DOC(
            Returns the circuit built by the circuit builder.
        )DOC");

    c_circuit_builder.def(
        "flame_chart_svg",
        [](PyCircuitBuilder &self) -> std::string {
            std::stringstream ss;
            self.builder.write_analysis_svg_to(ss, 100);
            return ss.str();
        },
        R"DOC(
            Returns a flame chart of CCX+CCZ and qubit utilization of the circuit being built.
        )DOC");

    c_circuit_builder.def(
        "x",
        &broadcast_x_obj,
        pybind11::arg("target"),
        R"DOC(
        )DOC");

    c_circuit_builder.def(
        "z",
        &append_z_obj,
        pybind11::arg("target"),
        R"DOC(
        )DOC");

    c_circuit_builder.def(
        "neg",
        &append_neg_obj,
        R"DOC(
        )DOC");

    c_circuit_builder.def(
        "neg_if",
        &append_neg_if_obj,
        pybind11::arg("condition"),
        R"DOC(
        )DOC");

    c_circuit_builder.def(
        "push_condition",
        &append_push_condition_obj,
        pybind11::arg("condition"),
        R"DOC(
        )DOC");

    c_circuit_builder.def(
        "pop_condition",
        &append_pop_condition_obj,
        R"DOC(
        )DOC");

    c_circuit_builder.def(
        "cx",
        &broadcast_cx_obj,
        pybind11::arg("control"),
        pybind11::arg("target"),
        R"DOC(
        )DOC");

    c_circuit_builder.def(
        "ccx",
        &broadcast_ccx_obj,
        pybind11::arg("control1"),
        pybind11::arg("control2"),
        pybind11::arg("target"),
        R"DOC(
            @signature def ccx(self, control1: bool | b | q | km.array, control2: bool | b | q | km.array, target: q | km.array) -> None:
            Appends many doubly-controlled NOT gates to the circuit.

            This method broadcasts over km.array arguments. If any of the arguments is a
            km.array, an append is performed for each array index. When multiple km.array
            arguments are used, they are combined by zipping (i.e. using two arrays of
            length n will produce n instructions rather than n*n instructions).

            The exact kind of operation(s) appended to the circuit depends on the types of
            the controls. For example, if one of the controls is the boolean value True
            instead of a qubit, then a CX instruction will be appended instead of a CCX
            instruction.

            Args:
                control1: One of the controls determining if the NOT gate happens or not.
                control2: The other control determining if the NOT gate happens or not.
                target: The qubit to bit flip when the controls are satisfied.

            Examples:
                >>> import kickmix as km

                >>> # Demo a simple ccx call
                >>> builder = km.CircuitBuilder()
                >>> builder.ccx(km.q(0), km.q(1), km.q(2))
                >>> builder.finish_circuit()
                km.Circuit('''
                    CCX q0 q1 q2
                ''')

                >>> # Demo a broadcasting call
                >>> builder = km.CircuitBuilder()
                >>> shared_control = builder.create_quantum_register(1, "shared_control")[0]
                >>> controls = builder.create_quantum_register(3, "controls")
                >>> targets = builder.create_quantum_register(3, "targets")[0]
                >>> builder.ccx(shared_control, controls, targets)
                >>> builder.finish_circuit()
                km.Circuit('''
                    APPEND_TO_REGISTER q0 r0
                    REGISTER r0 "shared_control"
                    APPEND_TO_REGISTER q1 r1
                    APPEND_TO_REGISTER q2 r1
                    APPEND_TO_REGISTER q3 r1
                    REGISTER r1 "controls"
                    APPEND_TO_REGISTER q4 r2
                    APPEND_TO_REGISTER q5 r2
                    APPEND_TO_REGISTER q6 r2
                    REGISTER r2 "targets"
                    CCX q0 q1 q4
                    CCX q0 q2 q4
                    CCX q0 q3 q4
                ''')
        )DOC");

    c_circuit_builder.def(
        "z_pow",
        &z_pow_obj,
        pybind11::arg("target"),
        pybind11::arg("exponent"),
        R"DOC(
            @signature def z_pow(self, target: km.q | km.array | Sequence[km.q], exponent: float | int | fractions.Fraction | str) -> None:
            Appends Z_POW gates to the circuit.

            This method broadcasts over km.array arguments. If the target is a sequence
            of qubits, rather than a single qubit, an operation is appended for each qubit.

            Args:
                target: The qubit(s) to apply Z_POW to.
                exponent: The exponent of the Z_POW operation.
                    The exponent is converted into a fractions.Fraction, then canonicalized
                    into the range [0, 2) and rounded to the nearest multiple of 2**-127.

            Examples:
                >>> import kickmix as km

                >>> # Demo a simple z_pow call
                >>> builder = km.CircuitBuilder()
                >>> builder.z_pow(km.q(0), 0.25)
                >>> builder.finish_circuit()
                km.Circuit('''
                    Z_POW q0 0.25
                ''')

                >>> # Demo a broadcasting call
                >>> builder = km.CircuitBuilder()
                >>> targets = builder.create_quantum_register(3, "targets")
                >>> builder.z_pow(targets, -0.5)
                >>> builder.finish_circuit()
                km.Circuit('''
                    APPEND_TO_REGISTER q0 r0
                    APPEND_TO_REGISTER q1 r0
                    APPEND_TO_REGISTER q2 r0
                    REGISTER r0 "targets"
                    Z_POW q0 1.5
                    Z_POW q1 1.5
                    Z_POW q2 1.5
                ''')
        )DOC");

    c_circuit_builder.def(
        "init_and",
        &broadcast_init_and_obj,
        pybind11::arg("control1"),
        pybind11::arg("control2"),
        pybind11::arg("target"),
        R"DOC(
            @signature def init_and(self, control1: bool | b | q | km.array, control2: bool | b | q | km.array, target: q | km.array) -> None:
            Initializes a qubit to `control1 and control2`, resetting it first.

            This is the computation partner of `del_and`. It resets `target`
            to |0> and then applies `CCX(control1, control2, target)`.

            This method broadcasts over km.array arguments.

            Args:
                control1: One of the controls of the AND.
                control2: The other control of the AND.
                target: The qubit initialized to the AND result.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> c1 = builder.alloc_qubits(1)[0]
                >>> c2 = builder.alloc_qubits(1)[0]
                >>> target = builder.alloc_qubits(1)[0]
                >>> builder.init_and(c1, c2, target)
                >>> builder.finish_circuit()
                km.Circuit('''
                    R q2
                    CCX q0 q1 q2
                ''')
        )DOC");

    c_circuit_builder.def(
        "del_and",
        &broadcast_del_and_obj,
        pybind11::arg("control1"),
        pybind11::arg("control2"),
        pybind11::arg("target"),
        R"DOC(
            @signature def del_and(self, control1: bool | b | q | km.array, control2: bool | b | q | km.array, target: q | km.array) -> None:
            Clears a qubit holding `control1 and control2`, using only Cliffords and measurement.

            This is the uncomputation partner of `init_and`, or of a `ccx` into
            a qubit that started at |0>. It applies an `HMR` (X-basis
            measurement and Z-basis reset to |0>) to `target`, followed by a
            phase correction `CZ(control1, control2)` conditioned on the
            measurement result being 1. The `target` qubit is left in |0> and is
            not freed, so the caller still decides when to `free` it.

            This method broadcasts over km.array arguments.

            Args:
                control1: One of the controls that was used to compute the target.
                control2: The other control that was used to compute the target.
                target: The qubit holding the AND, restored to |0>.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> c1 = builder.alloc_qubits(1)[0]
                >>> c2 = builder.alloc_qubits(1)[0]
                >>> target = builder.alloc_qubits(1)[0]
                >>> builder.del_and(c1, c2, target)
                >>> builder.finish_circuit()
                km.Circuit('''
                    HMR q2 b0
                    CZ q0 q1 if b0
                ''')
        )DOC");

    c_circuit_builder.def(
        "cz",
        &broadcast_cz_obj,
        pybind11::arg("control1"),
        pybind11::arg("control2"),
        R"DOC(
            @signature def cz(self, control1: bool | b | q | km.array, control2: bool | b | q | km.array) -> None:
            Appends many CZ gates to the circuit.

            This method broadcasts over km.array arguments. If any of the arguments is a
            km.array, an append is performed for each array index. When multiple km.array
            arguments are used, they are combined by zipping (i.e. using two arrays of
            length n will produce n instructions rather than n*n instructions).

            The exact kind of operation(s) appended to the circuit depends on the types of
            the controls. For example, if one of the controls is the boolean value True
            instead of a qubit, then a Z instruction will be appended instead of a CZ
            instruction.

            Args:
                control1: One of the controls determining if phasing occurs.
                control2: The other controls determining if phasing occurs.
        )DOC");

    c_circuit_builder.def(
        "swap",
        &broadcast_swap_obj,
        pybind11::arg("target1"),
        pybind11::arg("target2"),
        R"DOC(
        )DOC");

    c_circuit_builder.def(
        "cswap",
        &broadcast_cswap_obj,
        pybind11::arg("control"),
        pybind11::arg("target1"),
        pybind11::arg("target2"),
        R"DOC(
            @signature def cswap(self, control: q | b | bool, target1: km.array, target2: km.array) -> None:
            Appends operations to conditionally swap two equal-length registers.

            Performs `if control: target1, target2 = target2, target1`, pairing the
            registers index by index.

            The cost depends on the control. A qubit control expands to a
            cx/ccx/cx triple per index. A classical bit control pushes a
            classical condition around plain swaps, costing no Toffolis. A
            control of True reduces to a plain swap, and False emits nothing.

            Args:
                control: Determines if the swap happens.
                target1: The first register.
                target2: The second register, which must be the same length as the first.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> c = builder.create_quantum_register(1, name='c')[0]
                >>> x = builder.create_quantum_register(2, name='x')
                >>> y = builder.create_quantum_register(2, name='y')
                >>> builder.cswap(c, x, y)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -c[0]---@-----@---
                            |     |
                q1: -x[0]-@-X-@---|---
                          | | |   |
                q2: -x[1]-|-|-|-@-X-@-
                          | | | | | |
                q3: -y[0]-X-@-X-|-|-|-
                                | | |
                q4: -y[1]-------X-@-X-
        )DOC");

    c_circuit_builder.def(
        "bit_store",
        &broadcast_bit_store_obj,
        pybind11::arg("target"),
        pybind11::arg("value"),
        pybind11::kw_only(),
        pybind11::arg("control") = true,
        R"DOC(
            @signature def bit_store(self, target: b | km.array, value: bool, *, control: b | bool | km.array = True) -> None:
            Appends operations to perform `if control: target = value`.

            Writes a constant into classical bits. The control must be
            classical, since a classical bit cannot be coherently conditioned
            on a qubit.

            This method broadcasts over km.array arguments.

            Args:
                target: The classical bit (or bits) to write into.
                value: The constant to store.
                control: Defaults to True. Determines if the store occurs.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> a = builder.create_classical_register(1, name='a')[0]
                >>> b = builder.create_classical_register(1, name='b')[0]
                >>> out = builder.create_classical_register(1, name='out')[0]
                >>> # Compute logical OR: out = a | b
                >>> builder.bit_store(out, False)
                >>> builder.bit_store(out, True, control=km.array([a, b]))
                >>> builder.finish_circuit()
                km.Circuit('''
                    APPEND_TO_REGISTER b0 r0
                    REGISTER r0 "a"
                    APPEND_TO_REGISTER b1 r1
                    REGISTER r1 "b"
                    APPEND_TO_REGISTER b2 r2
                    REGISTER r2 "out"
                    BIT_STORE0 b2
                    BIT_STORE1 b2 if b0
                    BIT_STORE1 b2 if b1
                ''')
        )DOC");

    c_circuit_builder.def(
        "debug_print",
        &append_debug_print_obj,
        pybind11::arg("arg"),
        R"DOC(
        )DOC");

    c_circuit_builder.def(
        "reset",
        &broadcast_reset_obj,
        pybind11::arg("target"),
        R"DOC(
        )DOC");

    c_circuit_builder.def(
        "hmr",
        &broadcast_hmr_obj,
        pybind11::arg("target"),
        pybind11::arg("output"),
        R"DOC(
        )DOC");

    c_circuit_builder.def(
        "init_lookup",
        &append_init_lookup,
        pybind11::kw_only(),
        pybind11::arg("table"),
        pybind11::arg("address"),
        pybind11::arg("target"),
        R"DOC(
            @signature def init_lookup(self, *, table: Iterable[int], address: km.array, target: km.array) -> None:
            Appends operations to perform `let target := table[address]`.

            Looks up a value from a classical table, indexed by a quantum register.

            Because this is an `init` method, the target qubits must start in the 0 state.
            The target qubits will be reset before being used, transforming any bit errors
            into probabilistic phase errors.

            Args:
                table: The table of classical values, as a list of python integers.
                    The integers are interpreted in a little-endian fashion, meaning that
                    `table_data[address, out_index] == (table[address] >> out_index) & 1`.
                address: The address used to select a value from the table.
                target: The clean qubits to store the output into.
        )DOC");

    c_circuit_builder.def(
        "del_lookup",
        &append_del_lookup,
        pybind11::kw_only(),
        pybind11::arg("table"),
        pybind11::arg("address"),
        pybind11::arg("target"),
        R"DOC(
            @signature def del_lookup(self, *, table: Iterable[int], address: km.array, target: km.array) -> None:
            Appends operations to perform `del target := table[address]`.

            Clears a value created by looking up a value from a classical table, indexed by
            a quantum register.

            Because this is a `del` method, the target qubits will end in the 0 state.

            Args:
                table: The table of classical values, as a list of python integers.
                    The integers are interpreted in a little-endian fashion, meaning that
                    `table_data[address, out_index] == (table[address] >> out_index) & 1`.
                address: The address used to select a value from the table.
                    Qubits are in little-endian order.
                target: The qubits to clear the lookup result from.
        )DOC");

    c_circuit_builder.def(
        "iadd",
        &append_iadd_obj,
        pybind11::arg("offset"),
        pybind11::kw_only(),
        pybind11::arg("target"),
        pybind11::arg("control") = true,
        pybind11::arg("btol") = INFINITY,
        R"DOC(
            @signature def iadd(self, offset: km.array | int, *, target: km.array, control: q | b | bool = True, btol: float = float('inf')) -> None:
            Appends operations to perform `if control: target += offset`.

            Performs an inplace 2s-complement addition modulo 2**len(target).
            The offset can be a classical integer constant, classical bit
            register, or a quantum register.

            Args:
                offset: The little-endian value or integer to add. Can be
                    classical or quantum.
                target: The little-endian quantum register to add into.
                control: Defaults to True. Determines if the addition occurs.
                btol: Defaults to infinity (exact). Only applies to a classical
                    offset; passing a finite btol with a quantum offset is an
                    error. A finite value can lead to a cheaper circuit by not
                    carrying into the leading run of identical bits of offset,
                    so there's nothing to save unless that run is longer than
                    btol. In exchange, the sum can be wrong, but on no more than
                    a 2**-btol fraction of target values, assuming target is
                    uniformly random.

                    btol doesn't have to be an integer. Fractional values come up
                    naturally because approximation errors add up: doing k
                    approximate operations at btol + log2(k) each keeps the total
                    under 2**-btol.

                    A btol of len(target) or more is exact.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> a = builder.create_quantum_register(2, name='a')
                >>> b = builder.create_quantum_register(2, name='b')
                >>> builder.iadd(b, target=a)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -a[0]-@-X---
                          | |
                q1: -a[1]-X-|-X-
                          | | |
                q2: -b[0]-@-@-|-
                              |
                q3: -b[1]-----@-
        )DOC");

    c_circuit_builder.def(
        "isub",
        &append_isub_obj,
        pybind11::arg("offset"),
        pybind11::kw_only(),
        pybind11::arg("target"),
        pybind11::arg("control") = true,
        pybind11::arg("btol") = INFINITY,
        R"DOC(
            @signature def isub(self, offset: km.array | int, *, target: km.array, control: q | b | bool = True, btol: float = float('inf')) -> None:
            Appends operations to perform `if control: target -= offset`.

            Performs an inplace 2s-complement subtraction modulo 2**len(target).
            The offset can be a classical integer constant, classical bit
            register, or a quantum register.

            Args:
                offset: The little-endian value or integer to subtract. Can be
                    classical or quantum.
                target: The little-endian quantum register to subtract from.
                control: Defaults to True. Determines if the subtraction occurs.
                btol: Defaults to infinity (exact). Only applies to a classical
                    offset; passing a finite btol with a quantum offset is an
                    error. A finite value can lead to a cheaper circuit by not
                    borrowing into the leading run of identical bits of offset,
                    so there's nothing to save unless that run is longer than
                    btol. In exchange, the difference can be wrong, but on no
                    more than a 2**-btol fraction of target values, assuming
                    target is uniformly random.

                    btol doesn't have to be an integer. Fractional values come up
                    naturally because approximation errors add up: doing k
                    approximate operations at btol + log2(k) each keeps the total
                    under 2**-btol.

                    A btol of len(target) or more is exact.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> a = builder.create_quantum_register(2, name='a')
                >>> b = builder.create_quantum_register(2, name='b')
                >>> builder.isub(b, target=a)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -a[0]-X-@-X-X---
                            | |
                q1: -a[1]-X-X-|-X-X-
                            | | |
                q2: -b[0]---@-@-|---
                                |
                q3: -b[1]-------@---
        )DOC");

    c_circuit_builder.def(
        "left_rotate",
        &append_left_rotate_obj,
        pybind11::arg("target"),
        pybind11::arg("shift") = 1,
        pybind11::kw_only(),
        pybind11::arg("control") = true,
        R"DOC(
            @signature def left_rotate(self, target: km.array | Sequence[km.q], shift: int | km.array | Sequence[km.q | km.b | bool] = 1, *, control: km.q | km.b | bool = True) -> None:
            Left rotates the given qubits, cyclically permuting them.

            A left rotation is a permutation that moves the value of the qubit target[k]
            into the qubit target[(k + shift) % len(target)].

            Args:
                target: The qubits to permute.
                shift: Defaults to 1. A little-endian value specifying how much to
                    left-rotate the target qubits.
                control: Defaults to True. Determines if the left rotation actually occurs.

            Examples:
                >>> import kickmix as km

                >>> builder = km.CircuitBuilder()
                >>> target = builder.create_quantum_register(8, name='target')
                >>> builder.left_rotate(target)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -target[0]-SWAP-------------------------------
                               |
                q1: -target[1]-|----SWAP-----------SWAP-----------
                               |    |              |
                q2: -target[2]-|----|----SWAP------|----SWAP------
                               |    |    |         |    |
                q3: -target[3]-|----|----|----SWAP-|----|----SWAP-
                               |    |    |    |    |    |    |
                q4: -target[4]-|----|----|----SWAP-|----|----|----
                               |    |    |         |    |    |
                q5: -target[5]-|----|----SWAP------|----|----SWAP-
                               |    |              |    |
                q6: -target[6]-|----SWAP-----------|----SWAP------
                               |                   |
                q7: -target[7]-SWAP----------------SWAP-----------

                >>> builder = km.CircuitBuilder()
                >>> target = builder.create_quantum_register(8, name='target')
                >>> builder.left_rotate(target, 3)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -target[0]-SWAP-----------SWAP-----------
                               |              |
                q1: -target[1]-|----SWAP------|--------------
                               |    |         |
                q2: -target[2]-|----|----SWAP-SWAP-----------
                               |    |    |
                q3: -target[3]-|----|----|----SWAP-SWAP------
                               |    |    |    |    |
                q4: -target[4]-|----|----|----SWAP-|----SWAP-
                               |    |    |         |    |
                q5: -target[5]-|----|----SWAP------|----|----
                               |    |              |    |
                q6: -target[6]-|----SWAP-----------|----SWAP-
                               |                   |
                q7: -target[7]-SWAP----------------SWAP------

                >>> builder = km.CircuitBuilder()
                >>> target = builder.create_quantum_register(4, name='target')
                >>> control = builder.create_quantum_register(1, name='control')[0]
                >>> builder.left_rotate(target, 2, control=control)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -target[0]--@-X-@-------@-X-@-----
                                | | |       | | |
                q1: -target[1]--|-|-|-@-X-@-X-@-X-----
                                | | | | | |   |
                q2: -target[2]--|-|-|-X-@-X---|-@-X-@-
                                | | |   |     | | | |
                q3: -target[3]--X-@-X---|-----|-X-@-X-
                                  |     |     |   |
                q4: -control[0]---@-----@-----@---@---

                >>> builder = km.CircuitBuilder()
                >>> target = builder.create_quantum_register(6, name='target')
                >>> shift = builder.create_quantum_register(3, name='shift')
                >>> builder.left_rotate(target, shift)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -target[0]-----------@-X-@-----------@-X-@---------@-X-@-----------
                                         | | |           | | |         | | |
                q1: -target[1]-@-X-@-----X-@-X-----------|-|-|-@-X-@---X-@-X-----------
                               | | |       |             | | | | | |     |
                q2: -target[2]-|-|-|-@-X-@-|-@-X-@-------|-|-|-X-@-X-----|-@-X-@-------
                               | | | | | | | | | |       | | |   |       | | | |
                q3: -target[3]-|-|-|-|-|-|-|-|-|-|-@-X-@-X-@-X---|-------|-|-|-|-@-X-@-
                               | | | | | | | | | | | | |   |     |       | | | | | | |
                q4: -target[4]-|-|-|-X-@-X-|-|-|-|-X-@-X---|-----|-@-X-@-|-|-|-|-X-@-X-
                               | | |   |   | | | |   |     |     | | | | | | | |   |
                q5: -target[5]-X-@-X---|---|-X-@-X---|-----|-----|-X-@-X-|-X-@-X---|---
                                 |     |   |   |     |     |     |   |   |   |     |
                q6: -shift[0]----@-----@-@-|---|-----|-@---|-----|---|---|---|-----|---
                                         | |   |     | |   |     |   |   |   |     |
                q7: -shift[1]------------X-@---@-----@-X-@-|-----|---|-@-|---|-----|---
                                                         | |     |   | | |   |     |
                q8: -shift[2]----------------------------X-@-----@---@-X-@---@-----@---
        )DOC");

    c_circuit_builder.def(
        "right_rotate",
        &append_right_rotate_obj,
        pybind11::arg("target"),
        pybind11::arg("shift") = 1,
        pybind11::kw_only(),
        pybind11::arg("control") = true,
        R"DOC(
            @signature def right_rotate(self, target: km.array | Sequence[km.q], shift: int | km.array | Sequence[km.q | km.b | bool] = 1, *, control: km.q | km.b | bool = True) -> None:
            Right rotates the given qubits, cyclically permuting them.

            A right rotation is a permutation that moves the value of the qubit target[k]
            into the qubit target[(k - shift) % len(target)].

            Args:
                target: The qubits to permute.
                shift: Defaults to 1. A little-endian value specifying how much to
                    right-rotate the target qubits.
                control: Defaults to True. Determines if the right rotation actually occurs.

            Examples:
                >>> import kickmix as km

                >>> builder = km.CircuitBuilder()
                >>> target = builder.create_quantum_register(8, name='target')
                >>> builder.right_rotate(target)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -target[0]-SWAP----------------SWAP-----------
                               |                   |
                q1: -target[1]-|----SWAP-----------|----SWAP------
                               |    |              |    |
                q2: -target[2]-|----|----SWAP------|----|----SWAP-
                               |    |    |         |    |    |
                q3: -target[3]-|----|----|----SWAP-|----|----|----
                               |    |    |    |    |    |    |
                q4: -target[4]-|----|----|----SWAP-|----|----SWAP-
                               |    |    |         |    |
                q5: -target[5]-|----|----SWAP------|----SWAP------
                               |    |              |
                q6: -target[6]-|----SWAP-----------SWAP-----------
                               |
                q7: -target[7]-SWAP-------------------------------

                >>> builder = km.CircuitBuilder()
                >>> target = builder.create_quantum_register(8, name='target')
                >>> builder.right_rotate(target, 3)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -target[0]-SWAP----------------SWAP------
                               |                   |
                q1: -target[1]-|----SWAP-----------|----SWAP-
                               |    |              |    |
                q2: -target[2]-|----|----SWAP------|----|----
                               |    |    |         |    |
                q3: -target[3]-|----|----|----SWAP-|----SWAP-
                               |    |    |    |    |
                q4: -target[4]-|----|----|----SWAP-SWAP------
                               |    |    |
                q5: -target[5]-|----|----SWAP-----------SWAP-
                               |    |                   |
                q6: -target[6]-|----SWAP----------------|----
                               |                        |
                q7: -target[7]-SWAP---------------------SWAP-

                >>> builder = km.CircuitBuilder()
                >>> target = builder.create_quantum_register(4, name='target')
                >>> control = builder.create_quantum_register(1, name='control')[0]
                >>> builder.right_rotate(target, 2, control=control)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -target[0]--@-X-@-------@-X-@-----
                                | | |       | | |
                q1: -target[1]--|-|-|-@-X-@-X-@-X-----
                                | | | | | |   |
                q2: -target[2]--|-|-|-X-@-X---|-@-X-@-
                                | | |   |     | | | |
                q3: -target[3]--X-@-X---|-----|-X-@-X-
                                  |     |     |   |
                q4: -control[0]---@-----@-----@---@---

                >>> builder = km.CircuitBuilder()
                >>> target = builder.create_quantum_register(6, name='target')
                >>> shift = builder.create_quantum_register(3, name='shift')
                >>> builder.right_rotate(target, shift)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -target[0]-@-X-@---@-X-@-----------@-X-@-----------@-X-@-----------
                               | | |   | | |           | | |           | | |
                q1: -target[1]-|-|-|---X-@-X-----------|-|-|-@-X-@-----X-@-X-----------
                               | | |     |             | | | | | |       |
                q2: -target[2]-X-@-X-----|-@-X-@-------|-|-|-|-|-|-@-X-@-|-@-X-@-------
                                 |       | | | |       | | | | | | | | | | | | |
                q3: -target[3]---|-@-X-@-|-|-|-|-@-X-@-|-|-|-|-|-|-X-@-X-|-|-|-|-@-X-@-
                                 | | | | | | | | | | | | | | | | |   |   | | | | | | |
                q4: -target[4]---|-|-|-|-|-|-|-|-X-@-X-|-|-|-X-@-X---|---|-|-|-|-X-@-X-
                                 | | | | | | | |   |   | | |   |     |   | | | |   |
                q5: -target[5]---|-X-@-X-|-X-@-X---|---X-@-X---|-----|---|-X-@-X---|---
                                 |   |   |   |     |     |     |     |   |   |     |
                q6: -shift[0]----@---@-@-|---|-----|-@---|-----|-----|---|---|-----|---
                                       | |   |     | |   |     |     |   |   |     |
                q7: -shift[1]----------X-@---@-----@-X-@-|-----|-----|-@-|---|-----|---
                                                       | |     |     | | |   |     |
                q8: -shift[2]--------------------------X-@-----@-----@-X-@---@-----@---
        )DOC");

    c_circuit_builder.def(
        "multi_controlled_x",
        &multi_control_x_obj,
        pybind11::arg("controls"),
        pybind11::arg("target"),
        R"DOC(
        )DOC");

    c_circuit_builder.def(
        "flip_if_less_than",
        &append_flip_if_less_than_obj,
        pybind11::arg("lhs"),
        pybind11::arg("rhs"),
        pybind11::kw_only(),
        pybind11::arg("target"),
        pybind11::arg("or_equal") = false,
        pybind11::arg("control") = true,
        pybind11::arg("btol") = INFINITY,
        R"DOC(
            @signature def flip_if_less_than(self, lhs: km.array, rhs: km.array | int, *, target: q, or_equal: q | b | bool = false, control: q | b | bool = True, btol: float = float('inf')) -> None:
            Appends operations to perform `if control: target ^= lhs < rhs + or_equal`.

            Xors the result of an unsigned integer comparison into a target qubit.

            Args:
                lhs: The little-endian value on the left hand side of the comparison.
                rhs: The little-endian value or non-negative integer constant on the
                    right hand side of the comparison.
                target: The qubit to flip when lhs < rhs.
                or_equal: Defaults to False. Determines if the comparison is a
                    less-than-or-equal-to test rather than a less-than test.
                control: Defaults to True. Determines if the comparisons happens at all.
                btol: Defaults to infinity (exact). A finite value leads to a
                    cheaper circuit by only comparing the most significant bits,
                    since the rest only decide the answer when those tie. In
                    exchange, the answer can be wrong, but on no more than a
                    2**-btol fraction of inputs, assuming lhs and rhs are
                    independent and uniformly random. Holding one of them fixed
                    costs about a bit of that guarantee, and adversarial inputs
                    get no guarantee at all.

                    btol doesn't have to be an integer. Fractional values come up
                    naturally because approximation errors add up: doing k
                    approximate operations at btol + log2(k) each keeps the total
                    under 2**-btol.

                    A btol of len(lhs) or more is exact. A btol of 1 or less lets
                    the circuit skip the comparison entirely, since never
                    flipping the target is already right half the time.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> target = builder.create_quantum_register(1, name="target")[0]
                >>> lhs = builder.create_quantum_register(1, name="lhs")
                >>> rhs = builder.create_quantum_register(1, name="rhs")
                >>> builder.flip_if_less_than(lhs, rhs, target=target)
                >>> builder.finish_circuit()
                km.Circuit('''
                    APPEND_TO_REGISTER q0 r0
                    REGISTER r0 "target"
                    APPEND_TO_REGISTER q1 r1
                    REGISTER r1 "lhs"
                    APPEND_TO_REGISTER q2 r2
                    REGISTER r2 "rhs"
                    CX q2 q1
                    CCX q1 q2 q0
                    X q1
                    X q1
                    CX q2 q1
                ''')
        )DOC");

    c_circuit_builder.def(
        "flip_if_greater_than",
        &append_flip_if_greater_than_obj,
        pybind11::arg("lhs"),
        pybind11::arg("rhs"),
        pybind11::kw_only(),
        pybind11::arg("target"),
        pybind11::arg("or_equal") = false,
        pybind11::arg("control") = true,
        pybind11::arg("btol") = INFINITY,
        R"DOC(
            @signature def flip_if_greater_than(self, lhs: km.array, rhs: km.array | int, *, target: q, or_equal: q | b | bool = false, control: q | b | bool = True, btol: float = float('inf')) -> None:
            Appends operations to perform `if control: target ^= lhs > rhs - or_equal`.

            Xors the result of an unsigned integer comparison into a target qubit.

            Args:
                lhs: The little-endian value on the left hand side of the comparison.
                rhs: The little-endian value or non-negative integer constant on the
                    right hand side of the comparison.
                target: The qubit to flip when lhs > rhs.
                or_equal: Defaults to False. Determines if the comparison is a
                    greater-than-or-equal-to test rather than a greater-than test.
                control: Defaults to True. Determines if the comparisons happens at all.
                btol: Defaults to infinity (exact). A finite value leads to a
                    cheaper circuit by only comparing the most significant bits,
                    since the rest only decide the answer when those tie. In
                    exchange, the answer can be wrong, but on no more than a
                    2**-btol fraction of inputs, assuming lhs and rhs are
                    independent and uniformly random. Holding one of them fixed
                    costs about a bit of that guarantee, and adversarial inputs
                    get no guarantee at all.

                    btol doesn't have to be an integer. Fractional values come up
                    naturally because approximation errors add up: doing k
                    approximate operations at btol + log2(k) each keeps the total
                    under 2**-btol.

                    A btol of len(lhs) or more is exact. A btol of 1 or less lets
                    the circuit skip the comparison entirely, since never
                    flipping the target is already right half the time.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> t = builder.create_quantum_register(1, name='t')[0]
                >>> a = builder.create_quantum_register(2, name='a')
                >>> builder.flip_if_greater_than(a, 1, target=t)
                >>> print(builder.finish_circuit().text_diagram())
                                            push_cond if b0     pop_cond
                q0: -t[0]---------X----------------------------------------X-
                                  |         neg
                q1: -a[0]-X-X-----|-------------------------Z-Z----------X-X-
                                  |
                q2: -a[1]-X-------@----------------------------------------X-
                                  |
                q3:         |0>-X-@-HMR=b0
                >>> # Greater-than-or-equal-to comparison:
                >>> builder = km.CircuitBuilder()
                >>> t = builder.create_quantum_register(1, name='t')[0]
                >>> a = builder.create_quantum_register(2, name='a')
                >>> builder.flip_if_greater_than(a, 1, target=t, or_equal=True)
                >>> print(builder.finish_circuit().text_diagram())
                                              push_cond if b0   pop_cond
                q0: -t[0]-----------X--------------------------------------X-
                                    |         neg
                q1: -a[0]-X-X---@---|-------------------------Z----------X-X-
                                |   |
                q2: -a[1]-X-----|---@--------------------------------------X-
                                |   |
                q3:         |0>-X-X-@-HMR=b0
                >>> # Controlled comparison:
                >>> builder = km.CircuitBuilder()
                >>> c = builder.create_quantum_register(1, name='c')[0]
                >>> t = builder.create_quantum_register(1, name='t')[0]
                >>> a = builder.create_quantum_register(2, name='a')
                >>> builder.flip_if_greater_than(a, 1, target=t, control=c)
                >>> print(builder.finish_circuit().text_diagram())
                                                       push_cond if b0     pop_cond
                q0: -c[0]-----------@----------@--------------------------------------@-
                                    |          |       neg                            |
                q1: -t[0]-----------|-X--------|--------------------------------------X-
                                    | |        |
                q2: -a[0]-X-X-------|-|--------|-----------------------Z-Z----------X-X-
                                    | |        |
                q3: -a[1]-X---------@-|--------Z**b0----------------------------------X-
                                    | |
                q4:         |0>-X---|-@--------HMR=b0
                                    | |
                q5:             |0>-X-@-HMR=b0
                >>> # Quantum-quantum comparison:
                >>> builder = km.CircuitBuilder()
                >>> t = builder.create_quantum_register(1, name='t')[0]
                >>> a = builder.create_quantum_register(2, name='a')
                >>> b = builder.create_quantum_register(2, name='b')
                >>> builder.flip_if_greater_than(a, b, target=t)
                >>> print(builder.finish_circuit().text_diagram())
                                                            push_cond if b0     pop_cond
                q0: -t[0]-------------------X---X--------------------------------------------X-
                                            |   |
                q1: -a[0]-X-X-------@-------|---|---------------------------Z-@----------X-X---
                            |       |       |   |                             |          |
                q2: -a[1]-X-|-X-----|-------|---@-----------------------------|----------|-X-X-
                            | |     |       |   |                             |          | |
                q3: -b[0]---@-|---X-@-X-@---|---|---------------------------Z-Z----------@-|---
                              |     |   |   |   |                                          |
                q4: -b[1]-----@-----|---|-@-@-@-|-@----------------------------------------@---
                                    |   | | | | | |
                q5:           |0>---X---X-X-X-X-@-X-HMR=b0
                >>> # Bounded tolerance comparison (comparing top bits only):
                >>> builder = km.CircuitBuilder()
                >>> t = builder.create_quantum_register(1, name='t')[0]
                >>> a = builder.create_quantum_register(4, name='a')
                >>> builder.flip_if_greater_than(a, 7, target=t, btol=2)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -t[0]---X-X-
                            |
                q1: -a[0]---|---
                            |
                q2: -a[1]---|---
                            |
                q3: -a[2]---|---
                            |
                q4: -a[3]-X-@-X-
        )DOC");

    c_circuit_builder.def(
        "flip_if_equal",
        &append_flip_if_equal_obj,
        pybind11::arg("lhs"),
        pybind11::arg("rhs"),
        pybind11::kw_only(),
        pybind11::arg("target"),
        pybind11::arg("control") = true,
        R"DOC(
            @signature def flip_if_equal(self, lhs: km.array, rhs: km.array | int, *, target: q, control: q | b | bool = True) -> None:
            Appends operations to perform `if control: target ^= lhs == rhs`.

            Xors the result of an unsigned integer comparison into a target qubit.

            Args:
                lhs: The little-endian value on the left hand side of the comparison.
                rhs: The little-endian value or non-negative integer constant on the
                    right hand side of the comparison.
                target: The qubit to flip when lhs == rhs.
                control: Defaults to True. Determines if the comparisons happens at all.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> t = builder.create_quantum_register(1, name='t')[0]
                >>> a = builder.create_quantum_register(2, name='a')
                >>> builder.flip_if_equal(a, 2, target=t)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -t[0]-------------X-------------------------------
                                      |
                q1: -a[0]-X-----@-----|---------------------Z**b0-X---
                                |     |
                q2: -a[1]-X-X---|---@-@--------Z**b0--------------X-X-
                                |   | |        |
                q3:         |0>-X---@-|--------@-----HMR=b0
                                    | |
                q4:             |0>-X-@-HMR=b0
                >>> # Quantum-quantum comparison:
                >>> builder = km.CircuitBuilder()
                >>> t = builder.create_quantum_register(1, name='t')[0]
                >>> a = builder.create_quantum_register(2, name='a')
                >>> b = builder.create_quantum_register(2, name='b')
                >>> builder.flip_if_equal(a, b, target=t)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -t[0]---------------X---------------------------------
                                        |
                q1: -a[0]-X-X-----@-----|---------------------Z**b0-X-X---
                          |       |     |                             |
                q2: -a[1]-|-X-X---|---@-@--------Z**b0--------------X-|-X-
                          | |     |   | |        |                    | |
                q3: -b[0]-@-|-----|---|-|--------|--------------------@-|-
                            |     |   | |        |                      |
                q4: -b[1]---@-----|---|-|--------|----------------------@-
                                  |   | |        |
                q5:           |0>-X---@-|--------@-----HMR=b0
                                      | |
                q6:               |0>-X-@-HMR=b0
        )DOC");

    c_circuit_builder.def(
        "gf2_iadd",
        &append_gf2_iadd_obj,
        pybind11::arg("offset"),
        pybind11::kw_only(),
        pybind11::arg("target"),
        pybind11::arg("field") = pybind11::none(),
        pybind11::arg("control") = true,
        clean_doc_string(R"DOC(
            @signature def gf2_iadd(self, offset: km.array | int, *, target: km.array, field: km.GF2Field | int | None = None, control: q | b | bool = True) -> None:
            Appends operations to perform `if control: target ^= offset` in GF(2^m).

            Addition in a binary extension field is bitwise XOR, realized as a
            layer of CNOTs (or Toffolis when quantum-controlled, or NOTs/CNOTs
            when adding a classical constant).

            Args:
                offset: The register or classical constant (int) to add. Left unchanged.
                target: The register to add into in place.
                field: Optional field or modulus polynomial. If provided, its
                    degree must equal `len(target)`.
                control: Defaults to True. Determines if the addition occurs.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> target = builder.create_quantum_register(2, name='target')
                >>> offset = builder.create_quantum_register(2, name='offset')
                >>> builder.gf2_iadd(offset, target=target)
                >>> builder.gf2_iadd(1, target=target)
                >>> builder.finish_circuit()
                km.Circuit('''
                    APPEND_TO_REGISTER q0 r0
                    APPEND_TO_REGISTER q1 r0
                    REGISTER r0 "target"
                    APPEND_TO_REGISTER q2 r1
                    APPEND_TO_REGISTER q3 r1
                    REGISTER r1 "offset"
                    CX q2 q0
                    CX q3 q1
                    X q0
                ''')
        )DOC")
            .data());

    c_circuit_builder.def(
        "gf2_imul",
        &append_gf2_imul_obj,
        pybind11::arg("factor"),
        pybind11::kw_only(),
        pybind11::arg("target"),
        pybind11::arg("field") = pybind11::none(),
        clean_doc_string(R"DOC(
            @signature def gf2_imul(self, factor: int, *, target: km.array, field: km.GF2Field | int | None = None) -> None:
            Appends operations to perform `target = (target * factor) % modulus`.

            Multiplies a GF(2^m) register in place by a non-zero classical factor
            using only CNOT and SWAP gates (no ancillas or Toffolis).

            Currently only classical integer factors are supported; passing a
            non-classical input raises `NotImplementedError`.

            Args:
                factor: The non-zero classical field element to multiply by.
                target: The GF(2^m) register to multiply in place.
                field: Optional `km.GF2Field` or irreducible polynomial int.
                    Defaults to `km.GF2Field(len(target))`.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> target = builder.create_quantum_register(2, name='target')
                >>> builder.gf2_imul(2, target=target)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -target[0]-X-SWAP-
                               | |
                q1: -target[1]-@-SWAP-
        )DOC")
            .data());

    c_circuit_builder.def(
        "gf2_idiv",
        &append_gf2_idiv_obj,
        pybind11::arg("divisor"),
        pybind11::kw_only(),
        pybind11::arg("target"),
        pybind11::arg("field") = pybind11::none(),
        clean_doc_string(R"DOC(
            @signature def gf2_idiv(self, divisor: int, *, target: km.array, field: km.GF2Field | int | None = None) -> None:
            Appends operations to perform `target = (target / divisor) % modulus`.

            Divides a GF(2^m) register in place by a non-zero classical divisor.
            This is the exact inverse of `gf2_imul`.

            Currently only classical integer divisors are supported; passing a
            non-classical input raises `NotImplementedError`.

            Args:
                divisor: The non-zero classical field element to divide by.
                target: The GF(2^m) register to divide in place.
                field: Optional `km.GF2Field` or irreducible polynomial int.
                    Defaults to `km.GF2Field(len(target))`.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> target = builder.create_quantum_register(2, name='target')
                >>> builder.gf2_idiv(2, target=target)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -target[0]-X-@-
                               | |
                q1: -target[1]-@-X-
        )DOC")
            .data());

    c_circuit_builder.def(
        "init_gf2_mul",
        &append_init_gf2_mul_obj,
        pybind11::arg("lhs"),
        pybind11::arg("rhs"),
        pybind11::kw_only(),
        pybind11::arg("target") = "alloc",
        pybind11::arg("field") = pybind11::none(),
        pybind11::arg("control") = true,
        clean_doc_string(R"DOC(
            @signature def init_gf2_mul(self, lhs: km.array, rhs: km.array, *, target: km.array | str = "alloc", field: km.GF2Field | int | None = None, control: q | b | bool = True) -> km.array:
            Initializes a GF(2^m) register to `(lhs * rhs) % modulus`, resetting it first.

            This is the computation partner of `del_gf2_mul`. Resets `target`
            to |0> and then multiplies `lhs` and `rhs` into `target`.

            Args:
                lhs: The first factor register. Left unchanged.
                rhs: The second factor register. Left unchanged.
                target: The GF(2^m) register to initialize. Defaults to "alloc",
                    which allocates a register of the field's degree.
                field: Optional `km.GF2Field` or irreducible polynomial int.
                    Defaults to `km.GF2Field(len(lhs))`.
                control: Defaults to True. Determines if the operation occurs.

            Returns:
                The initialized register.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> lhs = builder.create_quantum_register(2, name='lhs')
                >>> rhs = builder.create_quantum_register(2, name='rhs')
                >>> target = builder.init_gf2_mul(lhs, rhs)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -lhs[0]---X----@-X---------------@-----
                              |    | |               |
                q1: -lhs[1]---@----|-@------@--------|-----
                                   |        |        |
                q2: -rhs[0]---X----@-X------|--------@-----
                              |    | |      |        |
                q3: -rhs[1]---@----|-@------@--------|-----
                                   |        |        |
                q4:  |0>----@-SWAP-X-X-SWAP-X-X-SWAP-X-X-@-
                            | |      | |      | |      | |
                q5:  |0>----X-SWAP---@-SWAP---@-SWAP---@-X-
        )DOC")
            .data());

    c_circuit_builder.def(
        "ixor_gf2_mul",
        &append_ixor_gf2_mul_obj,
        pybind11::arg("lhs"),
        pybind11::arg("rhs"),
        pybind11::kw_only(),
        pybind11::arg("target"),
        pybind11::arg("field") = pybind11::none(),
        pybind11::arg("control") = true,
        clean_doc_string(R"DOC(
            @signature def ixor_gf2_mul(self, lhs: km.array, rhs: km.array, *, target: km.array, field: km.GF2Field | int | None = None, control: q | b | bool = True) -> None:
            Appends operations to perform `if control: target ^= (lhs * rhs) % modulus`.

            Out-of-place multiplication accumulating the product into `target`
            via bitwise XOR. Does not reset `target` or assume it is in the
            |0> state.

            Args:
                lhs: The first factor register. Left unchanged.
                rhs: The second factor register. Left unchanged.
                target: The GF(2^m) register to XOR the product into.
                field: Optional `km.GF2Field` or irreducible polynomial int.
                    Defaults to `km.GF2Field(len(target))`.
                control: Defaults to True. Determines if the operation occurs.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> target = builder.create_quantum_register(2, name='target')
                >>> lhs = builder.create_quantum_register(2, name='lhs')
                >>> rhs = builder.create_quantum_register(2, name='rhs')
                >>> builder.ixor_gf2_mul(lhs, rhs, target=target)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -target[0]-@-SWAP-X-X-SWAP-X-X-SWAP-X-X-@-
                               | |    | | |    | | |    | | |
                q1: -target[1]-X-SWAP-|-@-SWAP-|-@-SWAP-|-@-X-
                                      |        |        |
                q2: -lhs[0]------X----@-X------|--------@-----
                                 |    | |      |        |
                q3: -lhs[1]------@----|-@------@--------|-----
                                      |        |        |
                q4: -rhs[0]------X----@-X------|--------@-----
                                 |      |      |
                q5: -rhs[1]------@------@------@--------------
        )DOC")
            .data());

    c_circuit_builder.def(
        "del_gf2_mul",
        &append_del_gf2_mul_obj,
        pybind11::arg("lhs"),
        pybind11::arg("rhs"),
        pybind11::kw_only(),
        pybind11::arg("target"),
        pybind11::arg("field") = pybind11::none(),
        pybind11::arg("control") = true,
        clean_doc_string(R"DOC(
            @signature def del_gf2_mul(self, lhs: km.array, rhs: km.array, *, target: km.array, field: km.GF2Field | int | None = None, control: q | b | bool = True) -> None:
            Clears a GF(2^m) register known to hold `(lhs * rhs) % modulus`.

            This is the uncomputation partner of `init_gf2_mul`. When
            uncontrolled, uncomputes `target` back to 0 using X-basis
            measurements and classically conditioned CZ fixups (zero Toffolis).

            Args:
                lhs: The first factor register. Left unchanged.
                rhs: The second factor register. Left unchanged.
                target: The GF(2^m) register holding `lhs * rhs`, cleared to 0.
                field: Optional `km.GF2Field` or irreducible polynomial int.
                    Defaults to `km.GF2Field(len(target))`.
                control: Defaults to True. Determines if the operation occurs.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> lhs = builder.create_quantum_register(1, name='lhs')
                >>> rhs = builder.create_quantum_register(1, name='rhs')
                >>> target = builder.init_gf2_mul(lhs, rhs)
                >>> builder.del_gf2_mul(lhs, rhs, target=target)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -lhs[0]-@--------@-----
                            |        |
                q1: -rhs[0]-@--------Z**b0-
                            |
                q2:  |0>----X-HMR=b0
        )DOC")
            .data());

    c_circuit_builder.def(
        "gf2_phase_by_product",
        &append_gf2_phase_by_product_obj,
        pybind11::arg("lhs"),
        pybind11::arg("rhs"),
        pybind11::kw_only(),
        pybind11::arg("mask"),
        pybind11::arg("field") = pybind11::none(),
        clean_doc_string(R"DOC(
            @signature def gf2_phase_by_product(self, lhs: km.array, rhs: km.array, *, mask: km.array, field: km.GF2Field | int | None = None) -> None:
            Negates amplitudes where `popcount(mask & (lhs * rhs) % modulus)` is odd.

            Kicks back the phase of the product directly without computing it
            into an intermediate register, using zero Toffolis and zero
            ancillas.

            Args:
                lhs: The first factor register. Left unchanged.
                rhs: The second factor register. Left unchanged.
                mask: Classical bits selecting which product bits contribute.
                field: Optional `km.GF2Field` or irreducible polynomial int.
                    Defaults to `km.GF2Field(len(lhs))`.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> lhs = builder.create_quantum_register(2, name='lhs')
                >>> rhs = builder.create_quantum_register(2, name='rhs')
                >>> mask = builder.create_classical_register(2, name='mask')
                >>> builder.gf2_phase_by_product(lhs, rhs, mask=mask)
                >>> print(builder.finish_circuit().text_diagram())
                            mask[0]=b0             b2^=1 if b0       b2^=1 if b0
                q0: -lhs[0]------------@-----------------------------------------
                            mask[1]=b1 |           b2^=1 if b1       b2^=1 if b1
                q1: -lhs[1]------------|-----@-----------------@-----------------
                                       |     |                 |
                q2: -rhs[0]------------Z**b0-Z**b1-------------|-----------------
                                       |                       |
                q3: -rhs[1]------------Z**b1-------------------Z**b2-------------
        )DOC")
            .data());

    c_circuit_builder.def(
        "init_gf2_inverse",
        &append_init_gf2_inverse_obj,
        pybind11::arg("input"),
        pybind11::kw_only(),
        pybind11::arg("target") = "alloc",
        pybind11::arg("field") = pybind11::none(),
        clean_doc_string(R"DOC(
            @signature def init_gf2_inverse(self, input: km.array, *, target: km.array | str = "alloc", field: km.GF2Field | int | None = None) -> km.array:
            Initializes `target := input ** -1` in GF(2^m) (mapping 0 to 0).

            This is the computation partner of `del_gf2_inverse`. Resets
            `target` to |0> and then writes the inverse into it, using the
            Itoh-Tsujii addition chain. The chain's intermediates are temporary
            workspace, allocated and cleaned up automatically. Use
            `init_gf2_inverse_with_scaffold` to keep them instead, which makes
            the matching uncomputation free of Toffolis.

            Args:
                input: The GF(2^m) register to invert. Left unchanged.
                target: The register to store the inverse into. Defaults to
                    "alloc", which allocates a register of the field's degree.
                field: Optional `km.GF2Field` or irreducible polynomial int.
                    Defaults to `km.GF2Field(len(input))`.

            Returns:
                The initialized register.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> inp = builder.create_quantum_register(2, name='inp')
                >>> inv = builder.init_gf2_inverse(inp)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -inp[0]-@-----
                            |
                q1: -inp[1]-|-@---
                            | |
                q2:  |0>----X-|-X-
                              | |
                q3:  |0>------X-@-
        )DOC")
            .data());

    c_circuit_builder.def(
        "init_gf2_inverse_with_scaffold",
        &append_init_gf2_inverse_with_scaffold_obj,
        pybind11::arg("input"),
        pybind11::kw_only(),
        pybind11::arg("target") = "alloc",
        pybind11::arg("field") = pybind11::none(),
        clean_doc_string(R"DOC(
            @signature def init_gf2_inverse_with_scaffold(self, input: km.array, *, target: km.array | str = "alloc", field: km.GF2Field | int | None = None) -> tuple[km.array, km.array]:
            Initializes `target := input ** -1`, keeping the addition chain.

            Same as `init_gf2_inverse`, except the Itoh-Tsujii addition chain
            intermediates are kept rather than uncomputed. Keeping them lets
            `del_gf2_inverse_with_scaffold` undo the whole thing with zero
            Toffolis, which is cheaper than recomputing the chain.

            This is the computation partner of `del_gf2_inverse_with_scaffold`.

            Args:
                input: The GF(2^m) register to invert. Left unchanged.
                target: The register to store the inverse into. Defaults to
                    "alloc", which allocates a register of the field's degree.
                field: Optional `km.GF2Field` or irreducible polynomial int.
                    Defaults to `km.GF2Field(len(input))`.

            Returns:
                A tuple of the initialized register and the scaffold register
                holding the addition chain.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> inp = builder.create_quantum_register(8, name='inp')
                >>> inv, scaffold = builder.init_gf2_inverse_with_scaffold(inp)
                >>> len(scaffold)
                24
                >>> builder.finish_circuit().max_magic()
                108
        )DOC")
            .data());

    c_circuit_builder.def(
        "del_gf2_inverse",
        &append_del_gf2_inverse_obj,
        pybind11::arg("input"),
        pybind11::kw_only(),
        pybind11::arg("target"),
        pybind11::arg("field") = pybind11::none(),
        clean_doc_string(R"DOC(
            @signature def del_gf2_inverse(self, input: km.array, *, target: km.array, field: km.GF2Field | int | None = None) -> None:
            Clears `target` (holding `input ** -1` in GF(2^m)) back to 0.

            This is the uncomputation partner of `init_gf2_inverse`. Temporary
            workspace is allocated to rebuild and uncompute the addition chain.
            If you still hold the scaffold from
            `init_gf2_inverse_with_scaffold`, use
            `del_gf2_inverse_with_scaffold` instead to avoid the rebuild.

            Args:
                input: The inverted GF(2^m) register. Left unchanged.
                target: The register holding `input ** -1`, cleared to 0.
                field: Optional `km.GF2Field` or irreducible polynomial int.
                    Defaults to `km.GF2Field(len(target))`.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> inp = builder.create_quantum_register(8, name='inp')
                >>> inv = builder.init_gf2_inverse(inp)
                >>> builder.del_gf2_inverse(inp, target=inv)
                >>> builder.finish_circuit().max_magic()  # 108 to build, 108 to rebuild
                216
        )DOC")
            .data());

    c_circuit_builder.def(
        "del_gf2_inverse_with_scaffold",
        &append_del_gf2_inverse_with_scaffold_obj,
        pybind11::arg("input"),
        pybind11::kw_only(),
        pybind11::arg("target"),
        pybind11::arg("scaffold"),
        pybind11::arg("field") = pybind11::none(),
        clean_doc_string(R"DOC(
            @signature def del_gf2_inverse_with_scaffold(self, input: km.array, *, target: km.array, scaffold: km.array, field: km.GF2Field | int | None = None) -> None:
            Clears `target` and its scaffold, using only Cliffords and measurement.

            This is the uncomputation partner of
            `init_gf2_inverse_with_scaffold`. Because the addition chain is
            still live, both `target` and `scaffold` are cleared to 0 without
            any Toffolis. Neither register is freed, so the caller still
            decides when to `free` them.

            Args:
                input: The inverted GF(2^m) register. Left unchanged.
                target: The register holding `input ** -1`, cleared to 0.
                scaffold: The addition chain from
                    `init_gf2_inverse_with_scaffold`, also cleared to 0.
                field: Optional `km.GF2Field` or irreducible polynomial int.
                    Defaults to `km.GF2Field(len(target))`.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> inp = builder.create_quantum_register(8, name='inp')
                >>> inv, scaffold = builder.init_gf2_inverse_with_scaffold(inp)
                >>> builder.del_gf2_inverse_with_scaffold(
                ...     inp, target=inv, scaffold=scaffold)
                >>> builder.finish_circuit().max_magic()  # no more than building it once
                108
        )DOC")
            .data());

    c_circuit_builder.def(
        "init_gf2_div",
        &append_init_gf2_div_obj,
        pybind11::arg("lhs"),
        pybind11::arg("rhs"),
        pybind11::kw_only(),
        pybind11::arg("target") = "alloc",
        pybind11::arg("field") = pybind11::none(),
        clean_doc_string(R"DOC(
            @signature def init_gf2_div(self, lhs: km.array, rhs: km.array, *, target: km.array | str = "alloc", field: km.GF2Field | int | None = None) -> km.array:
            Initializes a GF(2^m) register to `(lhs / rhs) % modulus`, resetting it first.

            This is the computation partner of `del_gf2_div`. Resets `target`
            to |0> and then writes `lhs / rhs` into it. A zero divisor produces
            a quotient of 0. The divisor's inverse and its addition chain are
            temporary workspace, allocated and cleaned up automatically. Use
            `init_gf2_div_with_scaffold` to keep them instead.

            Args:
                lhs: The dividend register. Left unchanged.
                rhs: The divisor register. Left unchanged.
                target: The GF(2^m) register to initialize. Defaults to "alloc",
                    which allocates a register of the field's degree.
                field: Optional `km.GF2Field` or irreducible polynomial int.
                    Defaults to `km.GF2Field(len(lhs))`.

            Returns:
                The initialized register.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> lhs = builder.create_quantum_register(8, name='lhs')
                >>> rhs = builder.create_quantum_register(8, name='rhs')
                >>> quotient = builder.init_gf2_div(lhs, rhs)
                >>> builder.finish_circuit().max_magic()
                135
        )DOC")
            .data());

    c_circuit_builder.def(
        "init_gf2_div_with_scaffold",
        &append_init_gf2_div_with_scaffold_obj,
        pybind11::arg("lhs"),
        pybind11::arg("rhs"),
        pybind11::kw_only(),
        pybind11::arg("target") = "alloc",
        pybind11::arg("field") = pybind11::none(),
        clean_doc_string(R"DOC(
            @signature def init_gf2_div_with_scaffold(self, lhs: km.array, rhs: km.array, *, target: km.array | str = "alloc", field: km.GF2Field | int | None = None) -> tuple[km.array, km.array]:
            Initializes `target := lhs / rhs`, keeping the division workspace.

            Same as `init_gf2_div`, except the divisor's inverse and its
            addition chain are kept rather than uncomputed. Keeping them lets
            `del_gf2_div_with_scaffold` undo the whole thing with zero
            Toffolis, which is cheaper than recomputing the inverse.

            This is the computation partner of `del_gf2_div_with_scaffold`.

            Args:
                lhs: The dividend register. Left unchanged.
                rhs: The divisor register. Left unchanged.
                target: The GF(2^m) register to initialize. Defaults to "alloc",
                    which allocates a register of the field's degree.
                field: Optional `km.GF2Field` or irreducible polynomial int.
                    Defaults to `km.GF2Field(len(lhs))`.

            Returns:
                A tuple of the initialized register and the scaffold register
                holding the divisor's inverse and addition chain.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> lhs = builder.create_quantum_register(8, name='lhs')
                >>> rhs = builder.create_quantum_register(8, name='rhs')
                >>> quotient, scaffold = builder.init_gf2_div_with_scaffold(lhs, rhs)
                >>> len(scaffold)
                32
                >>> builder.finish_circuit().max_magic()
                135
        )DOC")
            .data());

    c_circuit_builder.def(
        "ixor_gf2_div",
        &append_ixor_gf2_div_obj,
        pybind11::arg("lhs"),
        pybind11::arg("rhs"),
        pybind11::kw_only(),
        pybind11::arg("target"),
        pybind11::arg("field") = pybind11::none(),
        clean_doc_string(R"DOC(
            @signature def ixor_gf2_div(self, lhs: km.array, rhs: km.array, *, target: km.array, field: km.GF2Field | int | None = None) -> None:
            Appends operations to perform `target ^= (lhs / rhs) % modulus`.

            Out-of-place division accumulating the quotient into `target` via
            bitwise XOR. Does not reset `target` or assume it is in the |0>
            state. A zero divisor produces a quotient of 0. Temporary workspace
            is allocated and cleaned up automatically.

            Args:
                lhs: The dividend register. Left unchanged.
                rhs: The divisor register. Left unchanged.
                target: The GF(2^m) register to XOR the quotient into.
                field: Optional `km.GF2Field` or irreducible polynomial int.
                    Defaults to `km.GF2Field(len(target))`.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> lhs = builder.create_quantum_register(8, name='lhs')
                >>> rhs = builder.create_quantum_register(8, name='rhs')
                >>> target = builder.create_quantum_register(8, name='target')
                >>> builder.ixor_gf2_div(lhs, rhs, target=target)
                >>> builder.finish_circuit().max_magic()
                135
        )DOC")
            .data());

    c_circuit_builder.def(
        "del_gf2_div",
        &append_del_gf2_div_obj,
        pybind11::arg("lhs"),
        pybind11::arg("rhs"),
        pybind11::kw_only(),
        pybind11::arg("target"),
        pybind11::arg("field") = pybind11::none(),
        clean_doc_string(R"DOC(
            @signature def del_gf2_div(self, lhs: km.array, rhs: km.array, *, target: km.array, field: km.GF2Field | int | None = None) -> None:
            Clears a GF(2^m) register known to hold `(lhs / rhs) % modulus` to 0.

            This is the uncomputation partner of `init_gf2_div`. Temporary
            workspace is allocated to recompute the inverse. If you still hold
            the scaffold from `init_gf2_div_with_scaffold`, use
            `del_gf2_div_with_scaffold` instead to avoid the recompute.

            Args:
                lhs: The dividend register. Left unchanged.
                rhs: The divisor register. Left unchanged.
                target: The register holding `lhs / rhs`, cleared to 0.
                field: Optional `km.GF2Field` or irreducible polynomial int.
                    Defaults to `km.GF2Field(len(target))`.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> lhs = builder.create_quantum_register(8, name='lhs')
                >>> rhs = builder.create_quantum_register(8, name='rhs')
                >>> quotient = builder.init_gf2_div(lhs, rhs)
                >>> builder.del_gf2_div(lhs, rhs, target=quotient)
                >>> builder.finish_circuit().max_magic()  # 135 to build, 108 to rebuild
                243
        )DOC")
            .data());

    c_circuit_builder.def(
        "del_gf2_div_with_scaffold",
        &append_del_gf2_div_with_scaffold_obj,
        pybind11::arg("lhs"),
        pybind11::arg("rhs"),
        pybind11::kw_only(),
        pybind11::arg("target"),
        pybind11::arg("scaffold"),
        pybind11::arg("field") = pybind11::none(),
        clean_doc_string(R"DOC(
            @signature def del_gf2_div_with_scaffold(self, lhs: km.array, rhs: km.array, *, target: km.array, scaffold: km.array, field: km.GF2Field | int | None = None) -> None:
            Clears `target` and its scaffold, using only Cliffords and measurement.

            This is the uncomputation partner of `init_gf2_div_with_scaffold`.
            Because the divisor's inverse is still live, both `target` and
            `scaffold` are cleared to 0 without any Toffolis. Neither register
            is freed, so the caller still decides when to `free` them.

            Args:
                lhs: The dividend register. Left unchanged.
                rhs: The divisor register. Left unchanged.
                target: The register holding `lhs / rhs`, cleared to 0.
                scaffold: The workspace from `init_gf2_div_with_scaffold`, also
                    cleared to 0.
                field: Optional `km.GF2Field` or irreducible polynomial int.
                    Defaults to `km.GF2Field(len(target))`.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()
                >>> lhs = builder.create_quantum_register(8, name='lhs')
                >>> rhs = builder.create_quantum_register(8, name='rhs')
                >>> quotient, scaffold = builder.init_gf2_div_with_scaffold(lhs, rhs)
                >>> builder.del_gf2_div_with_scaffold(
                ...     lhs, rhs, target=quotient, scaffold=scaffold)
                >>> builder.finish_circuit().max_magic()  # no more than building it once
                135
        )DOC")
            .data());

    c_circuit_builder.def(
        "unary_iteration",
        &make_unary_iteration,
        pybind11::arg("address"),
        pybind11::arg("address_values") = pybind11::none(),
        pybind11::kw_only(),
        pybind11::arg("control") = true,
        pybind11::keep_alive<0, 1>(),
        clean_doc_string(R"DOC(
            @signature def unary_iteration(self, address: km.array, address_values: range | None = None, *, control: q | b | bool = True) -> km.UnaryIterationCursor:
            Returns a cursor for stepping through values of `address`.

            Use `with builder.unary_iteration(...) as it:` to open a pass. Inside the
            `with` block, loop over `it` to visit `(address_value, match_qubit)` pairs
            in order, or call `it.move_to(v)` to jump to any value `v`. At each step,
            `it.match_qubit` is a qubit that is ON in the parts of the superposition
            where `control` is active and `address == v`.

            Opening the `with` block allocates `max(1, len(address))` temporary qubits,
            and exiting the block uncomputes and frees them. Do not modify `address`,
            `control`, or `match_qubit` while the pass is open, and only use
            `match_qubit` as a control.

            For a non-empty `address`, positioning at the first in-range value currently
            costs at most `len(address)` Toffoli gates when `control` is a qubit, or
            `len(address) - 1` when `control` is `True` or a classical bit. Moving
            between distinct in-range values `a` and `b` costs at most
            `(a ^ b).bit_length() - 1` Toffoli gates, and closing the pass costs zero
            Toffoli gates.

            Args:
                address: The little-endian address to match against (at most 63 qubits).
                address_values: Defaults to `range(1 << len(address))`. The step-1 `range`
                    of values visited by `for` loops.
                control: Gates the iteration when provided. If `False`, no operations are
                    emitted and `for` loops visit zero values. Defaults to `True`.

            Returns:
                A reusable cursor over `address` that can be opened with `with`.

            Examples:
                >>> import kickmix as km
                >>> # Sequential iteration over a range of address values:
                >>> builder = km.CircuitBuilder()
                >>> address = builder.create_quantum_register(3, name='address')
                >>> target = builder.create_quantum_register(1, name='target')
                >>> with builder.unary_iteration(address, range(4)) as it:
                ...     for address_value, match_qubit in it:
                ...         if address_value == 1:
                ...             builder.cx(match_qubit, target[0])
                >>> circuit = builder.finish_circuit()
                >>> sim = km.Simulator(batch_size=8)
                >>> sim.use_same_registers_as(circuit)
                >>> for a in range(8):
                ...     sim.write_within_shot('address', a, a)
                >>> sim.do(circuit)
                >>> for a in range(8):
                ...     print(f"{a}: {sim.read_within_shot('target', a, out=int)}")
                0: 0
                1: 1
                2: 0
                3: 0
                4: 0
                5: 0
                6: 0
                7: 0

                >>> # Random-access jumps using move_to:
                >>> builder = km.CircuitBuilder()
                >>> address = builder.create_quantum_register(3, name='address')
                >>> target = builder.create_quantum_register(1, name='target')
                >>> with builder.unary_iteration(address) as it:
                ...     for address_value in [5, 2, 7]:
                ...         it.move_to(address_value)
                ...         builder.cx(it.match_qubit, target[0])
                >>> circuit = builder.finish_circuit()
                >>> sim = km.Simulator(batch_size=8)
                >>> sim.use_same_registers_as(circuit)
                >>> for a in range(8):
                ...     sim.write_within_shot('address', a, a)
                >>> sim.do(circuit)
                >>> for a in range(8):
                ...     print(f"{a}: {sim.read_within_shot('target', a, out=int)}")
                0: 0
                1: 0
                2: 1
                3: 0
                4: 0
                5: 1
                6: 0
                7: 1
        )DOC")
            .data());
}
