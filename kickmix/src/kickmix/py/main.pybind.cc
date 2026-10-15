#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kickmix/py/circuit/circuit.pybind.h"
#include "kickmix/py/circuit/circuit_builder.pybind.h"
#include "kickmix/py/circuit/subs/gf_arithmetic.pybind.h"
#include "kickmix/py/circuit/subs/unary_iteration.pybind.h"
#include "kickmix/py/interactive_simulator.pybind.h"
#include "kickmix/py/types.pybind.h"
#include "kickmix/py/util.pybind.h"
#include "kickmix/py/val/array.pybind.h"
#include "kickmix/py/val/id.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

PyObject *kickmix_py::PYBIND_TYPE_PTR_QUBIT_ID;
PyObject *kickmix_py::PYBIND_TYPE_PTR_BIT_ID;
PyObject *kickmix_py::PYBIND_TYPE_PTR_XBIT_ID;
PyObject *kickmix_py::PYBIND_TYPE_PTR_XBOOL;
PyObject *kickmix_py::PYBIND_TYPE_PTR_QCARRAY;
PyObject *kickmix_py::PYBIND_TYPE_PTR_REGISTER_ID;

PYBIND11_MODULE(kickmix, m) {
    m.attr("__version__") = "0.1";
    m.doc() = clean_doc_string(R"DOC(
        Kickgen: generate kickmix circuits.
    )DOC")
                  .data();

    auto c_circuit = pybind11::class_<Circuit>(
        m,
        "Circuit",
        clean_doc_string(R"DOC(
            An immutable kickmix circuit.

            Examples:
                >>> import kickmix as km
                >>> circuit = km.Circuit('''
                ...     X q0
                ...     CX q0 q9
                ... ''')

                >>> simulator = km.Simulator(batch_size=1)
                >>> print(simulator.read_within_shot(km.q(9), shot_index=0))
                False
                >>> simulator.do(circuit)
                >>> print(simulator.read_within_shot(km.q(9), shot_index=0))
                True
         )DOC")
            .data());

    auto c_circuit_builder = pybind11::class_<PyCircuitBuilder>(
        m,
        "CircuitBuilder",
        clean_doc_string(R"DOC(
            A builder class that helps create a kickmix circuit.

            Examples:
                >>> import kickmix as km
                >>> builder = km.CircuitBuilder()

                >>> target = builder.create_quantum_register(7, "target")
                >>> shift = builder.create_quantum_register(3, "shift")
                >>> builder.left_rotate(target, shift)
                >>> print(builder.finish_circuit().text_diagram())
                q0: -target[0]-@-X-@---------@-X-@-----------@---X---@-----@-X-@-----------
                               | | |         | | |           |   |   |     | | |
                q1: -target[1]-X-@-X---------|-|-|-----------|-@-|-X-|-@---X-@-X-----------
                                 |           | | |           | | | | | |     |
                q2: -target[2]---|-@---X---@-X-@-X-----------|-|-|-|-|-|-----|-@---X---@---
                                 | |   |   |   |             | | | | | |     | |   |   |
                q3: -target[3]---|-|-@-|-X-|-@-|-@---X---@---|-X-|-@-|-X-----|-|-@-|-X-|-@-
                                 | | | | | | | | |   |   |   |   | | |       | | | | | | |
                q4: -target[4]---|-|-|-|-|-|-|-|-|-@-|-X-|-@-X---@-|-X-------|-|-|-|-|-|-|-
                                 | | | | | | | | | | | | | |     | |         | | | | | | |
                q5: -target[5]---|-|-X-|-@-|-X-|-|-X-|-@-|-X-----|-|---@-X-@-|-|-X-|-@-|-X-
                                 | |   | | |   | |   | | |       | |   | | | | |   | | |
                q6: -target[6]---|-X---@-|-X---|-X---@-|-X-------|-|---X-@-X-|-X---@-|-X---
                                 |     | |     |     | |         | |     |   |     | |
                q7: -shift[0]----@-----@-@---@-|-----|-|---@-----|-|-----|---|-----|-|-----
                                             | |     | |   |     | |     |   |     | |
                q8: -shift[1]----------------X-@-----@-@---X-@---|-|-----|-@-|-----|-|-----
                                                             |   | |     | | |     | |
                q9: -shift[2]--------------------------------X---@-@-----@-X-@-----@-@-----
         )DOC")
            .data());

    auto c_qcarray = pybind11::class_<PyArrayXZ>(
        m,
        "array",
        clean_doc_string(R"DOC(
            An immutable array of qubit ids, bit ids, and/or boolean values.

            Circuit creation methods operate substantially faster when given a
            km.array,  rather than a normal python list of values, because a
            km.array's values are stored contiguously in memory alongside compressed
            type information.

            Examples:
                >>> import kickmix as km
                >>> arr = km.array([km.q(0), km.b(1), True])
                >>> arr
                km.array([km.q(0), km.b(1), True])
                >>> len(arr)
                3
         )DOC")
            .data());

    auto c_qid = pybind11::class_<kickmix::QubitId>(
        m,
        "q",
        clean_doc_string(R"DOC(
             A kickmix qubit index.
         )DOC")
            .data());

    auto c_bid = pybind11::class_<kickmix::BitId>(
        m,
        "b",
        clean_doc_string(R"DOC(
             A kickmix bit index.
         )DOC")
            .data());

    auto c_rid = pybind11::class_<kickmix::RegisterId>(
        m,
        "r",
        clean_doc_string(R"DOC(
             A kickmix register index.
         )DOC")
            .data());

    auto c_xbid = pybind11::class_<kickmix::XBitId>(
        m,
        "xb",
        clean_doc_string(R"DOC(
             A kickmix bit index being interpreted as an X basis value.

             X basis values act like controls when given as the targets
             of bit flip operations. For example, cx(BitId(0), XBitId(1)) is equivalent
             to cz(BitId(0), BitId(1)).

             Examples:
                 >>> import kickmix as km
                 >>> x = km.xb(5)
                 >>> x.id
                 5
         )DOC")
            .data());

    auto c_xbool = pybind11::class_<kickmix::XBool>(
        m,
        "xbool",
        clean_doc_string(R"DOC(
             A boolean that represents an X basis value.

             If the boolean is True, the X basis value is |->.
             If the boolean is False, the X basis value is |+>.

             X basis values act like controls when given as the targets
             of bit flip operations. For example, cx(a, xbool(b)) is equivalent
             to cz(a, b).

             Examples:
                 >>> import kickmix as km
                 >>> km.xbool(True)
                 km.xbool(True)
                 >>> bool(km.xbool(False))
                 False
                 >>> bool(km.xbool(True))
                 True
         )DOC")
            .data());

    auto c_gf2_field = register_gf2_field_class(m);

    auto c_unary_iteration = register_unary_iteration_class(m);

    auto c_interactive_simulator = register_interactive_simulator_class(m);

    PYBIND_TYPE_PTR_QUBIT_ID = pybind11::type::of<kickmix::QubitId>().ptr();
    PYBIND_TYPE_PTR_BIT_ID = pybind11::type::of<kickmix::BitId>().ptr();
    PYBIND_TYPE_PTR_QCARRAY = pybind11::type::of<PyArrayXZ>().ptr();
    PYBIND_TYPE_PTR_XBIT_ID = pybind11::type::of<kickmix::XBitId>().ptr();
    PYBIND_TYPE_PTR_XBOOL = pybind11::type::of<kickmix::XBool>().ptr();
    PYBIND_TYPE_PTR_REGISTER_ID = pybind11::type::of<kickmix::RegisterId>().ptr();

    register_circuit_methods(c_circuit);
    register_circuit_builder_mark_class(m);
    register_circuit_builder_methods(c_circuit_builder);
    register_qcarray_methods(c_qcarray);
    register_xbool_methods(c_xbool);
    register_qid_methods(c_qid);
    register_bid_methods(c_bid);
    register_rid_methods(c_rid);
    register_xbid_methods(c_xbid);
    register_gf2_field_methods(c_gf2_field);
    register_unary_iteration_methods(c_unary_iteration);
    register_interactive_simulator_methods(c_interactive_simulator);
}
