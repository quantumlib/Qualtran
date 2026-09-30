#include "kickmix/py/circuit/circuit.pybind.h"

#include <kickmix/id/register_id.h>
#include <kickmix/py/val/array.pybind.h>
#include <pybind11/iostream.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

using namespace kickmix;

void kickmix::register_circuit_methods(pybind11::class_<Circuit> &c_circuit) {
    c_circuit.def(
        "__str__",
        [](const Circuit &self) -> std::string {
            return self.str();
        },
        "Returns a description of the circuit.");

    c_circuit.def_property_readonly(
        "register_data",
        [](const Circuit &self) -> pybind11::object {
            pybind11::dict result;
            for (size_t k = 0; k < self.register_data.size(); ++k) {
                const auto &e = self.register_data[k];
                pybind11::object key;
                if (e.name.empty()) {
                    key = pybind11::cast(RegisterId((uint32_t)k));
                } else {
                    key = pybind11::cast(e.name);
                }
                result[key] = kickmix_py::PyArrayXZ::copy_of(stride_span<const QubitOrBit>(e.contents));
            }
            return result;
        },
        "Returns the names and contents of registers in the circuit.");

    c_circuit.def(
        "__repr__",
        [](const Circuit &self) -> std::string {
            std::stringstream ss;
            ss << "km.Circuit('''";
            if (self.num_ops > 0) {
                ss << "\n    ";
            }
            for (char c : self.str()) {
                ss << c;
                if (c == '\n') {
                    ss << "    ";
                }
            }
            ss << "\n''')";
            return ss.str();
        },
        "Returns a description of the circuit.");

    c_circuit.def(
        "text_diagram",
        [](const Circuit &self) -> std::string {
            return self.text_diagram();
        },
        "Returns a text diagram of the circuit.");

    c_circuit.def(
        "html_diagram",
        [](const Circuit &self) -> std::string {
            return self.html_diagram();
        },
        "Returns an html diagram of the circuit.");

    c_circuit.def(
        "max_magic",
        [](const Circuit &self) -> size_t {
            return self.max_magic();
        },
        "Returns an upper bound on the number of CCX/CCZ gates run by the circuit.");

    c_circuit.def(
        "reaction_depth",
        [](const Circuit &self) -> size_t {
            return self.reaction_depth();
        },
        "Returns the reaction depth of the circuit, assuming it is powered by CCZ states.");

    c_circuit.def_property_readonly(
        "num_qubits",
        [](const Circuit &self) -> size_t {
            return self.num_qubits;
        },
        "Returns the number of qubits used by the circuit.");

    c_circuit.def_property_readonly(
        "num_bits",
        [](const Circuit &self) -> size_t {
            return self.num_bits;
        },
        "Returns the number of bits used by the circuit.");

    c_circuit.def_property_readonly(
        "num_registers",
        [](const Circuit &self) -> size_t {
            return self.register_data.size();
        },
        "Returns the number of registers used by the circuit.");

    c_circuit.def(
        pybind11::init([](std::string_view text) -> Circuit {
            return Circuit(text);
        }),
        pybind11::arg("circuit_text") = "",
        "Creates a kickmix Circuit by parsing the given text.");

    c_circuit.def(
        "__len__",
        [](const Circuit &self) -> size_t {
            return self.num_ops;
        },
        "Returns the number of instructions in the circuit.");

    c_circuit.def(pybind11::self == pybind11::self, "Determines if two circuits have identical instructions.");
    c_circuit.def(pybind11::self != pybind11::self, "Determines if two circuits have different instructions.");
}
