#include "kickmix/py/circuit/circuit.pybind.h"

#include <kickmix/id/register_id.h>
#include <kickmix/py/val/array.pybind.h>
#include <pybind11/iostream.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "kickmix/py/util.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

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
        "max_t",
        [](const Circuit &self) -> size_t {
            return self.max_t();
        },
        clean_doc_string(R"DOC(
            Returns an upper bound on the number of T gates run by the circuit.

            T gates are `Z_POW` operations whose angle is an odd multiple of
            0.25 half turns (45 degrees).

            Examples:
                >>> import kickmix as km
                >>> c = km.Circuit('''
                ...     Z_POW q0 0.5
                ...     Z_POW q0 0.25
                ...     Z_POW q0 -0.25 if b0
                ...     Z_POW q0 0.125
                ... ''')
                >>> c.max_t()
                2
        )DOC")
            .data());

    c_circuit.def(
        "max_rotations",
        [](const Circuit &self) -> size_t {
            return self.max_rotations();
        },
        clean_doc_string(R"DOC(
            Returns an upper bound on the number of arbitrary-angle rotations.

            Rotations are `Z_POW` operations whose angle is not a multiple of
            0.25 half turns (i.e. neither Clifford nor T).

            Examples:
                >>> import kickmix as km
                >>> c = km.Circuit('''
                ...     Z_POW q0 0.5
                ...     Z_POW q0 0.25
                ...     Z_POW q0 0.125
                ...     Z_POW q0 0.1 if b0
                ... ''')
                >>> c.max_rotations()
                2
        )DOC")
            .data());

    c_circuit.def(
        "max_op_counts",
        [](const Circuit &self, const pybind11::object &z_pow_key) -> pybind11::dict {
            std::array<uint64_t, 256> raw_counts{};
            for (size_t k = 0; k < self.num_ops; k++) {
                raw_counts[static_cast<uint8_t>(self.op_types[k])]++;
            }
            bool custom_z_pow = !z_pow_key.is_none();
            pybind11::dict result;
            for (const auto &[name, ops] : OP_COUNT_GROUPS) {
                if (custom_z_pow && name == "Z_POW") {
                    continue;
                }
                uint64_t total = 0;
                for (OpType op : ops) {
                    total += raw_counts[static_cast<uint8_t>(op)];
                }
                result[pybind11::str(name.data(), name.size())] = total;
            }
            if (custom_z_pow) {
                std::map<FixedPrecisionAngle128, uint64_t> angle_counts;
                for (size_t k = 0; k < self.num_angle_ops; k++) {
                    angle_counts[self.angles[k]]++;
                }
                populate_z_pow_counts(result, z_pow_key, angle_counts);
            }
            return result;
        },
        pybind11::arg("z_pow_key") = pybind11::none(),
        clean_doc_string(R"DOC(
            @signature def max_op_counts(self, z_pow_key: Callable[[fractions.Fraction], str] | None = None) -> dict[str, int]:
            Returns an upper bound on the execution count of each operation type.

            Counts how many times each operation type appears in the circuit,
            including conditional operations (`OP_IF` and operations inside
            `PUSH_CONDITION` blocks). Unconditional and conditional variants of
            an operation (e.g. `CCX` and `CCX_IF`) are summed under the base
            operation name (`'CCX'`), matching C++ `count_operations` and
            `Simulator.op_counts()`.

            Args:
                z_pow_key: Optional callable taking a `Z_POW` angle (in half
                    turns in `[0, 2)` as a `fractions.Fraction`) and returning
                    a dictionary key `str` under which to accumulate its count.
                    If `None` (the default), all `Z_POW` operations are summed
                    under `'Z_POW'`.

            Returns:
                A dictionary mapping each operation name (`'NEG'`,
                `'BIT_INVERT'`, `'BIT_STORE0'`, `'BIT_STORE1'`, `'X'`, `'Z'`,
                `'R'`, `'HMR'`, `'CX'`, `'CZ'`, `'SWAP'`, `'CCX'`, `'CCZ'`,
                `'Z_POW'`, `'DEBUG_PRINT'`, `'POP_CONDITION'`,
                `'PUSH_CONDITION'`, or custom `z_pow_key` names) to its count.

            Examples:
                >>> import kickmix as km
                >>> c = km.Circuit('''
                ...     X q0
                ...     CX q0 q1 if b0
                ...     Z_POW q0 0.5
                ...     Z_POW q0 0.25
                ...     Z_POW q0 -0.25 if b0
                ...     Z_POW q0 0.125
                ... ''')
                >>> c.max_op_counts()['X']
                1
                >>> c.max_op_counts()['Z_POW']
                4
                >>> def classify_z_pow(angle):
                ...     if angle.denominator <= 2:
                ...         return 'Z_POW_CLIFFORD'
                ...     if angle.denominator == 4:
                ...         return 'T'
                ...     return f'Z_POW_2^-{angle.denominator.bit_length() - 1}'
                >>> grouped = c.max_op_counts(z_pow_key=classify_z_pow)
                >>> grouped['Z_POW_CLIFFORD'], grouped['T'], grouped['Z_POW_2^-3']
                (1, 2, 1)
        )DOC")
            .data());

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
