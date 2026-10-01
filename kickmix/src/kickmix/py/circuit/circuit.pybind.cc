#include "kickmix/py/circuit/circuit.pybind.h"

#include <cstdio>
#include <kickmix/id/register_id.h>
#include <kickmix/py/val/array.pybind.h>
#include <memory>
#include <pybind11/iostream.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "kickmix/py/util.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

static std::string resolve_file_path(const pybind11::handle &path) {
    pybind11::object fspath = pybind11::reinterpret_steal<pybind11::object>(PyOS_FSPath(path.ptr()));
    if (!fspath) {
        throw pybind11::error_already_set();
    }
    if (pybind11::isinstance<pybind11::str>(fspath) || pybind11::isinstance<pybind11::bytes>(fspath)) {
        return pybind11::cast<std::string>(fspath);
    }
    throw pybind11::type_error("Expected path to be a str or pathlib.Path.");
}

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

    c_circuit.def_static(
        "from_file",
        [](const pybind11::object &path, std::string_view format) -> Circuit {
            std::string path_str = resolve_file_path(path);
            if (format != "kmx" && format != "kmb") {
                throw std::invalid_argument(
                    "Unrecognized format '" + std::string(format) + "'. Expected 'kmx' or 'kmb'.");
            }
            std::unique_ptr<FILE, int (*)(FILE *)> file(fopen(path_str.c_str(), "rb"), &fclose);
            if (file == nullptr) {
                throw std::invalid_argument("Failed to open file for reading: '" + path_str + "'.");
            }
            if (format == "kmx") {
                return Circuit::from_kmx_file(file.get());
            }
            return Circuit::from_kmb_file(file.get());
        },
        pybind11::arg("path"),
        pybind11::arg("format") = "kmx",
        clean_doc_string(R"DOC(
            @signature def from_file(path: str | pathlib.Path, format: Literal['kmx', 'kmb'] = 'kmx') -> km.Circuit:
            Reads a `km.Circuit` from a file.

            Args:
                path: The path to the file to read from.
                format: The file format to parse. Defaults to `'kmx'`.
                    `'kmx'`: Human-readable text kickmix format.
                    `'kmb'`: Binary kickmix format.

            Returns:
                The parsed `km.Circuit`.

            Examples:
                >>> import pathlib
                >>> import tempfile
                >>> import kickmix as km
                >>> circuit = km.Circuit('''
                ...     CCX q0 q1 q2
                ...     CX q0 q1
                ...     X q0
                ... ''')
                >>> with tempfile.TemporaryDirectory() as d:
                ...     path = pathlib.Path(d) / 'circuit.kmx'
                ...     circuit.to_file(path)
                ...     loaded = km.Circuit.from_file(path)
                >>> loaded == circuit
                True
        )DOC")
            .data());

    c_circuit.def(
        "to_file",
        [](const Circuit &self, const pybind11::object &path, std::string_view format) {
            std::string path_str = resolve_file_path(path);
            if (format != "kmx" && format != "kmb") {
                throw std::invalid_argument(
                    "Unrecognized format '" + std::string(format) + "'. Expected 'kmx' or 'kmb'.");
            }
            std::unique_ptr<FILE, int (*)(FILE *)> file(fopen(path_str.c_str(), "wb"), &fclose);
            if (file == nullptr) {
                throw std::invalid_argument("Failed to open file for writing: '" + path_str + "'.");
            }
            if (format == "kmx") {
                self.write_kmx_to(file.get());
            } else {
                self.write_kmb_to(file.get());
            }
        },
        pybind11::arg("path"),
        pybind11::arg("format") = "kmx",
        clean_doc_string(R"DOC(
            @signature def to_file(self, path: str | pathlib.Path, format: Literal['kmx', 'kmb'] = 'kmx') -> None:
            Writes the circuit to a file.

            Args:
                path: The path to the file to write to.
                format: The file format to write. Defaults to `'kmx'`.
                    `'kmx'`: Human-readable text kickmix format.
                    `'kmb'`: Binary kickmix format.

            Examples:
                >>> import pathlib
                >>> import tempfile
                >>> import kickmix as km
                >>> circuit = km.Circuit('''
                ...     CCX q0 q1 q2
                ...     CX q0 q1
                ...     X q0
                ... ''')
                >>> with tempfile.TemporaryDirectory() as d:
                ...     path = pathlib.Path(d) / 'circuit.kmb'
                ...     circuit.to_file(path, format='kmb')
                ...     loaded = km.Circuit.from_file(path, format='kmb')
                >>> loaded == circuit
                True
        )DOC")
            .data());

    c_circuit.def(pybind11::self == pybind11::self, "Determines if two circuits have identical instructions.");
    c_circuit.def(pybind11::self != pybind11::self, "Determines if two circuits have different instructions.");
}
