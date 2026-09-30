#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kickmix/id/qubit_or_true.h"
#include "kickmix/py/circuit/circuit.pybind.h"
#include "kickmix/py/circuit/circuit_builder.pybind.h"
#include "kickmix/py/util.pybind.h"
#include "kickmix/py/val/id.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

array_z kickmix_py::read_python_int_list_into_table_data(const pybind11::object &table, size_t word_length) {
    pybind11::int_ num_bits_obj = pybind11::cast(word_length);
    pybind11::int_ num_bytes_obj = pybind11::cast((word_length + 7) / 8);
    size_t num_entries = pybind11::len(table);

    array_z result = array_z::alloc_noinit(num_entries * word_length);
    QubitOrBitOrBool *out = result.items;
    for (size_t k = 0; k < num_entries; k++) {
        pybind11::int_ val = table[pybind11::int_(k)];
        pybind11::int_ val_bit_len = pybind11::cast<size_t>(val.attr("bit_length")());
        if (val < pybind11::int_(0) || val_bit_len > num_bits_obj) {
            std::stringstream ss;
            ss << "table[" << k << "]=";
            ss << pybind11::repr(val);
            ss << " isn't in range(0, 2**len(target)) (len(target)=" << num_bits_obj << ")";
            throw std::invalid_argument(ss.str());
        }
        pybind11::bytes bytes_obj = val.attr("to_bytes")(num_bytes_obj, pybind11::cast("little"));
        std::string_view x = bytes_obj;
        for (size_t k2 = 0; k2 < word_length; k2++) {
            *out++ = (bool)(((uint8_t)x[k2 / 8] >> (k2 & 7)) & 1);
        }
    }

    result.recompute_common_type();
    return result;
}
