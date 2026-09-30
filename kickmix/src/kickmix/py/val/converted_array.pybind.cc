#include "kickmix/py/val/converted_array.pybind.h"

#include <pybind11/stl.h>

#include "kickmix/py/types.pybind.h"
#include "kickmix/py/val/array.pybind.h"
#include "kickmix/py/val/id.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

ConvertedArrayXZ::~ConvertedArrayXZ() {
    if (free_on_destruct != nullptr) {
        free(free_on_destruct);
        free_on_destruct = nullptr;
    }
}

static const QubitOrXZBitOrXZBool GLOBAL_TRUE = true;
static const QubitOrXZBitOrXZBool GLOBAL_FALSE = false;
static const QubitOrXZBitOrXZBool GLOBAL_PLUS = XBool(false);
static const QubitOrXZBitOrXZBool GLOBAL_MINUS = XBool(true);

template <typename T>
static T *aligned_alloc_32(size_t count) {
    if (count == 0) {
        return nullptr;
    }
    size_t bytes = count * sizeof(T);
    bytes += size_t{31};
    bytes &= ~size_t{31};
    return (T *)std::aligned_alloc(32, bytes);
}

ConvertedArrayXZ ConvertedArrayXZ::from_obj_expecting_list(
    const pybind11::handle &obj, const char *context_value_name) {
    PyObject *obj_ptr = obj.ptr();
    void *type_ptr = Py_TYPE(obj_ptr);

    if (type_ptr == PYBIND_TYPE_PTR_QCARRAY) {
        return ConvertedArrayXZ(unsafe_pyobject_ptr_to_qcarray_ptr(obj_ptr)->view, nullptr, false);
    }

    // Treat list-like objects as km.array.
    if (pybind11::hasattr(obj, "__len__")) {
        size_t n = pybind11::len(obj);
        QubitOrXZBitOrXZBool *out = aligned_alloc_32<QubitOrXZBitOrXZBool>(n);
        ConvertedArrayXZ result(stride_span_xz(out, 1, n, QXZTypeTag8::MAY_BE_MIXED), out, false);
        for (const auto &e : obj) {
            *out++ = obj_to_qubit_or_xzbit_or_xzbool(e, "item");
        }
        result.span.common_type = result.span.compute_common_type();
        return result;
    }

    std::stringstream ss;
    ss << "Expected a km.array | Sequence[q | b | bool] but got ";
    ss << context_value_name;
    ss << "=" << pybind11::repr(obj);
    throw pybind11::type_error(ss.str());
}

ConvertedArrayXZ ConvertedArrayXZ::from_obj(const pybind11::handle &obj, const char *context_value_name) {
    PyObject *obj_ptr = obj.ptr();
    void *type_ptr = Py_TYPE(obj_ptr);

    if (type_ptr == PYBIND_TYPE_PTR_QCARRAY) {
        return ConvertedArrayXZ(unsafe_pyobject_ptr_to_qcarray_ptr(obj_ptr)->view, nullptr, false);
    }
    if (type_ptr == PYBIND_TYPE_PTR_QUBIT_ID) {
        return ConvertedArrayXZ(
            stride_span_xz(
                (QubitOrXZBitOrXZBool *)unsafe_pyobject_ptr_to_qubit_id_ptr(obj_ptr), 0, 1, QXZTypeTag8::QUBIT_ID),
            nullptr,
            true);
    }
    if (type_ptr == PYBIND_TYPE_PTR_BIT_ID) {
        return ConvertedArrayXZ(
            stride_span_xz(
                (QubitOrXZBitOrXZBool *)unsafe_pyobject_ptr_to_bit_id_ptr(obj_ptr), 0, 1, QXZTypeTag8::BIT_ID),
            nullptr,
            true);
    }
    if (type_ptr == &PyBool_Type) {
        return ConvertedArrayXZ(
            stride_span_xz(obj_ptr == Py_True ? &GLOBAL_TRUE : &GLOBAL_FALSE, 0, 1, QXZTypeTag8::BOOL_VAL),
            nullptr,
            true);
    }
    if (type_ptr == PYBIND_TYPE_PTR_XBIT_ID) {
        return ConvertedArrayXZ(
            stride_span_xz(
                (QubitOrXZBitOrXZBool *)unsafe_pyobject_ptr_to_xbit_id_ptr(obj_ptr), 0, 1, QXZTypeTag8::XBIT_ID),
            nullptr,
            true);
    }
    if (type_ptr == PYBIND_TYPE_PTR_XBOOL) {
        return ConvertedArrayXZ(
            stride_span_xz(
                (QubitOrXZBitOrXZBool *)unsafe_pyobject_ptr_to_xbool_ptr(obj_ptr), 0, 1, QXZTypeTag8::XBOOL_VAL),
            nullptr,
            true);
    }

    // Treat 0 and 1 as booleans.
    if (type_ptr == &PyLong_Type) {
        long val = PyLong_AsLong(obj.ptr());
        if (val == 0 || val == 1) {
            return ConvertedArrayXZ(
                stride_span_xz(val == 1 ? &GLOBAL_TRUE : &GLOBAL_FALSE, 0, 1, QXZTypeTag8::BOOL_VAL), nullptr, true);
        }
        if (val == -1 && PyErr_Occurred()) {
            PyErr_Clear();
        }
    }
    // Treat certain strings as special objects.
    if (type_ptr == &PyUnicode_Type) {
        auto value = pybind11::cast<std::string_view>(obj);
        if (value == "|+>") {
            return ConvertedArrayXZ(stride_span_xz(&GLOBAL_PLUS, 0, 1, QXZTypeTag8::XBOOL_VAL), nullptr, true);
        }
        if (value == "|->") {
            return ConvertedArrayXZ(stride_span_xz(&GLOBAL_MINUS, 0, 1, QXZTypeTag8::XBOOL_VAL), nullptr, true);
        }
    }

    // Treat list-like objects as kmarrays.
    if (pybind11::hasattr(obj, "__len__")) {
        size_t n = pybind11::len(obj);
        QubitOrXZBitOrXZBool *out = aligned_alloc_32<QubitOrXZBitOrXZBool>(n);
        ConvertedArrayXZ result(stride_span_xz(out, 1, n, QXZTypeTag8::MAY_BE_MIXED), out, false);
        for (const auto &e : obj) {
            *out++ = obj_to_qubit_or_xzbit_or_xzbool(e, "item");
        }
        result.span.common_type = result.span.compute_common_type();
        return result;
    }

    std::stringstream ss;
    ss << "Expected a q | b | bool | km.array | Sequence[q | b | bool] but got ";
    ss << context_value_name;
    ss << "=" << pybind11::repr(obj);
    throw pybind11::type_error(ss.str());
}

ConvertedArrayXZ ConvertedArrayXZ::from_obj_or_int(
    const pybind11::handle &obj, size_t int_num_bits, const char *context_value_name, bool allow_twos_complement) {
    if (pybind11::isinstance<pybind11::int_>(obj) && !pybind11::isinstance<pybind11::bool_>(obj)) {
        pybind11::int_ val = pybind11::cast<pybind11::int_>(obj);
        if (allow_twos_complement && val < pybind11::int_(0)) {
            pybind11::int_ modulus = pybind11::int_(1).attr("__lshift__")(int_num_bits);
            val = val.attr("__add__")(modulus);
        }
        size_t bit_len = pybind11::cast<size_t>(val.attr("bit_length")());
        if (val < pybind11::int_(0) || bit_len > int_num_bits) {
            std::stringstream ss;
            ss << context_value_name << " int must be in range(";
            ss << (allow_twos_complement ? "-(1 << n)" : "0");
            ss << ", 1 << n)";
            throw std::invalid_argument(ss.str());
        }
        QubitOrXZBitOrXZBool *out = aligned_alloc_32<QubitOrXZBitOrXZBool>(int_num_bits);
        if (int_num_bits <= 64) {
            uint64_t u = pybind11::cast<uint64_t>(val);
            for (size_t k = 0; k < int_num_bits; k++) {
                out[k] = QubitOrXZBitOrXZBool((bool)((u >> k) & 1));
            }
        } else {
            size_t num_bytes = (int_num_bits + 7) / 8;
            pybind11::bytes bytes_obj = val.attr("to_bytes")(num_bytes, "little");
            std::string_view bytes_view = bytes_obj;
            for (size_t k = 0; k < int_num_bits; k++) {
                bool bit = ((uint8_t)bytes_view[k / 8] >> (k % 8)) & 1;
                out[k] = QubitOrXZBitOrXZBool(bit);
            }
        }
        return ConvertedArrayXZ(stride_span_xz(out, 1, int_num_bits, QXZTypeTag8::BOOL_VAL), out, false);
    }
    return ConvertedArrayXZ::from_obj(obj, context_value_name);
}
