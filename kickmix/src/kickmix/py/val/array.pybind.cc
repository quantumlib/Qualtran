#include "kickmix/py/val/array.pybind.h"

#include <pybind11/iostream.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "kickmix/py/util.pybind.h"
#include "kickmix/py/val/id.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

bool PyArrayXZ::operator==(const PyArrayXZ &other) const {
    return view.has_same_contents_as(other.view);
}

PyArrayXZ PyArrayXZ::copy_of(const kickmix::stride_span_xz &other) {
    PyArrayXZ result;
    result.owner = std::make_shared<array_xz>(std::move(array_xz::copy_of(other)));
    result.view = *result.owner;
    return result;
}

std::string QXZArray_str(const PyArrayXZ &obj) {
    std::stringstream ss;
    ss << "km.array([";
    bool first = true;
    for (const auto &e : obj.view) {
        if (first) {
            first = false;
        } else {
            ss << ", ";
        }
        if (e.is_qubit()) {
            ss << "q" << e.untagged_id();
        } else if (e.is_bit()) {
            ss << "b" << e.untagged_id();
        } else if (e.is_xbit()) {
            ss << "xb" << e.untagged_id();
        } else if (e.is_xbool()) {
            ss << (e.is_minus_ket() ? "|->" : "|+>");
        } else if (e.is_bool()) {
            ss << ((bool)e ? "True" : "False");
        } else {
            ss << "{.tagged_id=" << e.tagged_id << "}";
        }
    }
    ss << "])";
    return ss.str();
}

std::string QXZArray_repr(const PyArrayXZ &obj) {
    std::stringstream ss;
    bool first = true;
    ss << "km.array([";
    for (const auto &e : obj.view) {
        if (first) {
            first = false;
        } else {
            ss << ", ";
        }
        if (e.is_qubit()) {
            ss << "q(" << e.untagged_id() << ")";
        } else if (e.is_bit()) {
            ss << "b(" << e.untagged_id() << ")";
        } else if (e.is_xbit()) {
            ss << "xb(" << e.untagged_id() << ")";
        } else if (e.is_xbool()) {
            ss << (e.is_minus_ket() ? "xbool(True)" : "xbool(False)");
        } else if (e.is_bool()) {
            ss << ((bool)e ? "True" : "False");
        } else {
            ss << "{.tagged_id=" << e.tagged_id << "}";
        }
    }
    ss << "])";
    return ss.str();
}

PyArrayXZ qc_array_from_obj(const pybind11::object &obj) {
    size_t n = pybind11::len(obj);
    PyArrayXZ result;
    result.owner = std::make_shared<array_xz>(array_xz::alloc_noinit(n));

    QubitOrXZBitOrXZBool *out = result.owner->items;
    for (const pybind11::handle &e : obj) {
        *out++ = obj_to_qubit_or_xzbit_or_xzbool(e, "item");
    }
    result.owner->recompute_common_type();
    result.view = *result.owner;
    return result;
}

static pybind11::object id_to_obj(QubitOrXZBitOrXZBool val) {
    switch ((QXZTypeTag8)(val.tagged_id >> 29)) {
        case QXZTypeTag8::BOOL_VAL:
            return pybind11::cast((bool)val);
        case QXZTypeTag8::BIT_ID:
            return pybind11::cast((BitId)val);
        case QXZTypeTag8::QUBIT_ID:
            return pybind11::cast((QubitId)val);
        case QXZTypeTag8::XBIT_ID:
            return pybind11::cast((XBitId)val);
        case QXZTypeTag8::XBOOL_VAL:
            return pybind11::cast((XBool)val);
        default: {
            std::stringstream ss;
            ss << "km.array::operator[] - unhandled km.array entry type: ";
            ss << val;
            throw pybind11::type_error(ss.str());
        }
    }
}

pybind11::object qc_array_obj_at(const stride_span_xz &array, pybind11::ssize_t index) {
    if (-(int64_t)array.count <= index && index < (int64_t)array.count) {
        if (index < 0) {
            index += array.count;
        }
        return id_to_obj(array[index]);
    } else {
        throw pybind11::index_error("index out of range");
    }
}

static PyArrayXZ qcarray_slice(const PyArrayXZ &array, const pybind11::slice &slice) {
    pybind11::ssize_t slice_start;
    pybind11::ssize_t slice_stride;
    pybind11::ssize_t slice_stop;
    pybind11::ssize_t slice_len;
    if (!slice.compute(array.view.size(), &slice_start, &slice_stop, &slice_stride, &slice_len)) {
        throw pybind11::error_already_set();
    }

    return PyArrayXZ{
        .view = stride_span_xz(
            array.view.ptr + array.view.stride * slice_start,
            array.view.stride * slice_stride,
            (size_t)slice_len,
            array.view.common_type),
        .owner = array.owner,
    };
}

static pybind11::object qxz_array_getitem(const PyArrayXZ &self, const pybind11::object &index) {
    if (pybind11::isinstance<pybind11::int_>(index)) {
        return qc_array_obj_at(self.view, pybind11::cast<pybind11::ssize_t>(index));
    } else if (pybind11::isinstance<pybind11::slice>(index)) {
        return pybind11::cast(qcarray_slice(self, pybind11::cast<pybind11::slice>(index)));
    } else {
        std::stringstream ss;
        ss << "Not an index: ";
        ss << pybind11::repr(index);
        throw pybind11::index_error(ss.str());
    }
}

void kickmix_py::register_qcarray_methods(pybind11::class_<PyArrayXZ> &c_qcarray) {
    c_qcarray.def(
        pybind11::init(&qc_array_from_obj),
        pybind11::arg("arg") = pybind11::make_tuple(),
        pybind11::pos_only(),
        clean_doc_string(R"DOC(
        )DOC")
            .data());

    c_qcarray.def(
        "__getitem__",
        &qxz_array_getitem,
        pybind11::arg("index"),
        clean_doc_string(R"DOC(
            Returns an item or slice from the array.

            Arg:
                index: An int or a slice.

            Returns:
                 If the index is an int: the item at that index.
                 If the index is a slice: a view of the array.
        )DOC")
            .data());

    c_qcarray.def(
        "__len__",
        [](const PyArrayXZ &self) -> size_t {
            return self.view.count;
        },
        clean_doc_string(R"DOC(
            Returns the length of the array.
        )DOC")
            .data());

    c_qcarray.def(
        "__add__",
        [](const PyArrayXZ &self, const PyArrayXZ &other) -> PyArrayXZ {
            PyArrayXZ result;
            result.owner = std::make_shared<array_xz>(array_xz::copy_of_concat(self.view, other.view));
            result.view = *result.owner;
            return result;
        },
        clean_doc_string(R"DOC(
            Returns the concatenation of two arrays.
        )DOC")
            .data());

    c_qcarray.def(
        "__str__",
        &QXZArray_str,
        clean_doc_string(R"DOC(
            Returns a text representation of the array.
        )DOC")
            .data());

    c_qcarray.def(
        "__repr__",
        &QXZArray_repr,
        clean_doc_string(R"DOC(
            Returns a parseable text representation of the array.
        )DOC")
            .data());

    c_qcarray.def(
        "_UNSTABLE_internal_values",
        [](const PyArrayXZ &self) -> pybind11::object {
            std::map<std::string, int64_t> result;
            result["offset"] = (intptr_t)(self.view.ptr - self.owner->items);
            result["len"] = (int64_t)self.view.count;
            result["stride"] = (int64_t)self.view.stride;
            result["common_type"] = (int64_t)self.view.common_type;
            return pybind11::cast(result);
        },
        "A private helper method used for unit testing. Do not use.");

    c_qcarray.def(pybind11::self == pybind11::self, "Determines if two km.arrays have identical contents.");
    c_qcarray.def(pybind11::self != pybind11::self, "Determines if two km.arrays have different contents.");
}
