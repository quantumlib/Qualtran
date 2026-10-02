#include "kickmix/py/val/id.pybind.h"

#include <pybind11/iostream.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "array.pybind.h"
#include "kickmix/py/types.pybind.h"
#include "kickmix/py/util.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

bool is_lt(QubitOrXZBitOrXZBool self, const pybind11::object &rhs) {
    try {
        return self < obj_to_qubit_or_xzbit_or_xzbool(rhs, "rhs");
    } catch (const pybind11::type_error &ex) {
        std::stringstream ss;
        ss << "Don't know how to compare to ";
        ss << pybind11::repr(rhs);
        throw pybind11::type_error(ss.str());
    }
}

QubitOrXBitOrXBool kickmix_py::obj_to_qubit_or_xbit_or_xbool(
    const pybind11::handle &obj, const char *context_value_name) {
    PyObject *obj_ptr = obj.ptr();
    void *type_ptr = Py_TYPE(obj_ptr);
    if (type_ptr == PYBIND_TYPE_PTR_QUBIT_ID) {
        return *unsafe_pyobject_ptr_to_qubit_id_ptr(obj_ptr);
    }
    if (type_ptr == PYBIND_TYPE_PTR_XBIT_ID) {
        return *unsafe_pyobject_ptr_to_xbit_id_ptr(obj_ptr);
    }
    if (type_ptr == PYBIND_TYPE_PTR_XBOOL) {
        return *unsafe_pyobject_ptr_to_xbool_ptr(obj_ptr);
    }
    if (type_ptr == &PyUnicode_Type) {
        auto value = pybind11::cast<std::string_view>(obj);
        if (value == "|+>") {
            return PLUS_KET;
        }
        if (value == "|->") {
            return MINUS_KET;
        }
    }

    std::stringstream ss;
    ss << "Expected a q | xb | xbool | Literal['|+>', '|->'] but got ";
    ss << context_value_name;
    ss << "=" << pybind11::repr(obj);
    throw pybind11::type_error(ss.str());
}

QubitOrBitOrBool kickmix_py::obj_to_qubit_or_bit_or_bool(const pybind11::handle &obj, const char *context_value_name) {
    PyObject *obj_ptr = obj.ptr();
    void *type_ptr = Py_TYPE(obj_ptr);
    if (type_ptr == PYBIND_TYPE_PTR_QUBIT_ID) {
        return *unsafe_pyobject_ptr_to_qubit_id_ptr(obj_ptr);
    }
    if (type_ptr == PYBIND_TYPE_PTR_BIT_ID) {
        return *unsafe_pyobject_ptr_to_bit_id_ptr(obj_ptr);
    }
    if (type_ptr == &PyBool_Type) {
        if (obj_ptr == Py_True) {
            return true;
        }
        if (obj_ptr == Py_False) {
            return false;
        }
    }
    if (type_ptr == &PyLong_Type) {
        long val = PyLong_AsLong(obj.ptr());
        if (val == 0 || val == 1) {
            return (bool)val;
        }
        if (val == -1 && PyErr_Occurred()) {
            PyErr_Clear();
        }
    }

    std::stringstream ss;
    ss << "Expected a q | b | bool but got ";
    ss << context_value_name;
    ss << "=" << pybind11::repr(obj);
    throw pybind11::type_error(ss.str());
}

QubitOrBit kickmix_py::obj_to_qubit_or_bit(const pybind11::handle &obj, const char *context_value_name) {
    PyObject *obj_ptr = obj.ptr();
    void *type_ptr = Py_TYPE(obj_ptr);
    if (type_ptr == PYBIND_TYPE_PTR_QUBIT_ID) {
        return *unsafe_pyobject_ptr_to_qubit_id_ptr(obj_ptr);
    }
    if (type_ptr == PYBIND_TYPE_PTR_BIT_ID) {
        return *unsafe_pyobject_ptr_to_bit_id_ptr(obj_ptr);
    }

    std::stringstream ss;
    ss << "Expected a q | b but got ";
    ss << context_value_name;
    ss << "=" << pybind11::repr(obj);
    throw pybind11::type_error(ss.str());
}

QubitOrXZBitOrXZBool kickmix_py::obj_to_qubit_or_xzbit_or_xzbool(
    const pybind11::handle &obj, const char *context_value_name) {
    PyObject *obj_ptr = obj.ptr();
    void *type_ptr = Py_TYPE(obj_ptr);
    if (type_ptr == PYBIND_TYPE_PTR_QUBIT_ID) {
        return *unsafe_pyobject_ptr_to_qubit_id_ptr(obj_ptr);
    }
    if (type_ptr == PYBIND_TYPE_PTR_BIT_ID) {
        return *unsafe_pyobject_ptr_to_bit_id_ptr(obj_ptr);
    }
    if (type_ptr == PYBIND_TYPE_PTR_XBIT_ID) {
        return *unsafe_pyobject_ptr_to_xbit_id_ptr(obj_ptr);
    }
    if (type_ptr == PYBIND_TYPE_PTR_XBOOL) {
        return *unsafe_pyobject_ptr_to_xbool_ptr(obj_ptr);
    }
    if (type_ptr == &PyBool_Type) {
        return obj_ptr == Py_True;
    }
    if (&PyLong_Type == type_ptr) {
        long val = PyLong_AsLong(obj.ptr());
        if (val == 0 || val == 1) {
            return (bool)val;
        }
        if (val == -1 && PyErr_Occurred()) {
            PyErr_Clear();
        }
    }

    std::stringstream ss;
    ss << "Expected a km.q | km.b | bool | km.xbool | km.xb but got ";
    ss << context_value_name;
    ss << "=" << pybind11::repr(obj);
    throw pybind11::type_error(ss.str());
}

QubitId kickmix_py::obj_to_qubit_id(const pybind11::handle &obj, const char *context_value_name) {
    PyObject *obj_ptr = obj.ptr();
    void *type_ptr = Py_TYPE(obj_ptr);
    if (type_ptr == PYBIND_TYPE_PTR_QUBIT_ID) {
        return *pybind11::cast<QubitId *>(obj);
    }

    std::stringstream ss;
    ss << "Expected a km.q but got ";
    ss << context_value_name;
    ss << "=" << pybind11::repr(obj);
    throw pybind11::type_error(ss.str());
}

BitId kickmix_py::obj_to_bit_id(const pybind11::handle &obj, const char *context_value_name) {
    PyObject *obj_ptr = obj.ptr();
    void *type_ptr = Py_TYPE(obj_ptr);
    if (type_ptr == PYBIND_TYPE_PTR_BIT_ID) {
        return *pybind11::cast<BitId *>(obj);
    }

    std::stringstream ss;
    ss << "Expected a qm.b but got ";
    ss << context_value_name;
    ss << "=" << pybind11::repr(obj);
    throw pybind11::type_error(ss.str());
}

void kickmix_py::register_bid_methods(pybind11::class_<BitId> &c_bid) {
    c_bid.def(
        pybind11::init([](uint32_t index) -> BitId {
            return BitId{index};
        }),
        pybind11::arg("arg"),
        pybind11::pos_only(),
        clean_doc_string(R"DOC(
            Initializes a bit identifier with the given value.
        )DOC")
            .data());

    c_bid.def_property_readonly(
        "id",
        [](BitId &self) -> uint32_t {
            return self.untagged_id();
        },
        clean_doc_string(R"DOC(
            Returns the index of the bit.
        )DOC")
            .data());

    c_bid.def(
        "__hash__",
        [](BitId &self) {
            return pybind11::hash(pybind11::cast(self.tagged_id));
        },
        clean_doc_string(R"DOC(
            Returns a hash of the bit.
        )DOC")
            .data());

    c_bid.def(
        "__repr__",
        [](BitId &self) -> std::string {
            std::stringstream ss;
            ss << "b(";
            ss << self.untagged_id();
            ss << ")";
            return ss.str();
        },
        clean_doc_string(R"DOC(
            Returns a parseable text representation of the bit.
        )DOC")
            .data());

    c_bid.def(
        "__str__",
        [](BitId &self) -> std::string {
            std::stringstream ss;
            ss << self;
            return ss.str();
        },
        clean_doc_string(R"DOC(
            Returns a description of the bit identifier.
        )DOC")
            .data());

    c_bid.def(
        "__lt__",
        [](BitId &self, const pybind11::object &rhs) {
            return is_lt(self, rhs);
        },
        clean_doc_string(R"DOC(
            Does a less-than comparison.
        )DOC")
            .data());

    c_bid.def(pybind11::self == pybind11::self, "Determines if two bit identifiers are equal.");
    c_bid.def(pybind11::self != pybind11::self, "Determines if two bit identifiers are not equal.");

    {
        auto tmp = pybind11::cast(BitId(5));
        BitId *ptr = unsafe_pyobject_ptr_to_bit_id_ptr(tmp.ptr());
        if (ptr != pybind11::cast<BitId *>(tmp)) {
            throw std::invalid_argument(
                "unsafe_pyobject_ptr_to_bit_id_ptr returned the wrong value. "
                "It needs to be rewritten to work with a new version of pybind11.");
        }
    }
}

void kickmix_py::register_rid_methods(pybind11::class_<RegisterId> &c_rid) {
    c_rid.def(
        pybind11::init([](uint32_t index) -> RegisterId {
            return RegisterId{index};
        }),
        pybind11::arg("arg"),
        pybind11::pos_only(),
        clean_doc_string(R"DOC(
            Returns a register identifier with the given index.
        )DOC")
            .data());

    c_rid.def_property_readonly(
        "id",
        [](RegisterId &self) -> uint32_t {
            return self.id;
        },
        clean_doc_string(R"DOC(
            Returns the index of the bit.
        )DOC")
            .data());

    c_rid.def(
        "__hash__",
        [](RegisterId &self) {
            return pybind11::hash(pybind11::cast(self.id));
        },
        clean_doc_string(R"DOC(
            Returns a hash of the bit.
        )DOC")
            .data());

    c_rid.def(
        "__repr__",
        [](RegisterId &self) -> std::string {
            std::stringstream ss;
            ss << "km.r(";
            ss << self.id;
            ss << ")";
            return ss.str();
        },
        clean_doc_string(R"DOC(
            Returns a parseable text representation of the bit.
        )DOC")
            .data());

    c_rid.def(
        "__str__",
        [](RegisterId &self) -> std::string {
            std::stringstream ss;
            ss << self;
            return ss.str();
        },
        clean_doc_string(R"DOC(
            Returns a description of the register identifier.
        )DOC")
            .data());

    c_rid.def(pybind11::self == pybind11::self, "Determines if two register identifiers are equal.");
    c_rid.def(pybind11::self != pybind11::self, "Determines if two register identifiers are not equal.");

    {
        auto tmp = pybind11::cast(RegisterId(5));
        RegisterId *ptr = unsafe_pyobject_ptr_to_register_id_ptr(tmp.ptr());
        if (ptr != pybind11::cast<RegisterId *>(tmp)) {
            throw std::invalid_argument(
                "unsafe_pyobject_ptr_to_register_id_ptr returned the wrong value. "
                "It needs to be rewritten to work with a new version of pybind11.");
        }
    }
}

void kickmix_py::register_xbid_methods(pybind11::class_<XBitId> &c_xbid) {
    c_xbid.def(
        pybind11::init([](uint32_t index) -> XBitId {
            return XBitId{index};
        }),
        pybind11::arg("arg"),
        pybind11::pos_only(),
        clean_doc_string(R"DOC(
            Returns an xb with the given index.
        )DOC")
            .data());

    c_xbid.def_property_readonly(
        "id",
        [](XBitId &self) -> uint32_t {
            return self.untagged_id();
        },
        clean_doc_string(R"DOC(
            Returns the index of the xb.
        )DOC")
            .data());

    c_xbid.def(
        "__hash__",
        [](XBitId &self) {
            return pybind11::hash(pybind11::cast(self.tagged_id));
        },
        clean_doc_string(R"DOC(
            Returns a hash of the xb.
        )DOC")
            .data());

    c_xbid.def(
        "__repr__",
        [](XBitId &self) -> std::string {
            std::stringstream ss;
            ss << "xb(";
            ss << self.untagged_id();
            ss << ")";
            return ss.str();
        },
        clean_doc_string(R"DOC(
            Returns a parseable text representation of the xb.
        )DOC")
            .data());

    c_xbid.def(
        "__str__",
        [](XBitId &self) -> std::string {
            std::stringstream ss;
            ss << self;
            return ss.str();
        },
        clean_doc_string(R"DOC(
            Returns a description of the xb.
        )DOC")
            .data());

    c_xbid.def(
        "__lt__",
        [](XBitId &self, const pybind11::object &rhs) {
            return is_lt(self, rhs);
        },
        clean_doc_string(R"DOC(
            Does a less-than comparison.
        )DOC")
            .data());

    c_xbid.def(pybind11::self == pybind11::self, "Determines if two XBIds are equal.");
    c_xbid.def(pybind11::self != pybind11::self, "Determines if two XBIds are not equal.");

    {
        auto tmp = pybind11::cast(XBitId(5));
        XBitId *ptr = unsafe_pyobject_ptr_to_xbit_id_ptr(tmp.ptr());
        if (ptr != pybind11::cast<XBitId *>(tmp)) {
            throw std::invalid_argument(
                "unsafe_pyobject_ptr_to_xbit_id_ptr returned the wrong value. "
                "It needs to be rewritten to work with a new version of pybind11.");
        }
    }
}

void kickmix_py::register_qid_methods(pybind11::class_<QubitId> &c_qid) {
    c_qid.def(
        pybind11::init([](uint32_t index) -> QubitId {
            return QubitId{index};
        }),
        pybind11::arg("arg"),
        pybind11::pos_only(),
        clean_doc_string(R"DOC(
            Returns a q with the given index.
        )DOC")
            .data());

    c_qid.def_property_readonly(
        "id",
        [](QubitId &self) -> uint32_t {
            return self.untagged_id();
        },
        clean_doc_string(R"DOC(
            Returns the index of the q.
        )DOC")
            .data());

    c_qid.def(
        "__hash__",
        [](QubitId &self) {
            return pybind11::hash(pybind11::cast(self.tagged_id));
        },
        clean_doc_string(R"DOC(
            Returns a hash of the bit.
        )DOC")
            .data());

    c_qid.def(
        "__str__",
        [](QubitId &self) -> std::string {
            std::stringstream ss;
            ss << self;
            return ss.str();
        },
        clean_doc_string(R"DOC(
            Returns a description of the q.
        )DOC")
            .data());

    c_qid.def(
        "__repr__",
        [](QubitId &self) -> std::string {
            std::stringstream ss;
            ss << "q(";
            ss << self.untagged_id();
            ss << ")";
            return ss.str();
        },
        clean_doc_string(R"DOC(
            Returns a parseable text representation of the q.
        )DOC")
            .data());

    c_qid.def(
        "__lt__",
        [](QubitId &self, const pybind11::object &rhs) {
            return is_lt(self, rhs);
        },
        clean_doc_string(R"DOC(
            Does a less-than comparison.
        )DOC")
            .data());

    c_qid.def(pybind11::self < pybind11::self, "Compares two QIds.");
    c_qid.def(pybind11::self == pybind11::self, "Determines if two QIds are equal.");
    c_qid.def(pybind11::self != pybind11::self, "Determines if two QIds are not equal.");

    {
        auto tmp = pybind11::cast(QubitId(5));
        QubitId *ptr = unsafe_pyobject_ptr_to_qubit_id_ptr(tmp.ptr());
        if (ptr != pybind11::cast<QubitId *>(tmp)) {
            throw std::invalid_argument(
                "unsafe_pyobject_ptr_to_qubit_id_ptr returned the wrong value. "
                "It needs to be rewritten to work with a new version of pybind11.");
        }
    }
}

void kickmix_py::register_xbool_methods(pybind11::class_<XBool> &c_xbool) {
    c_xbool.def(
        pybind11::init([](bool value) -> XBool {
            return XBool(value);
        }),
        pybind11::arg("arg"),
        pybind11::pos_only(),
        clean_doc_string(R"DOC(
            Returns an xbool of the given boolean.
        )DOC")
            .data());

    c_xbool.def(
        "__hash__",
        [](XBool &self) {
            return pybind11::hash(pybind11::cast(self.tagged_id));
        },
        clean_doc_string(R"DOC(
            Returns a hash of the bit.
        )DOC")
            .data());

    c_xbool.def(
        "__repr__",
        [](XBool &self) -> std::string_view {
            return self.is_minus_ket() ? "xbool(True)" : "xbool(False)";
        },
        clean_doc_string(R"DOC(
            Returns a parseable text representation of the xbool.
        )DOC")
            .data());

    c_xbool.def(
        "__lt__",
        [](XBool &self, const pybind11::object &rhs) {
            return is_lt(self, rhs);
        },
        clean_doc_string(R"DOC(
            Does a less-than comparison.
        )DOC")
            .data());

    c_xbool.def(pybind11::self == pybind11::self, "Determines if two xbools are equal.");
    c_xbool.def(pybind11::self != pybind11::self, "Determines if two xbools are not equal.");

    {
        auto tmp = pybind11::cast(XBool(true));
        XBool *ptr = unsafe_pyobject_ptr_to_xbool_ptr(tmp.ptr());
        if (ptr != pybind11::cast<XBool *>(tmp)) {
            throw std::invalid_argument(
                "unsafe_pyobject_ptr_to_xbool_ptr returned the wrong value. "
                "It needs to be rewritten to work with a new version of pybind11.");
        }
    }
}
