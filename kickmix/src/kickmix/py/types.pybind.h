#ifndef KICKGEN_PYBIND_TYPES_H
#define KICKGEN_PYBIND_TYPES_H

#include <pybind11/pybind11.h>

#include "kickmix/id/qubit_or_xzbit_or_xzbool.h"
#include "kickmix/id/register_id.h"

namespace kickmix_py {

struct PyArrayXZ;

extern PyObject *PYBIND_TYPE_PTR_QUBIT_ID;
extern PyObject *PYBIND_TYPE_PTR_BIT_ID;
extern PyObject *PYBIND_TYPE_PTR_XBIT_ID;
extern PyObject *PYBIND_TYPE_PTR_XBOOL;
extern PyObject *PYBIND_TYPE_PTR_REGISTER_ID;
extern PyObject *PYBIND_TYPE_PTR_QCARRAY;

/// This is a faster version of pybind11::cast<QubitId *>(obj).
///
/// This method REQUIRES that the given value is a QubitId.
/// Do not call it unless `Py_TYPE(obj_ptr) == PYBIND_TYPE_PTR_QUBIT_ID`!!!
///
/// Also, pybind doesn't guarantee that what this method is doing will
/// stay stable over time. But it's ~3x faster than pybind::cast when
/// you know the type and so here we are.
static inline kickmix::QubitId *unsafe_pyobject_ptr_to_qubit_id_ptr(PyObject *obj_ptr) {
    return (kickmix::QubitId *)((pybind11::detail::instance *)obj_ptr)->simple_value_holder[0];
}

/// This is a faster version of pybind11::cast<BitId *>(obj).
///
/// This method REQUIRES that the given value is a BitId.
/// Do not call it unless `Py_TYPE(obj_ptr) == PYBIND_TYPE_PTR_BIT_ID`!!!
///
/// Also, pybind doesn't guarantee that what this method is doing will
/// stay stable over time. But it's ~3x faster than pybind::cast when
/// you know the type and so here we are.
static inline kickmix::BitId *unsafe_pyobject_ptr_to_bit_id_ptr(PyObject *obj_ptr) {
    return (kickmix::BitId *)((pybind11::detail::instance *)obj_ptr)->simple_value_holder[0];
}

/// This is a faster version of pybind11::cast<QubitId *>(obj).
///
/// This method REQUIRES that the given value is a QubitId.
/// Do not call it unless `Py_TYPE(obj_ptr) == PYBIND_TYPE_PTR_QUBIT_ID`!!!
///
/// Also, pybind doesn't guarantee that what this method is doing will
/// stay stable over time. But it's ~3x faster than pybind::cast when
/// you know the type and so here we are.
static inline kickmix::RegisterId *unsafe_pyobject_ptr_to_register_id_ptr(PyObject *obj_ptr) {
    return (kickmix::RegisterId *)((pybind11::detail::instance *)obj_ptr)->simple_value_holder[0];
}

/// This is a faster version of pybind11::cast<XBitId *>(obj).
///
/// This method REQUIRES that the given value is a XBitId.
/// Do not call it unless `Py_TYPE(obj_ptr) == PYBIND_TYPE_PTR_XBIT_ID`!!!
///
/// Also, pybind doesn't guarantee that what this method is doing will
/// stay stable over time. But it's ~3x faster than pybind::cast when
/// you know the type and so here we are.
static inline kickmix::XBitId *unsafe_pyobject_ptr_to_xbit_id_ptr(PyObject *obj_ptr) {
    return (kickmix::XBitId *)((pybind11::detail::instance *)obj_ptr)->simple_value_holder[0];
}

/// This is a faster version of pybind11::cast<XBool *>(obj).
///
/// This method REQUIRES that the given value is a XBool.
/// Do not call it unless `Py_TYPE(obj_ptr) == PYBIND_TYPE_PTR_XBOOL`!!!
///
/// Also, pybind doesn't guarantee that what this method is doing will
/// stay stable over time. But it's ~3x faster than pybind::cast when
/// you know the type and so here we are.
static inline kickmix::XBool *unsafe_pyobject_ptr_to_xbool_ptr(PyObject *obj_ptr) {
    return (kickmix::XBool *)((pybind11::detail::instance *)obj_ptr)->simple_value_holder[0];
}

/// This is a faster version of pybind11::cast<PyArrayXZ *>(obj).
///
/// This method REQUIRES that the given value is a PyArrayXZ.
/// Do not call it unless `Py_TYPE(obj_ptr) == PYBIND_TYPE_PTR_QCARRAY`!!!
///
/// Also, pybind doesn't guarantee that what this method is doing will
/// stay stable over time. But it's ~3x faster than pybind::cast when
/// you know the type and so here we are.
static inline PyArrayXZ *unsafe_pyobject_ptr_to_qcarray_ptr(PyObject *obj_ptr) {
    return (PyArrayXZ *)((pybind11::detail::instance *)obj_ptr)->simple_value_holder[0];
}

}  // namespace kickmix_py

#endif
