#ifndef KICKGEN_PYBIND_ID_H
#define KICKGEN_PYBIND_ID_H

#include <pybind11/pybind11.h>

#include "kickmix/build/circuit_builder.h"
#include "kickmix/id/qubit_or_xzbit_or_xzbool.h"
#include "kickmix/id/stride_span_xz.h"

namespace kickmix_py {
struct PyArrayXZ;

kickmix::QubitOrXBitOrXBool obj_to_qubit_or_xbit_or_xbool(const pybind11::handle &obj, const char *context_value_name);
kickmix::QubitOrBitOrBool obj_to_qubit_or_bit_or_bool(const pybind11::handle &obj, const char *context_value_name);
kickmix::QubitId obj_to_qubit_id(const pybind11::handle &obj, const char *context_value_name);
kickmix::BitId obj_to_bit_id(const pybind11::handle &obj, const char *context_value_name);
kickmix::QubitOrXZBitOrXZBool obj_to_qubit_or_xzbit_or_xzbool(
    const pybind11::handle &obj, const char *context_value_name);
kickmix::QubitOrBit obj_to_qubit_or_bit(const pybind11::handle &obj, const char *context_value_name);

void register_xbid_methods(pybind11::class_<kickmix::XBitId> &c_xbid);
void register_bid_methods(pybind11::class_<kickmix::BitId> &c_bid);
void register_rid_methods(pybind11::class_<kickmix::RegisterId> &c_bid);
void register_qid_methods(pybind11::class_<kickmix::QubitId> &c_qid);
void register_xbool_methods(pybind11::class_<kickmix::XBool> &c_xbool);
}  // namespace kickmix_py

#endif
