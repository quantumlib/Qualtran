#ifndef KICKGEN_PYBIND_GF_ARITHMETIC_H
#define KICKGEN_PYBIND_GF_ARITHMETIC_H

#include <pybind11/pybind11.h>

#include "kickmix/py/circuit/circuit_builder.pybind.h"
#include "kickmix/util/gf2_field.h"

namespace kickmix_py {

pybind11::class_<kickmix::GF2Field> register_gf2_field_class(pybind11::module &m);
void register_gf2_field_methods(pybind11::class_<kickmix::GF2Field> &c);

void append_gf2_iadd_obj(
    PyCircuitBuilder &self,
    const pybind11::object &offset_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj,
    const pybind11::object &control_obj);

void append_gf2_imul_obj(
    PyCircuitBuilder &self,
    const pybind11::object &constant_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj);

void append_gf2_idiv_obj(
    PyCircuitBuilder &self,
    const pybind11::object &constant_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj);

pybind11::object append_init_gf2_mul_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj,
    const pybind11::object &control_obj);

void append_ixor_gf2_mul_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj,
    const pybind11::object &control_obj);

void append_del_gf2_mul_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj,
    const pybind11::object &control_obj);

pybind11::object append_init_gf2_inverse_obj(
    PyCircuitBuilder &self,
    const pybind11::object &input_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj);

pybind11::object append_init_gf2_inverse_with_scaffold_obj(
    PyCircuitBuilder &self,
    const pybind11::object &input_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj);

void append_del_gf2_inverse_obj(
    PyCircuitBuilder &self,
    const pybind11::object &input_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj);

void append_del_gf2_inverse_with_scaffold_obj(
    PyCircuitBuilder &self,
    const pybind11::object &input_obj,
    const pybind11::object &target_obj,
    const pybind11::object &scaffold_obj,
    const pybind11::object &field_obj);

pybind11::object append_init_gf2_div_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj);

pybind11::object append_init_gf2_div_with_scaffold_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj);

void append_ixor_gf2_div_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj);

void append_del_gf2_div_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &field_obj);

void append_del_gf2_div_with_scaffold_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &scaffold_obj,
    const pybind11::object &field_obj);

void append_gf2_phase_by_product_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &mask_obj,
    const pybind11::object &field_obj);

}  // namespace kickmix_py

#endif
