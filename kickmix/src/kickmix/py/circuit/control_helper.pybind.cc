#include "control_helper.pybind.h"

#include "kickmix/py/val/id.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

RaiiXControlObjHelper::RaiiXControlObjHelper(RaiiXControlObjHelper &&other) noexcept
    : builder(other.builder), skip(other.skip), value(other.value), classical_control(other.classical_control) {
    other.builder = nullptr;
    other.classical_control = false;
}
RaiiControlObjHelper::RaiiControlObjHelper(RaiiControlObjHelper &&other) noexcept
    : builder(other.builder), skip(other.skip), value(other.value), classical_control(other.classical_control) {
    other.builder = nullptr;
    other.classical_control = false;
}

RaiiXControlObjHelper &RaiiXControlObjHelper::operator=(RaiiXControlObjHelper &&other) noexcept {
    if (this == &other) {
        return *this;
    }
    builder = other.builder;
    skip = other.skip;
    value = other.value;
    classical_control = other.classical_control;
    other.builder = nullptr;
    other.classical_control = false;
    return *this;
}

RaiiControlObjHelper &RaiiControlObjHelper::operator=(RaiiControlObjHelper &&other) noexcept {
    if (this == &other) {
        return *this;
    }
    builder = other.builder;
    skip = other.skip;
    value = other.value;
    classical_control = other.classical_control;
    other.builder = nullptr;
    other.classical_control = false;
    return *this;
}

RaiiXControlObjHelper::RaiiXControlObjHelper(
    PyCircuitBuilder &builder, const pybind11::object &target_obj, const char *name)
    : builder(&builder) {
    QubitOrXBitOrXBool target_mux = obj_to_qubit_or_xbit_or_xbool(target_obj, name);
    if (target_mux.is_qubit()) {
        skip = false;
        value = (QubitId)target_mux;
        classical_control = false;
    } else if (target_mux.is_xbit()) {
        skip = false;
        value = MINUS_KET;
        BitId b = ((XBitId)target_mux).conjugated_by_h();
        classical_control = b;
        builder.builder.push_condition(b);
    } else if (target_mux == MINUS_KET) {
        skip = false;
        value = MINUS_KET;
        classical_control = false;
    } else {
        skip = true;
        value = MINUS_KET;
        classical_control = false;
    }
}

RaiiXControlObjHelper::~RaiiXControlObjHelper() {
    if (classical_control.is_bit() && builder != nullptr) {
        builder->builder.pop_condition();
        classical_control = false;
        builder = nullptr;
    }
}
RaiiXControlObjHelper::operator QubitOrMinusState() const {
    if (skip) {
        throw std::invalid_argument("Implemented a target incorrectly: didn't check `skip`.");
    }
    return value;
}

RaiiControlObjHelper::RaiiControlObjHelper(
    PyCircuitBuilder &builder, const pybind11::object &control_obj, const char *name)
    : builder(&builder) {
    QubitOrBitOrBool control_mux = obj_to_qubit_or_bit_or_bool(control_obj, name);
    if (control_mux.is_qubit()) {
        skip = false;
        value = (QubitId)control_mux;
        classical_control = false;
    } else if (control_mux.is_bit()) {
        skip = false;
        value = true;
        BitId b = (BitId)control_mux;
        classical_control = b;
        builder.builder.push_condition(b);
    } else if ((bool)control_mux) {
        skip = false;
        value = true;
        classical_control = false;
    } else {
        skip = true;
        value = true;
        classical_control = false;
    }
}
RaiiControlObjHelper::~RaiiControlObjHelper() {
    if (classical_control.is_bit() && builder != nullptr) {
        builder->builder.pop_condition();
        classical_control = false;
        builder = nullptr;
    }
}
RaiiControlObjHelper::operator QubitOrTrue() const {
    if (skip) {
        throw std::invalid_argument("Implemented a control incorrectly: didn't check `skip`.");
    }
    return value;
}
