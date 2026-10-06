#ifndef KICKGEN_PYBIND_CONTROL_HELPER_H
#define KICKGEN_PYBIND_CONTROL_HELPER_H

#include "circuit_builder.pybind.h"

namespace kickmix_py {

struct RaiiXControlObjHelper {
    PyCircuitBuilder *builder;
    bool skip = false;
    kickmix::QubitOrMinusState value = kickmix::MINUS_KET;
    kickmix::BitIdOrFalse classical_control = false;

    RaiiXControlObjHelper(PyCircuitBuilder &builder, const pybind11::object &xcontrol_obj, const char *name);
    ~RaiiXControlObjHelper();
    RaiiXControlObjHelper(const RaiiXControlObjHelper &) = delete;
    RaiiXControlObjHelper &operator=(const RaiiXControlObjHelper &) = delete;
    RaiiXControlObjHelper(RaiiXControlObjHelper &&) noexcept;
    RaiiXControlObjHelper &operator=(RaiiXControlObjHelper &&) noexcept;

    operator kickmix::QubitOrMinusState() const;
};

struct RaiiControlObjHelper {
    PyCircuitBuilder *builder;
    bool skip = false;
    kickmix::QubitOrTrue value = true;
    kickmix::BitIdOrFalse classical_control = false;

    RaiiControlObjHelper(PyCircuitBuilder &builder, const pybind11::object &control_obj, const char *name);
    ~RaiiControlObjHelper();
    RaiiControlObjHelper(const RaiiControlObjHelper &) = delete;
    RaiiControlObjHelper &operator=(const RaiiControlObjHelper &) = delete;
    RaiiControlObjHelper(RaiiControlObjHelper &&) noexcept;
    RaiiControlObjHelper &operator=(RaiiControlObjHelper &&) noexcept;

    operator kickmix::QubitOrTrue() const;
};

}  // namespace kickmix_py

#endif
