#include "iadd.pybind.h"

#include <cmath>

#include "kickmix/gen/adders/gen_iadd.h"
#include "kickmix/gen/adders/gen_iadd_classical.h"
#include "kickmix/py/circuit/control_helper.pybind.h"
#include "kickmix/py/val/converted_array.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

static void append_iadd_or_isub_obj(
    PyCircuitBuilder &self,
    const pybind11::object &target_obj,
    const pybind11::object &offset_obj,
    const pybind11::object &control_obj,
    double btol,
    bool is_sub) {
    auto target_conv = ConvertedArrayXZ::from_obj(target_obj, "target");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    if (target.empty()) {
        return;
    }

    auto offset_conv =
        ConvertedArrayXZ::from_obj_or_int(offset_obj, target.size(), "offset", /*allow_twos_complement=*/true);
    bool use_classical = offset_conv.span.is_all_classical_and_zlike();

    RaiiControlObjHelper control(self, control_obj, "control");
    if (control.skip) {
        return;
    }

    if (use_classical) {
        auto offset = offset_conv.span.checked_cast_to_qubit_or_bit_or_bool("offset");
        size_t n = target.size();
        size_t num_clean = std::min(self.num_free_qubits(), n);
        array_z clean = array_z::alloc_noinit(num_clean, QXZTypeTag8::QUBIT_ID);
        for (size_t k = 0; k < num_clean; k++) {
            clean[k] = self.alloc_qubit();
        }
        std::span<const QubitId> clean_span{(QubitId *)clean.items, clean.size()};

        if (is_sub) {
            gen_isub_classical(
                self.builder,
                CircuitGenCtx{clean_span},
                target,
                offset,
                /*borrow_in=*/false,
                /*control=*/(QubitOrTrue)control,
                btol);
        } else {
            gen_iadd_classical(
                self.builder,
                CircuitGenCtx{clean_span},
                target,
                offset,
                /*carry_in=*/false,
                /*control=*/(QubitOrTrue)control,
                btol);
        }

        for (size_t k = clean.size(); k--;) {
            self.free_qubit((QubitId)clean[k]);
        }
    } else {
        if (std::isfinite(btol)) {
            throw std::invalid_argument("btol is only supported for classical addition.");
        }
        auto offset = offset_conv.span.checked_cast_to_qubit_ids("offset");
        array_z clean = array_z::alloc_noinit(std::min(self.num_free_qubits(), target.size()), QXZTypeTag8::QUBIT_ID);
        for (size_t k = 0; k < clean.size(); k++) {
            clean[k] = self.alloc_qubit();
        }
        std::span<const QubitId> clean_span{(QubitId *)clean.items, clean.size()};

        if (is_sub) {
            gen_isub(self.builder, CircuitGenCtx{clean_span}, target, offset, (QubitOrTrue)control);
        } else {
            gen_iadd(self.builder, CircuitGenCtx{clean_span}, target, offset, (QubitOrTrue)control);
        }

        for (size_t k = clean.size(); k--;) {
            self.free_qubit((QubitId)clean[k]);
        }
    }
}

void kickmix_py::append_iadd_obj(
    PyCircuitBuilder &self,
    const pybind11::object &offset_obj,
    const pybind11::object &target_obj,
    const pybind11::object &control_obj,
    double btol) {
    append_iadd_or_isub_obj(self, target_obj, offset_obj, control_obj, btol, /*is_sub=*/false);
}

void kickmix_py::append_isub_obj(
    PyCircuitBuilder &self,
    const pybind11::object &offset_obj,
    const pybind11::object &target_obj,
    const pybind11::object &control_obj,
    double btol) {
    append_iadd_or_isub_obj(self, target_obj, offset_obj, control_obj, btol, /*is_sub=*/true);
}
