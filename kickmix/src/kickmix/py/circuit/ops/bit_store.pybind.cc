#include "bit_store.pybind.h"

#include "kickmix/py/val/broadcast_resolver.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

void kickmix_py::broadcast_bit_store_obj(
    PyCircuitBuilder &self,
    const pybind11::object &target_obj,
    const pybind11::object &value_obj,
    const pybind11::object &control_obj) {
    bool value = pybind11::cast<bool>(value_obj);
    BroadcastResolver<2> resolver({target_obj, control_obj}, "broadcast_bit_store", {"target", "control"});
    auto target = resolver.results_xz[0].span.checked_cast_to_bit_ids("target");
    auto control = resolver.results_xz[1].span.checked_cast_to_qubit_or_bit_or_bool("control");
    self.builder.broadcast_bit_store(target, value, control);
}
