#include "cx.pybind.h"

#include "kickmix/py/val/broadcast_resolver.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

void kickmix_py::broadcast_cx_obj(
    PyCircuitBuilder &self, const pybind11::object &control_obj, const pybind11::object &target_obj) {
    BroadcastResolver<2> resolver({control_obj, target_obj}, "broadcast_cx", {"control", "target"});
    self.builder.mux_broadcast_cx(resolver.results_xz[0], resolver.results_xz[1]);
}
