#include "cz.pybind.h"

#include <pybind11/pybind11.h>

#include "kickmix/py/val/broadcast_resolver.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

void kickmix_py::broadcast_cz_obj(
    PyCircuitBuilder &self, const pybind11::object &control1_obj, const pybind11::object &control2_obj) {
    BroadcastResolver<2> resolver({control1_obj, control2_obj}, "broadcast_cz", {"control1", "control2"});
    self.builder.mux_broadcast_cz(resolver.results_xz[0], resolver.results_xz[1]);
}
