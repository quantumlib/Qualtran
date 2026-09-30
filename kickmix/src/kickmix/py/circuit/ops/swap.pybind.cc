#include "swap.pybind.h"

#include "kickmix/py/val/broadcast_resolver.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

void kickmix_py::broadcast_swap_obj(
    PyCircuitBuilder &self, const pybind11::object &target1_obj, const pybind11::object &target2_obj) {
    BroadcastResolver<2> resolver({target1_obj, target2_obj}, "broadcast_swap", {"target1", "target2"});
    self.builder.mux_broadcast_swap(resolver.results_xz[0], resolver.results_xz[1]);
}
