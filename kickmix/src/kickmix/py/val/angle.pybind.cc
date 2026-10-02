#include "angle.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

FixedPrecisionAngle128 kickmix_py::fixed_precision_angle_from_half_turns_obj(const pybind11::object &obj) {
    auto fraction_class = pybind11::module_::import("fractions").attr("Fraction");
    auto round_func = pybind11::module_::import("builtins").attr("round");

    auto normalized = fraction_class(obj).attr("__mod__")(pybind11::cast(2));
    auto rounded = round_func(normalized * (pybind11::cast(1) << pybind11::cast(127)));
    auto w0 = pybind11::cast<size_t>(rounded & pybind11::cast(UINT64_MAX));
    auto w1 = pybind11::cast<size_t>((rounded >> pybind11::cast(64)) & pybind11::cast(UINT64_MAX));
    return FixedPrecisionAngle128{w0, w1};
}
