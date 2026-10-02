#ifndef KICKGEN_PYBIND_ANGLE_H
#define KICKGEN_PYBIND_ANGLE_H

#include <pybind11/pybind11.h>

#include "kickmix/util/fixed_precision_angle_128.h"

namespace kickmix_py {

kickmix::FixedPrecisionAngle128 fixed_precision_angle_from_half_turns_obj(const pybind11::object &obj);

}  // namespace kickmix_py

#endif
