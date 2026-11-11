#include "z.pybind.h"

#include "kickmix/py/val/converted_array.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

static inline void broadcast_z_leaf(PyCircuitBuilder &self, stride_ptr<const QubitId> t, size_t n) {
    self.builder.mut.q0.push_back_many({(uint32_t *)t.ptr, t.stride, n});
    self.builder.mut.op_types.push_back_repeat(OpType::Z, n);
}

static inline void broadcast_z_leaf(PyCircuitBuilder &self, stride_ptr<const BitId> t, size_t n) {
    self.builder.mut.bc.push_back_many({(uint32_t *)t.ptr, t.stride, n});
    self.builder.mut.op_types.push_back_repeat(OpType::NEG_IF, n);
}

static inline void broadcast_z_leaf(PyCircuitBuilder &self, const stride_span<const bool> &t) {
    size_t n = t.size();
    size_t total = 0;
    for (size_t k = 0; k < n; k++) {
        total += t[k];
    }
    // DIDNTDO: reduce modulo 2 (because it would break equivalence with the naive loop).
    self.builder.mut.op_types.push_back_repeat(OpType::NEG, total);
}

static inline void broadcast_z_mux(PyCircuitBuilder &self, const stride_span_xz &t) {
    size_t n = t.size();
    switch (t.common_type) {
        case QXZTypeTag8::BIT_ID:
            broadcast_z_leaf(self, t.cast_data<BitId>(), n);
            break;
        case QXZTypeTag8::QUBIT_ID:
            broadcast_z_leaf(self, t.cast_data<QubitId>(), n);
            break;
        case QXZTypeTag8::BOOL_VAL:
            broadcast_z_leaf(self, t.cast_data<bool>());
            break;
        case QXZTypeTag8::MAY_BE_MIXED:
        case QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE: {
            stride_span_z tz = t.checked_cast_to_qubit_or_bit_or_bool("target");
            for (size_t k = 0; k < n; k++) {
                self.builder.z(tz[k]);
            }
            break;
        }
        default: {
            std::stringstream ss;
            ss << "z: can't phase flip ";
            ss << t.common_type << " values";
            throw std::invalid_argument(ss.str());
        }
    }
}

void kickmix_py::broadcast_z_obj(PyCircuitBuilder &self, const pybind11::object &target_obj) {
    auto conv = ConvertedArrayXZ::from_obj(target_obj, "target");
    broadcast_z_mux(self, conv.span);
}
