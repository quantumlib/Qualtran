#include "x.pybind.h"

#include "kickmix/py/val/converted_array.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

static inline void broadcast_x_leaf(PyCircuitBuilder &self, stride_ptr<const QubitId> t, size_t n) {
    self.builder.mut.q0.push_back_many({(uint32_t *)t.ptr, t.stride, n});
    self.builder.mut.op_types.push_back_repeat(OpType::X, n);
}

static inline void broadcast_x_leaf(PyCircuitBuilder &self, stride_ptr<const BitId> t, size_t n) {
    self.builder.mut.b0.push_back_many({(uint32_t *)t.ptr, t.stride, n});
    self.builder.mut.op_types.push_back_repeat(OpType::BIT_INVERT, n);
}

static inline void broadcast_x_leaf(PyCircuitBuilder &self, stride_ptr<const XBitId> t, size_t n) {
    self.builder.mut.bc.push_back_many({(uint32_t *)t.ptr, t.stride, n});
    self.builder.mut.op_types.push_back_repeat(OpType::NEG_IF, n);
}

static inline void broadcast_x_leaf(PyCircuitBuilder &self, stride_ptr<const XBool> t, size_t n) {
    size_t total = 0;
    for (size_t k = 0; k < n; k++) {
        total += t[k].is_minus_ket();
    }
    // DIDNTDO: reduce modulo 2 (because it would break equivalence with the naive loop).
    self.builder.mut.op_types.push_back_repeat(OpType::NEG, total);
}

static void x(PyCircuitBuilder &self, QubitOrXZBitOrXZBool t) {
    switch (t.tag()) {
        case QXZTypeTag8::BIT_ID:
            self.builder.bit_invert((BitId)t);
            break;
        case QXZTypeTag8::QUBIT_ID:
            self.builder.x((QubitId)t);
            break;
        case QXZTypeTag8::XBIT_ID:
            self.builder.neg_if(((XBitId)t).conjugated_by_h());
            break;
        case QXZTypeTag8::XBOOL_VAL:
            if (t.is_minus_ket()) {
                self.builder.neg();
            }
            break;
        default: {
            std::stringstream ss;
            ss << "Can't bit flip " << t;
            throw std::invalid_argument(ss.str());
        }
    }
}

static inline void broadcast_x_leaf(PyCircuitBuilder &self, const stride_ptr<const QubitOrXZBitOrXZBool> &t, size_t n) {
    for (size_t k = 0; k < n; k++) {
        x(self, t[k]);
    }
}

static inline void broadcast_x_mux(PyCircuitBuilder &self, const stride_span_xz &t) {
    size_t n = t.size();
    switch (t.common_type) {
        case QXZTypeTag8::BIT_ID:
            broadcast_x_leaf(self, t.cast_data<BitId>(), n);
            break;
        case QXZTypeTag8::QUBIT_ID:
            broadcast_x_leaf(self, t.cast_data<QubitId>(), n);
            break;
        case QXZTypeTag8::XBIT_ID:
            broadcast_x_leaf(self, t.cast_data<XBitId>(), n);
            break;
        case QXZTypeTag8::XBOOL_VAL:
            broadcast_x_leaf(self, t.cast_data<XBool>(), n);
            break;
        case QXZTypeTag8::MAY_BE_MIXED:
        case QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE:
        case QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_XLIKE:
            broadcast_x_leaf(self, t, n);
            break;
        default: {
            std::stringstream ss;
            ss << "x: can't bit flip ";
            ss << t.common_type << " values";
            throw std::invalid_argument(ss.str());
        }
    }
}

void kickmix_py::broadcast_x_obj(PyCircuitBuilder &self, const pybind11::object &target_obj) {
    auto conv = ConvertedArrayXZ::from_obj(target_obj, "target");
    broadcast_x_mux(self, conv.span);
}
