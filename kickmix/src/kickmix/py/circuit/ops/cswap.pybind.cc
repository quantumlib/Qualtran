#include "cswap.pybind.h"

#include "kickmix/py/val/broadcast_resolver.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

static inline void broadcast_cswap_case(
    PyCircuitBuilder &self,
    stride_ptr<const bool> c,
    stride_ptr<const QubitId> t1,
    stride_ptr<const QubitId> t2,
    size_t n) {
    QubitId *t1_out = (QubitId *)self.builder.mut.qq0.grab_writeable(n);
    QubitId *t2_out = (QubitId *)self.builder.mut.qq1.grab_writeable(n);
    QubitId *t1_out_start = t1_out;
    for (size_t k = 0; k < n; k++) {
        *t1_out = *t1++;
        *t2_out = *t2++;
        bool keep = *c++ != 0;
        t1_out += keep;
        t2_out += keep;
    }
    self.builder.mut.qq0.rewind_tail((uint32_t *)t1_out);
    self.builder.mut.qq1.rewind_tail((uint32_t *)t2_out);
    self.builder.mut.op_types.push_back_repeat(OpType::SWAP, t1_out - t1_out_start);
}

static inline void broadcast_cswap_case(
    PyCircuitBuilder &self,
    stride_ptr<const BitId> c,
    stride_ptr<const QubitId> t1,
    stride_ptr<const QubitId> t2,
    size_t n) {
    self.builder.mut.qq0.push_back_many({(const uint32_t *)t1.ptr, t1.stride, n});
    self.builder.mut.qq1.push_back_many({(const uint32_t *)t2.ptr, t2.stride, n});
    self.builder.mut.bc.push_back_many({(const uint32_t *)c.ptr, c.stride, n});
    self.builder.mut.op_types.push_back_repeat(OpType::SWAP_IF, n);
}

static inline void broadcast_cswap_case(
    PyCircuitBuilder &self,
    stride_ptr<const QubitId> c,
    stride_ptr<const QubitId> t1,
    stride_ptr<const QubitId> t2,
    size_t n) {
    self.builder.mut.qqq0.push_back_many({(const uint32_t *)c.ptr, c.stride, n});
    self.builder.mut.qqq1.push_back_many({(const uint32_t *)t1.ptr, t1.stride, n});
    self.builder.mut.qqq2.push_back_many({(const uint32_t *)t2.ptr, t2.stride, n});
    self.builder.mut.op_types.push_back_repeat(OpType::CSWAP, n);
}

void kickmix_py::broadcast_cswap_obj(
    PyCircuitBuilder &self,
    const pybind11::object &control_obj,
    const pybind11::object &target1_obj,
    const pybind11::object &target2_obj) {
    BroadcastResolver<3> resolver({control_obj, target1_obj, target2_obj}, "cswap", {"control", "target1", "target2"});
    stride_span_z c = resolver.results_xz[0].span.checked_cast_to_qubit_or_bit_or_bool("control");
    stride_span<const QubitId> t1 = resolver.results_xz[1].span.checked_cast_to_qubit_ids("target1");
    stride_span<const QubitId> t2 = resolver.results_xz[2].span.checked_cast_to_qubit_ids("target2");
    size_t n = c.count;
    switch (c.common_type) {
        case QXZTypeTag8::BOOL_VAL:
            broadcast_cswap_case(self, (stride_ptr<const bool>)c, t1, t2, n);
            break;
        case QXZTypeTag8::BIT_ID:
            broadcast_cswap_case(self, c.cast_data<BitId>(), t1, t2, n);
            break;
        case QXZTypeTag8::QUBIT_ID:
            broadcast_cswap_case(self, c.cast_data<QubitId>(), t1, t2, n);
            break;
        default:
            for (size_t k = 0; k < n; k++) {
                self.builder.cswap(c[k], t1[k], t2[k]);
            }
            break;
    }
}
