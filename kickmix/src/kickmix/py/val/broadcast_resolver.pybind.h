#ifndef KICKGEN_PYBIND_BROADCAST_RESOLVER_H
#define KICKGEN_PYBIND_BROADCAST_RESOLVER_H

#include "kickmix/py/val/converted_array.pybind.h"

namespace kickmix_py {

template <size_t m>
struct BroadcastResolver {
    ConvertedArrayXZ results_xz[m];

    BroadcastResolver(
        std::array<pybind11::handle, m> values,
        const char *context_operation_name,
        std::array<const char *, m> context_value_names) {
        for (size_t k = 0; k < m; k++) {
            results_xz[k] = ConvertedArrayXZ::from_obj(values[k], context_value_names[k]);
        }

        size_t n = 1;
        for (size_t k = 0; k < m; k++) {
            if (!results_xz[k].was_singleton) {
                n = results_xz[k].span.count;
                break;
            }
        }
        for (size_t k = 0; k < m; k++) {
            if (!results_xz[k].was_singleton && results_xz[k].span.count != n) {
                std::stringstream ss;
                ss << "Inconsistent broadcast sizes for " << context_operation_name << ":\n";
                for (size_t k2 = 0; k2 < m; k2++) {
                    if (results_xz[k2].was_singleton) {
                        ss << "    (singleton) ";
                    } else {
                        ss << "    (len=" << results_xz[k2].span.count << ") ";
                    }
                    ss << context_value_names[k2] << " = " << pybind11::repr(values[k2]) << "\n";
                }
                throw std::invalid_argument(ss.str());
            }
        }
        for (size_t k = 0; k < m; k++) {
            if (results_xz[k].was_singleton) {
                results_xz[k].span.stride = 0;
                results_xz[k].span.count = n;
            }
        }
    }
};

}  // namespace kickmix_py

#endif
