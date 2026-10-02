#ifndef KICKMIX_REGISTER_DATA_H
#define KICKMIX_REGISTER_DATA_H

#include <vector>

#include "kickmix/id/qubit_or_bit.h"

namespace kickmix {

/// Just the name and contents (qubits and bits) of a register.
struct RegisterData {
    std::string name;
    std::vector<QubitOrBit> contents;
    bool operator==(const RegisterData &other) const = default;
};

inline std::ostream &operator<<(std::ostream &out, const RegisterData &rhs) {
    out << "RegisterData{";
    out << ".name=\"" << rhs.name << "\"";
    out << ", .contents={";
    for (auto e : rhs.contents) {
        out << e << ", ";
    }
    out << "}}";
    return out;
}

}  // namespace kickmix

#endif
