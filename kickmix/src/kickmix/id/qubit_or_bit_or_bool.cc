#include "qubit_or_bit_or_bool.h"

using namespace kickmix;

std::string QubitOrBitOrBool::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

std::ostream &kickmix::operator<<(std::ostream &out, const QubitOrBitOrBool &v) {
    if (v.is_qubit()) {
        out << (QubitId)v;
    } else if (v.is_bit()) {
        out << (BitId)v;
    } else if ((bool)v) {
        out << "True";
    } else {
        out << "False";
    }
    return out;
}
