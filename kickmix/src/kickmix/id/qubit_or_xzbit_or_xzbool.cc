#include "qubit_or_xzbit_or_xzbool.h"

#include <sstream>

using namespace kickmix;

std::string QubitOrXZBitOrXZBool::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

std::ostream &kickmix::operator<<(std::ostream &out, const QubitOrXZBitOrXZBool &v) {
    if (v.is_qubit()) {
        out << (QubitId)v;
    } else if (v.is_bit()) {
        out << (BitId)v;
    } else if (v.is_xbit()) {
        out << (XBitId)v;
    } else if (v.is_xbool()) {
        out << (XBool)v;
    } else if (v.is_bool()) {
        out << ((bool)v ? "true" : "false");
    } else {
        out << "QubitOrXZBitOrXZBool{.tagged_id=" << v.tagged_id << "}";
    }
    return out;
}
