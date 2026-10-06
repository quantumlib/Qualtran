#include "qubit_or_xbit_or_xbool.h"

#include <sstream>

using namespace kickmix;

std::string QubitOrXBitOrXBool::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

std::ostream &kickmix::operator<<(std::ostream &out, const QubitOrXBitOrXBool &v) {
    if (v.is_qubit()) {
        out << (QubitId)v;
    } else if (v.is_xbit()) {
        out << "(X)b" << v.untagged_id();
    } else if (v.is_minus_ket()) {
        out << "|->";
    } else if (v.is_plus_ket()) {
        out << "|+>";
    } else {
        out << "QubitOrXBitOrXBool{???}";
    }
    return out;
}
