#include "xbool.h"

#include <sstream>

using namespace kickmix;

std::ostream &kickmix::operator<<(std::ostream &out, XBool v) {
    out << (v.is_minus_ket() ? "XBool(true)" : "XBool(false)");
    return out;
}

std::string XBool::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}
