#include "xbit_id.h"

#include <sstream>

using namespace kickmix;

std::ostream &kickmix::operator<<(std::ostream &out, XBitId v) {
    out << "xb";
    out << v.untagged_id();
    return out;
}

std::string XBitId::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}
