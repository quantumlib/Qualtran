#include "bit_id.h"

#include <sstream>

#include "xbit_id.h"

using namespace kickmix;

XBitId BitId::conjugated_by_h() const {
    return XBitId(untagged_id());
}

std::ostream &kickmix::operator<<(std::ostream &out, BitId v) {
    out << 'b';
    out << v.untagged_id();
    return out;
}

std::string BitId::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}
