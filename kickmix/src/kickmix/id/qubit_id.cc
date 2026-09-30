#include "qubit_id.h"

#include <sstream>

using namespace kickmix;

std::ostream &kickmix::operator<<(std::ostream &out, QubitId v) {
    out << 'q';
    out << v.untagged_id();
    return out;
}

std::string QubitId::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}
