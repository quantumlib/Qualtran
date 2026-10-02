#include "register_id.h"

#include <sstream>

using namespace kickmix;

std::ostream &kickmix::operator<<(std::ostream &out, RegisterId v) {
    out << 'r';
    out << v.id;
    return out;
}
