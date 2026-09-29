#include "bit_or_false.h"

std::ostream &kickmix::operator<<(std::ostream &out, const BitIdOrFalse &op) {
    if (op.is_bit()) {
        out << op.bit();
    } else {
        out << "False";
    }
    return out;
}
