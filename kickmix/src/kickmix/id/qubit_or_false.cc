#include "qubit_or_false.h"

std::ostream &kickmix::operator<<(std::ostream &out, const QubitOrFalse &op) {
    if (op.is_qubit()) {
        out << op.qubit();
    } else {
        out << "False";
    }
    return out;
}
