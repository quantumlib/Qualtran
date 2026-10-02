#ifndef KICKMIX_CIRCUIT_BUILDER_TEST_H
#define KICKMIX_CIRCUIT_BUILDER_TEST_H

#include "kickmix/circuit/circuit.h"

namespace kickmix {

Circuit circuit_with_every_operation();

void expect_circuit_has_text_diagram(const Circuit &circuit, std::string_view diagram);

}  // namespace kickmix

#endif
