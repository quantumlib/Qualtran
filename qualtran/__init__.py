"""qualtran

isort:skip_file
"""

# --------------------------------------------------------------------------------------------------
# Tier 1: Basic bloq data structures.
#
# Allowed external dependencies: numpy, networkx
# External dependencies allowed for expediency: cirq_ft. This is for implementing the default
#                                                        CompositeBloq.t_complexity() method.
# Allowed internal dependencies: other modules in tier 1.

# Internal imports: none
# External imports: none
from .quantum_graph.bloq import Bloq

# External imports:
#     networkx - create binst graph, topological sorting
#     numpy - managing soquets in builder and map_soqs
#     cirq_ft - tcomplexity is sum of children
# Internal imports: bloq, fancy_registers, quantum_graph
from .quantum_graph.composite_bloq import (
    BloqError,
    CompositeBloq,
    BloqBuilder,
    DidNotFlattenAnythingError,
    SoquetT,
)

# Internal imports: none
# External:
#  - numpy: multiplying bitsizes, making cirq quregs
from .quantum_graph.fancy_registers import Register, Signature, Side

# Internal imports: none
# External imports: none
from .quantum_graph.quantum_graph import (
    BloqInstance,
    Connection,
    DanglingT,
    LeftDangle,
    RightDangle,
    Soquet,
)

# --------------------------------------------------------------------------------------------------
