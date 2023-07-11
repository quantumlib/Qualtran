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
from ._infra.bloq import Bloq

# External imports:
#     networkx - create binst graph, topological sorting
#     numpy - managing soquets in builder and map_soqs
#     cirq_ft - tcomplexity is sum of children
# Internal imports: bloq, registers, _infra
from ._infra.composite_bloq import (
    BloqError,
    CompositeBloq,
    BloqBuilder,
    DidNotFlattenAnythingError,
    SoquetT,
)

# Internal imports: none
# External:
#  - numpy: multiplying bitsizes, making cirq quregs
from ._infra.registers import Register, Signature, Side

# Internal imports: none
# External imports: none
from ._infra.quantum_graph import (
    BloqInstance,
    Connection,
    DanglingT,
    LeftDangle,
    RightDangle,
    Soquet,
)

# --------------------------------------------------------------------------------------------------
