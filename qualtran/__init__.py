#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

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
from ._infra.bloq import Bloq, DecomposeTypeError, DecomposeNotImplementedError

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
    ConnectionT,
)

from ._infra.data_types import (
    QDType,
    QInt,
    QBit,
    QAny,
    QFxp,
    QIntOnesComp,
    QUInt,
    BQUInt,
    QMontgomeryUInt,
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

from ._infra.gate_with_registers import GateWithRegisters

from ._infra.adjoint import Adjoint

from ._infra.controlled import Controlled, CtrlSpec, AddControlledT

from ._infra.bloq_example import BloqExample, bloq_example, BloqDocSpec

# --------------------------------------------------------------------------------------------------
