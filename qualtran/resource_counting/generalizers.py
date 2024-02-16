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

"""Bloq counting generalizers.

`Bloq.get_bloq_counts_graph(...)` takes a `generalizer` argument which can combine
multiple bloqs whose attributes differ in ways that do not affect the cost estimates
into one, more general bloq. The functions in this module can be used as generalizers
for this argument.
"""
from typing import Optional

import attrs
import sympy

from qualtran import Bloq

PHI = sympy.Symbol(r'\phi')
CV = sympy.Symbol("cv")


def ignore_split_join(b: Bloq) -> Optional[Bloq]:
    """A generalizer that ignores split and join operations."""
    from qualtran.bloqs.util_bloqs import Join, Split

    if isinstance(b, (Split, Join)):
        return None
    return b


def ignore_alloc_free(b: Bloq) -> Optional[Bloq]:
    """A generalizer that ignores allocations and frees."""
    from qualtran.bloqs.util_bloqs import Allocate, Free

    if isinstance(b, (Allocate, Free)):
        return None
    return b


def generalize_rotation_angle(b: Bloq) -> Optional[Bloq]:
    """A generalizer that replaces rotation angles with a shared symbol."""
    from qualtran.bloqs.basic_gates import Rx, Ry, Rz

    if isinstance(b, (Rx, Ry, Rz)):
        return attrs.evolve(b, angle=PHI)

    return b


def generalize_cvs(b: Bloq) -> Optional[Bloq]:
    """A generalizer that replaces control variables with a shared symbol."""
    from qualtran.bloqs.and_bloq import And, MultiAnd

    if isinstance(b, And):
        return attrs.evolve(b, cv1=CV, cv2=CV)
    if isinstance(b, MultiAnd):
        return attrs.evolve(b, cvs=(CV,) * len(b.cvs))

    return b


def ignore_cliffords(b: Bloq) -> Optional[Bloq]:
    """A generalizer that ignores known clifford bloqs."""
    from qualtran.bloqs.basic_gates import CNOT, Hadamard, TwoBitSwap, XGate, ZGate
    from qualtran.bloqs.util_bloqs import ArbitraryClifford

    if isinstance(b, (TwoBitSwap, Hadamard, XGate, ZGate, ArbitraryClifford, CNOT)):
        return None

    return b


def cirq_to_bloqs(b: Bloq) -> Optional[Bloq]:
    """A generalizer that replaces Cirq gates with their equivalent bloq, where possible."""
    import cirq

    from qualtran.bloqs.basic_gates import CNOT, Hadamard, TGate, Toffoli, TwoBitSwap, XGate, ZGate
    from qualtran.cirq_interop import CirqGateAsBloq

    if not isinstance(b, CirqGateAsBloq):
        return b

    gate = b.gate
    if gate == cirq.T:
        return TGate()
    if gate == cirq.H:
        return Hadamard()
    if gate == cirq.CNOT:
        return CNOT()
    if gate == cirq.TOFFOLI:
        return Toffoli()
    if gate == cirq.X:
        return XGate()
    if gate == cirq.Z:
        return ZGate()
    if gate == cirq.SWAP:
        return TwoBitSwap()

    return b
