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
from typing import Callable, Optional

import attrs
import sympy

from qualtran import Bloq
from qualtran.symbolics import HasLength

PHI = sympy.Symbol(r'\phi')
CV = sympy.Symbol("cv")


def _ignore_wrapper(f: Callable[[Bloq], Optional[Bloq]], b: Bloq) -> Optional[Bloq]:
    """Helper for generalizers to traverse wrapper bloqs."""
    from qualtran.bloqs.bookkeeping import AutoPartition

    if isinstance(b, AutoPartition):
        bloq = f(b.bloq)
        return None if bloq is None else attrs.evolve(b, bloq=bloq)
    return b


def ignore_split_join(b: Bloq) -> Optional[Bloq]:
    """A generalizer that ignores split and join operations."""
    from qualtran.bloqs.bookkeeping import AutoPartition, Cast, Join, Partition, Split

    if isinstance(b, (Split, Join, Partition, Cast)):
        return None
    if isinstance(b, AutoPartition):
        return ignore_split_join(b.bloq)
    return b


def ignore_alloc_free(b: Bloq) -> Optional[Bloq]:
    """A generalizer that ignores allocations and frees."""
    from qualtran.bloqs.bookkeeping import Allocate, Free

    if isinstance(b, (Allocate, Free)):
        return None

    return _ignore_wrapper(ignore_alloc_free, b)


def generalize_rotation_angle(b: Bloq) -> Optional[Bloq]:
    """A generalizer that replaces rotation angles with a shared symbol."""
    from qualtran.bloqs.basic_gates import Rx, Ry, Rz, SGate, TGate, XPowGate, YPowGate, ZPowGate

    if isinstance(b, (Rx, Ry, Rz)):
        return attrs.evolve(b, angle=PHI)

    if isinstance(b, (XPowGate, YPowGate, ZPowGate)):
        return attrs.evolve(b, exponent=PHI, global_shift=0)

    if isinstance(b, (TGate, SGate)):
        # ignore `is_adjoint`.
        return attrs.evolve(b, is_adjoint=False)

    return _ignore_wrapper(generalize_rotation_angle, b)


def generalize_cvs(b: Bloq) -> Optional[Bloq]:
    """A generalizer that replaces control variables with a shared symbol."""
    from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd

    if isinstance(b, And):
        return attrs.evolve(b, cv1=CV, cv2=CV)
    if isinstance(b, MultiAnd):
        return attrs.evolve(b, cvs=HasLength(b.n_ctrls))

    return _ignore_wrapper(generalize_cvs, b)


def generalize_cswap_approx(b: Bloq) -> Optional[Bloq]:
    """A generalizer to replace CSwapApprox with a regular-old CSwap."""
    from qualtran import Adjoint
    from qualtran.bloqs.basic_gates import CSwap
    from qualtran.bloqs.swap_network import CSwapApprox

    if isinstance(b, CSwapApprox):
        return CSwap(b.bitsize)
    if isinstance(b, Adjoint) and isinstance(b.subbloq, CSwapApprox):
        return CSwap(b.subbloq.bitsize).adjoint()

    return _ignore_wrapper(generalize_cswap_approx, b)


def ignore_cliffords(b: Bloq) -> Optional[Bloq]:
    """A generalizer that ignores known clifford bloqs."""
    from qualtran.resource_counting.classify_bloqs import bloq_is_clifford

    if bloq_is_clifford(b):
        return None
    return _ignore_wrapper(ignore_cliffords, b)


def cirq_to_bloqs(b: Bloq) -> Optional[Bloq]:
    """A generalizer that replaces Cirq gates with their equivalent bloq, where possible."""
    from qualtran.cirq_interop import CirqGateAsBloq
    from qualtran.cirq_interop._cirq_to_bloq import cirq_gate_to_bloq

    if not isinstance(b, CirqGateAsBloq):
        return _ignore_wrapper(cirq_to_bloqs, b)

    return cirq_gate_to_bloq(b.gate)
