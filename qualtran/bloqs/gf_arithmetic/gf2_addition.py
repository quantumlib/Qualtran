#  Copyright 2024 Google LLC
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
from functools import cached_property
from typing import Dict, Set, TYPE_CHECKING, Union

import attrs

from qualtran import Bloq, bloq_example, BloqDocSpec, DecomposeTypeError, QGF, Register, Signature
from qualtran.bloqs.basic_gates import CNOT
from qualtran.symbolics import is_symbolic, SymbolicInt

if TYPE_CHECKING:
    from qualtran import BloqBuilder, Soquet
    from qualtran.resource_counting import BloqCountDictT, BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@attrs.frozen
class GF2Addition(Bloq):
    r"""In place addition over GF($2^m$).

    The bloq implements in place addition of two quantum registers storing elements
    from GF($2^m$). Addition in GF($2^m$) simply reduces to a component wise XOR, which
    can be implemented via CNOT gates. The addition is performed in-place such that

    $$
    |x\rangle |y\rangle \rightarrow |x\rangle |x + y\rangle
    $$

    Args:
        bitsize: The degree $m$ of the galois field $GF(2^m)$. Also corresponds to the number of
            qubits in each of the two input registers x and y that should be added.

    Registers:
        x: Input THRU register of size $m$ that stores elements from $GF(2^m)$.
        y: Input THRU register of size $m$ that stores elements from $GF(2^m)$.
    """

    bitsize: SymbolicInt

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', dtype=self.qgf), Register('y', dtype=self.qgf)])

    @cached_property
    def qgf(self) -> QGF:
        return QGF(characteristic=2, degree=self.bitsize)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', *, x: 'Soquet', y: 'Soquet'
    ) -> Dict[str, 'Soquet']:
        if is_symbolic(self.bitsize):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self}")
        x, y = bb.split(x), bb.split(y)
        m = int(self.bitsize)
        for i in range(m):
            x[i], y[i] = bb.add(CNOT(), ctrl=x[i], target=y[i])
        x, y = (bb.join(x, dtype=self.qgf), bb.join(y, dtype=self.qgf))
        return {'x': x, 'y': y}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {CNOT(): self.bitsize}

    def on_classical_vals(self, *, x, y) -> Dict[str, 'ClassicalValT']:
        assert isinstance(x, self.qgf.gf_type) and isinstance(y, self.qgf.gf_type)
        return {'x': x, 'y': x + y}


@bloq_example
def _gf16_addition() -> GF2Addition:
    gf16_addition = GF2Addition(4)
    return gf16_addition


@bloq_example
def _gf2_addition_symbolic() -> GF2Addition:
    import sympy

    m = sympy.Symbol('m')
    gf2_addition_symbolic = GF2Addition(m)
    return gf2_addition_symbolic


_GF2_ADDITION_DOC = BloqDocSpec(
    bloq_cls=GF2Addition, examples=(_gf16_addition, _gf2_addition_symbolic)
)
