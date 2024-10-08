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
from typing import Dict, Sequence, TYPE_CHECKING

import attrs

from qualtran import Bloq, bloq_example, BloqDocSpec, DecomposeTypeError, QGF, Register, Signature
from qualtran.bloqs.basic_gates import XGate
from qualtran.symbolics import is_symbolic, SymbolicInt

if TYPE_CHECKING:
    from qualtran import BloqBuilder, Soquet
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@attrs.frozen
class GF2AddK(Bloq):
    r"""In place addition of a constant $k$ for elements in GF($2^m$).

    The bloq implements in place addition of a classical constant $k$ and a quantum register
    $|x\rangle$ storing elements from GF($2^m$). Addition in GF($2^m$) simply reduces to a component
    wise XOR, which can be implemented via X gates.

    $$
    |x\rangle  \rightarrow |x + k\rangle
    $$

    Args:
        bitsize: The degree $m$ of the galois field GF($2^m$). Also corresponds to the number of
            qubits in the input register x.
        k: Integer representation of constant over GF($2^m$) that should be added to the input
            register x.

    Registers:
        x: Input THRU register of size $m$ that stores elements from $GF(2^m)$.
    """

    bitsize: SymbolicInt
    k: SymbolicInt

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', dtype=self.qgf)])

    @cached_property
    def qgf(self) -> QGF:
        return QGF(characteristic=2, degree=self.bitsize)

    @cached_property
    def _bits_k(self) -> Sequence[int]:
        return self.qgf.to_bits(self.qgf.gf_type(self.k))

    def is_symbolic(self):
        return is_symbolic(self.k, self.bitsize)

    def build_composite_bloq(self, bb: 'BloqBuilder', *, x: 'Soquet') -> Dict[str, 'Soquet']:
        if self.is_symbolic():
            raise DecomposeTypeError(f"Cannot decompose symbolic {self}")
        xs = bb.split(x)

        for i, bit in enumerate(self._bits_k):
            if bit == 1:
                xs[i] = bb.add(XGate(), q=xs[i])

        x = bb.join(xs, dtype=self.qgf)

        return {'x': x}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        num_flips = self.bitsize if self.is_symbolic() else sum(self._bits_k)
        return {XGate(): num_flips}

    def on_classical_vals(self, *, x) -> Dict[str, 'ClassicalValT']:
        assert isinstance(x, self.qgf.gf_type)
        return {'x': x + self.qgf.gf_type(self.k)}


@bloq_example
def _gf16_add_k() -> GF2AddK:
    gf16_add_k = GF2AddK(4, 1)
    return gf16_add_k


@bloq_example
def _gf2_add_k_symbolic() -> GF2AddK:
    import sympy

    m, k = sympy.symbols('m, k', positive=True, integers=True)
    gf2_add_k_symbolic = GF2AddK(m, k)
    return gf2_add_k_symbolic


_GF2_ADD_K_DOC = BloqDocSpec(bloq_cls=GF2AddK, examples=(_gf16_add_k, _gf2_add_k_symbolic))
