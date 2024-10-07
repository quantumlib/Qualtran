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
import numpy as np
from galois import GF, Poly

from qualtran import Bloq, bloq_example, BloqDocSpec, DecomposeTypeError, QGF, Register, Signature
from qualtran.bloqs.gf_arithmetic.gf2_multiplication import SynthesizeLRCircuit
from qualtran.symbolics import is_symbolic, Shaped, SymbolicInt

if TYPE_CHECKING:
    from qualtran import BloqBuilder, Soquet
    from qualtran.resource_counting import BloqCountDictT, BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@attrs.frozen
class GF2Square(Bloq):
    r"""In place squaring for elements in GF($2^m$)

    The bloq implements in-place squaring of a quantum registers storing elements
    from GF($2^m$). Specifically, it implements the transformation

    $$
        |a\rangle \rightarrow |a^2\rangle
    $$

    The key insight is that for elements in GF($2^m$),
    $$
        a^2 =a_0 + a_1 x^2 + a_2 x^4 + ... + a_{n-1} x^{2(n - 1)}
    $$

    Thus, squaring can be implemented via a linear reversible circuit using only CNOT gates.

    Args:
        bitsize: The degree $m$ of the galois field $GF(2^m)$. Also corresponds to the number of
            qubits in the input register to be squared.

    Registers:
        x: Input THRU register of size $m$ that stores elements from $GF(2^m)$.
    """

    bitsize: SymbolicInt

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', dtype=self.qgf)])

    @cached_property
    def qgf(self) -> QGF:
        return QGF(characteristic=2, degree=self.bitsize)

    @cached_property
    def squaring_matrix(self) -> np.ndarray:
        r"""$m \times m$ matrix that maps the input $x^{i}$ to $x^{2 * i} % P(x)$"""
        m = int(self.bitsize)
        f = self.qgf.gf_type.irreducible_poly
        M = np.zeros((m, m))
        alpha = [0] * m
        for i in range(m):
            # x ** (2 * i) % f
            alpha[-i - 1] = 1
            coeffs = ((Poly(alpha, GF(2)) * Poly(alpha, GF(2))) % f).coeffs.tolist()[::-1]
            coeffs = coeffs + [0] * (m - len(coeffs))
            M[i] = coeffs
            alpha[-i - 1] = 0
        return np.transpose(M)

    @cached_property
    def synthesize_squaring_matrix(self) -> SynthesizeLRCircuit:
        m = self.bitsize
        return (
            SynthesizeLRCircuit(Shaped((m, m)))
            if is_symbolic(m)
            else SynthesizeLRCircuit(self.squaring_matrix)
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', *, x: 'Soquet') -> Dict[str, 'Soquet']:
        if is_symbolic(self.bitsize):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self}")
        x = bb.split(x)[::-1]
        x = bb.add(self.synthesize_squaring_matrix, q=x)
        x = bb.join(x[::-1], dtype=self.qgf)
        return {'x': x}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return {self.synthesize_squaring_matrix: 1}

    def on_classical_vals(self, *, x) -> Dict[str, 'ClassicalValT']:
        assert isinstance(x, self.qgf.gf_type)
        return {'x': x**2}


@bloq_example
def _gf16_square() -> GF2Square:
    gf16_square = GF2Square(4)
    return gf16_square


@bloq_example
def _gf2_square_symbolic() -> GF2Square:
    import sympy

    m = sympy.Symbol('m')
    gf2_square_symbolic = GF2Square(m)
    return gf2_square_symbolic


_GF2_SQUARE_DOC = BloqDocSpec(bloq_cls=GF2Square, examples=(_gf16_square, _gf2_square_symbolic))
