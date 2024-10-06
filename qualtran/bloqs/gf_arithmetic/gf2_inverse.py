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

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    DecomposeTypeError,
    QGF,
    Register,
    Side,
    Signature,
)
from qualtran.bloqs.gf_arithmetic.gf2_addition import GF2Addition
from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2Multiplication
from qualtran.bloqs.gf_arithmetic.gf2_square import GF2Square
from qualtran.symbolics import is_symbolic, SymbolicInt

if TYPE_CHECKING:
    from qualtran import BloqBuilder, Soquet
    from qualtran.resource_counting import BloqCountDictT, BloqCountT, CostKey, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@attrs.frozen
class GF2Inverse(Bloq):
    r"""Out of place inversion for elements in GF($2^m$)

    Given a quantum register storing elements from GF($2^m$), this bloq computes the inverse
    of the given element in a new output register, out-of-place. Specifically,
    it implements the transformation

    $$
        |a\rangle |0\rangle \rightarrow |a\rangle |a^{-1}\rangle
    $$

    Inverse is computed by using Fermat's little theorem for Finite Fields, which states that
    for a finite field $\mathbb{F}$ with $m$ elements, $\forall a \in \mathbb{F}$
    $$
        a^{m} = a
    $$

    When the finite field is GF($2^m$), Fermat's little theorem can be used to obtain the relation

    $$
        a^{-1} = a^{2^m - 2}
    $$

    Thus, the inverse can be obtained via $m - 1$ squaring and multiplication operations.

    Args:
        bitsize: The degree $m$ of the galois field $GF(2^m)$. Also corresponds to the number of
            qubits in the input register whose inverse should be calculated.

    Registers:
        x: Input THRU register of size $m$ that stores elements from $GF(2^m)$.
        result: Output RIGHT register of size $m$ that stores $x^{-1}$ from $GF(2^m)$.
        junk: Output RIGHT register of size $m$ and shape ($m - 2$) that stores
            results from intermediate multiplications.
    """

    bitsize: SymbolicInt

    @cached_property
    def signature(self) -> 'Signature':
        junk_reg = (
            [Register('junk', dtype=self.qgf, shape=(self.bitsize - 2,), side=Side.RIGHT)]
            if is_symbolic(self.bitsize) or self.bitsize > 2
            else []
        )
        return Signature(
            [
                Register('x', dtype=self.qgf),
                Register('result', dtype=self.qgf, side=Side.RIGHT),
                *junk_reg,
            ]
        )

    @cached_property
    def qgf(self) -> QGF:
        return QGF(characteristic=2, degree=self.bitsize)

    def my_static_costs(self, cost_key: 'CostKey'):
        from qualtran.resource_counting import QubitCount

        if isinstance(cost_key, QubitCount):
            return self.signature.n_qubits()

        return NotImplemented

    def build_composite_bloq(self, bb: 'BloqBuilder', *, x: 'Soquet') -> Dict[str, 'Soquet']:
        if is_symbolic(self.bitsize):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self}")
        result = bb.allocate(dtype=self.qgf)
        if self.bitsize == 1:
            x, result = bb.add(GF2Addition(self.bitsize), x=x, y=result)
            return {'x': x, 'result': result}

        x = bb.add(GF2Square(self.bitsize), x=x)
        x, result = bb.add(GF2Addition(self.bitsize), x=x, y=result)

        junk = []
        for i in range(2, self.bitsize):
            x = bb.add(GF2Square(self.bitsize), x=x)
            x, result, new_result = bb.add(GF2Multiplication(self.bitsize), x=x, y=result)
            junk.append(result)
            result = new_result
        x = bb.add(GF2Square(self.bitsize), x=x)
        return {'x': x, 'result': result} | ({'junk': np.array(junk)} if junk else {})

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        if is_symbolic(self.bitsize) or self.bitsize > 2:
            return {
                GF2Addition(self.bitsize): 1,
                GF2Square(self.bitsize): self.bitsize - 1,
                GF2Multiplication(self.bitsize): self.bitsize - 2,
            }
        return {GF2Addition(self.bitsize): 1} | (
            {GF2Square(self.bitsize): 1} if self.bitsize == 2 else {}
        )

    def on_classical_vals(self, *, x) -> Dict[str, 'ClassicalValT']:
        assert isinstance(x, self.qgf.gf_type)
        x_temp = x**2
        result = x_temp
        junk = []
        for i in range(2, int(self.bitsize)):
            junk.append(result)
            x_temp = x_temp * x_temp
            result = result * x_temp
        return {'x': x, 'result': x ** (-1), 'junk': np.array(junk)}


@bloq_example
def _gf16_inverse() -> GF2Inverse:
    gf16_inverse = GF2Inverse(4)
    return gf16_inverse


@bloq_example
def _gf2_inverse_symbolic() -> GF2Inverse:
    import sympy

    m = sympy.Symbol('m', positive=True, integer=True)
    gf2_inverse_symbolic = GF2Inverse(m)
    return gf2_inverse_symbolic


_GF2_INVERSE_DOC = BloqDocSpec(bloq_cls=GF2Inverse, examples=(_gf16_inverse,))
