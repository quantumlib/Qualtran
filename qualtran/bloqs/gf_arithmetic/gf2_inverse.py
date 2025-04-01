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
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join
from qualtran.symbolics import bit_length, ceil, is_symbolic, log2, SymbolicInt

if TYPE_CHECKING:
    from qualtran import BloqBuilder, Soquet, SoquetT
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

    The exponential $a^{2^m - 2}$ is computed using $\mathcal{O}(m)$ squaring and
    $\mathcal{O}(\log_2(m))$ multiplications via Itoh-Tsujii inversion. The algorithm is described on
    page 4 and 5 of Ref[1] and resembles binary exponentiation. The inverse is computed as $B_{n-1}^2$,
    where $B_1 = x$ and $B_{i+j} = B_i B_j^{2^i}$.

    Args:
        bitsize: The degree $m$ of the galois field $GF(2^m)$. Also corresponds to the number of
            qubits in the input register whose inverse should be calculated.

    Registers:
        x: Input THRU register of size $m$ that stores elements from $GF(2^m)$.
        result: Output RIGHT register of size $m$ that stores $x^{-1}$ from $GF(2^m)$.
        junk: Output RIGHT register of size $m$ and shape ($m - 2$) that stores
            results from intermediate multiplications.

    References:
        [Efficient quantum circuits for binary elliptic curve arithmetic: reducing T-gate complexity](https://arxiv.org/abs/1209.6348).
        Amento et al. 2012. Section 2.3

        [Structure of parallel multipliers for a class of fields GF(2^m)](https://doi.org/10.1016/0890-5401(89)90045-X).
        Itoh and Tsujii. 1989.
    """

    bitsize: SymbolicInt

    @cached_property
    def signature(self) -> 'Signature':
        junk_reg = (
            [Register('junk', dtype=self.qgf, shape=(self.n_junk_regs,), side=Side.RIGHT)]
            if is_symbolic(self.bitsize) or self.bitsize > 1
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

    @cached_property
    def n_junk_regs(self) -> SymbolicInt:
        return 2 * bit_length(self.bitsize - 1) + self.bitsize_hamming_weight

    @cached_property
    def bitsize_hamming_weight(self) -> SymbolicInt:
        """Hamming weight of self.bitsize - 1"""
        return (
            bit_length(self.bitsize - 1)
            if is_symbolic(self.bitsize)
            else int(self.bitsize - 1).bit_count()
        )

    def my_static_costs(self, cost_key: 'CostKey'):
        from qualtran._infra.gate_with_registers import total_bits
        from qualtran.resource_counting import QubitCount

        if isinstance(cost_key, QubitCount):
            return total_bits(self.signature.rights())

        return NotImplemented

    def build_composite_bloq(self, bb: 'BloqBuilder', *, x: 'Soquet') -> Dict[str, 'SoquetT']:
        if is_symbolic(self.bitsize):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self}")

        result = bb.allocate(dtype=self.qgf)
        if self.bitsize == 1:
            x, result = bb.add(GF2Addition(self.bitsize), x=x, y=result)
            return {'x': x, 'result': result}

        junk = []
        beta = bb.allocate(dtype=self.qgf)
        x, beta = bb.add(GF2Addition(self.bitsize), x=x, y=beta)
        is_first = True
        bitsize_minus_one = int(self.bitsize - 1)
        for i in range(bitsize_minus_one.bit_length()):
            if (1 << i) & bitsize_minus_one:
                if is_first:
                    beta, result = bb.add(GF2Addition(self.bitsize), x=beta, y=result)
                    is_first = False
                else:
                    for j in range(2**i):
                        result = bb.add(GF2Square(self.bitsize), x=result)
                    beta, result, new_result = bb.add(
                        GF2Multiplication(self.bitsize), x=beta, y=result
                    )
                    junk.append(result)
                    result = new_result
            beta_squared = bb.allocate(dtype=self.qgf)
            beta, beta_squared = bb.add(GF2Addition(self.bitsize), x=beta, y=beta_squared)
            for j in range(2**i):
                beta_squared = bb.add(GF2Square(self.bitsize), x=beta_squared)
            beta, beta_squared, beta_new = bb.add(
                GF2Multiplication(self.bitsize), x=beta, y=beta_squared
            )
            junk.extend([beta, beta_squared])
            beta = beta_new
        junk.append(beta)
        result = bb.add(GF2Square(self.bitsize), x=result)
        assert len(junk) == self.n_junk_regs, f'{len(junk)=}, {self.n_junk_regs=}'
        return {'x': x, 'result': result, 'junk': np.array(junk)}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        if not is_symbolic(self.bitsize) and self.bitsize == 1:
            return {GF2Addition(self.bitsize): 1}
        square_count = self.bitsize + 2 ** ceil(log2(self.bitsize)) - 1
        if not is_symbolic(self.bitsize):
            n = self.bitsize - 1
            square_count -= n & (-n)
        return {
            GF2Addition(self.bitsize): 2 + ceil(log2(self.bitsize)),
            GF2Square(self.bitsize): square_count,
            GF2Multiplication(self.bitsize): ceil(log2(self.bitsize))
            + self.bitsize_hamming_weight
            - 1,
        }

    def on_classical_vals(self, *, x) -> Dict[str, 'ClassicalValT']:
        assert isinstance(x, self.qgf.gf_type)
        junk = []
        bitsize_minus_one = int(self.bitsize - 1)
        beta = x
        result = self.qgf.gf_type(0)
        is_first = True
        for i in range(bitsize_minus_one.bit_length()):
            if (1 << i) & bitsize_minus_one:
                if is_first:
                    is_first = False
                    result = beta
                else:
                    for j in range(2**i):
                        result = result**2
                    junk.append(result)
                    result = result * beta
            beta_squared = beta ** (2 ** (2**i))
            junk.extend([beta, beta_squared])
            beta = beta * beta_squared
        junk.append(beta)
        return {'x': x, 'result': x ** (-1), 'junk': np.array(junk)}


@bloq_example(generalizer=[ignore_split_join, ignore_alloc_free])
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
