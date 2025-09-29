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
from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2MulViaKaratsuba, SynthesizeLRCircuit
from qualtran.bloqs.gf_arithmetic.gf2_square import GF2Square
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join
from qualtran.symbolics import bit_length, is_symbolic, Shaped, SymbolicInt

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

        [Concrete quantum cryptanalysisof binary elliptic curves](https://tches.iacr.org/index.php/TCHES/article/view/8741/8341)
        Algorithm 2.
    """

    bitsize: SymbolicInt
    qgf: QGF = attrs.field()

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

    @qgf.default
    def _qgf_default(self) -> QGF:
        return QGF(characteristic=2, degree=self.bitsize)

    @cached_property
    def n_junk_regs(self) -> SymbolicInt:
        if is_symbolic(self.bitsize):
            return 2 * bit_length(self.bitsize - 1) - 2
        return bit_length(self.bitsize - 1) - 2 + max(self.bitsize_hamming_weight - 1, 1)

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

    @cached_property
    def _bits(self) -> list[int]:
        k1 = bit_length(self.bitsize - 1) - 1
        return [-1] + [k1 - i for i, b in enumerate(np.binary_repr(self.bitsize - 1)) if b == '1']

    def build_composite_bloq(self, bb: 'BloqBuilder', *, x: 'Soquet') -> Dict[str, 'SoquetT']:
        if is_symbolic(self.bitsize):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self}")

        if self.bitsize == 1:
            result = bb.allocate(dtype=self.qgf)
            x, result = bb.add(GF2Addition(self.bitsize, self.qgf), x=x, y=result)
            return {'x': x, 'result': result}

        t = (self.bitsize - 1).bit_count()
        k1 = bit_length(self.bitsize - 1) - 1
        k = max(k1 + t - 1, k1 + 1)
        f = [x] + [None] * k
        f[k] = bb.allocate(self.bitsize, self.qgf)
        for i in range(1, k1 + 1):
            f[i - 1], f[k] = bb.add(GF2Addition(self.bitsize, self.qgf), x=f[i - 1], y=f[k])
            f[k] = bb.add(GF2Square(self.bitsize, 2 ** (i - 1), qgf=self.qgf), x=f[k])
            f[i - 1], f[k], f[i] = bb.add(GF2MulViaKaratsuba(self.qgf), x=f[i - 1], y=f[k])
            f[k] = bb.add(GF2Square(self.bitsize, 2 ** (i - 1), qgf=self.qgf).adjoint(), x=f[k])
            f[i - 1], f[k] = bb.add(GF2Addition(self.bitsize, self.qgf), x=f[i - 1], y=f[k])
        bits = self._bits
        if k1 + t - 1 == k:
            bb.free(f[k])
        for s in range(1, t):
            f[k1 + s - 1] = bb.add(
                GF2Square(self.bitsize, 2 ** bits[s + 1], qgf=self.qgf), x=f[k1 + s - 1]
            )
            f[k1 + s - 1], f[bits[s + 1]], f[k1 + s] = bb.add(
                GF2MulViaKaratsuba(self.qgf), x=f[k1 + s - 1], y=f[bits[s + 1]]
            )

        if t == 1:
            if k1 == 0:
                assert self.bitsize == 2
                f[0], f[k] = bb.add(GF2Addition(self.bitsize, self.qgf), x=f[0], y=f[k])
            f[k1], f[k] = f[k], f[k1]

        f[k] = bb.add(GF2Square(self.bitsize, qgf=self.qgf), x=f[k])

        return {'x': f[0], 'result': f[k], 'junk': np.array(f[1:k])}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        if not is_symbolic(self.bitsize) and self.bitsize == 1:
            return {GF2Addition(self.bitsize): 1}
        k1 = bit_length(self.bitsize - 1) - 1
        if is_symbolic(self.bitsize):
            t = bit_length(self.bitsize - 1)
            return {
                GF2Addition(self.bitsize, self.qgf): 2 * k1,
                GF2MulViaKaratsuba(self.qgf): k1 + t - 1,
                SynthesizeLRCircuit(Shaped((self.bitsize, self.bitsize))): 2 * k1 + t,
            }

        t = (self.bitsize - 1).bit_count()
        squaring = (
            {GF2Square(self.bitsize, 2 ** (i - 1), qgf=self.qgf): 1 for i in range(2, k1 + 1)}
            | {
                GF2Square(self.bitsize, 2 ** (i - 1), qgf=self.qgf).adjoint(): 1
                for i in range(1, k1 + 1)
            }
            | {GF2Square(self.bitsize, qgf=self.qgf): 1 + (k1 > 0)}
        )

        for i in self._bits[2:]:
            s = GF2Square(self.bitsize, 2**i, qgf=self.qgf)
            squaring[s] = squaring.get(s, 0) + 1
        mul_count = k1 + t - 1
        add_count = 2 * k1 + (self.bitsize == 2)
        return (
            ({GF2Addition(self.bitsize, self.qgf): add_count} if add_count else {})
            | ({GF2MulViaKaratsuba(self.qgf): mul_count} if mul_count else {})
            | squaring
        )

    def on_classical_vals(self, *, x) -> Dict[str, 'ClassicalValT']:
        assert isinstance(x, self.qgf.gf_type)
        t = (self.bitsize - 1).bit_count()
        k1 = bit_length(self.bitsize - 1) - 1
        k = max(k1 + t - 1, k1 + 1)
        f = [x] + [0] * k
        for i in range(1, k1 + 1):
            f[i] = f[i - 1] ** (2 ** (2 ** (i - 1)) + 1)
        bits = self._bits
        for s in range(1, t):
            f[k1 + s - 1] = f[k1 + s - 1] ** (2 ** (2 ** bits[s + 1]))
            f[k1 + s] = f[k1 + s - 1] * f[bits[s + 1]]

        if t == 1:
            f[k1], f[k] = f[k], f[k1]

        f[k] = f[k] ** 2

        return {
            'x': x,
            'result': x ** (-1) if x else self.qgf.gf_type(0),
            'junk': np.array(f[1:-1]),
        }


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


if __name__ == '__main__':
    poly = [2, 1, 0]
    m = max(poly)
    blq = GF2Inverse(m, QGF(2, m, poly))
    cblq = blq.decompose_bloq()
    import galois

    gf = galois.GF(2, m, irreducible_poly='x^2+x+1', verify=False)
    for x in gf.elements[1:]:
        assert x**-1 == cblq.call_classically(x=x)[1]

    a = blq.bloq_counts()
    b = cblq.bloq_counts()
    for k in a.keys() & b.keys():
        print(repr(k), '\n\t', a[k], b[k])
    # for k in (a.keys() & b.keys()):
    #     assert a[k] == b[k], f'{k=} {a[k]} {b[k]}'

    print('D1:')
    for k in a.keys() - b.keys():
        print(k, a[k])
    print()
    print('D2:')
    for k in b.keys() - a.keys():
        print(k, b[k])
