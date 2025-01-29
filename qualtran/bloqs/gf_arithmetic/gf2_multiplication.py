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
from typing import Dict, Sequence, Set, TYPE_CHECKING, Union

import attrs
import galois
import numpy as np
from galois import GF, Poly

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    DecomposeTypeError,
    QBit,
    QGF,
    Register,
    Side,
    Signature,
)
from qualtran.bloqs.basic_gates import CNOT, Toffoli
from qualtran.symbolics import ceil, is_symbolic, log2, Shaped, SymbolicInt

if TYPE_CHECKING:
    from qualtran import BloqBuilder, Soquet, SoquetT
    from qualtran.resource_counting import BloqCountDictT, BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


def _data_or_shape_to_tuple(data_or_shape: Union[np.ndarray, Shaped]) -> tuple:
    return (
        tuple(data_or_shape.flatten())
        if isinstance(data_or_shape, np.ndarray)
        else (data_or_shape,)
    )


@attrs.frozen
class SynthesizeLRCircuit(Bloq):
    """Synthesize linear reversible circuit using CNOT gates.

    Args:
        matrix: An n x n matrix describing the linear transformation.

    References:
        [Efficient Synthesis of Linear Reversible Circuits](https://arxiv.org/abs/quant-ph/0302002)
    """

    matrix: Union[Shaped, np.ndarray] = attrs.field(eq=_data_or_shape_to_tuple)
    is_adjoint: bool = False

    def __attrs_post_init__(self):
        assert len(self.matrix.shape) == 2
        n, m = self.matrix.shape
        assert is_symbolic(n, m) or n == m

    @cached_property
    def signature(self) -> 'Signature':
        n, _ = self.matrix.shape
        return Signature([Register('q', QBit(), shape=(n,))])

    def on_classical_vals(self, *, q: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        matrix = self.matrix
        assert isinstance(matrix, np.ndarray)
        if self.is_adjoint:
            matrix = np.linalg.inv(matrix)
            assert np.allclose(matrix, matrix.astype(int))
            matrix = matrix.astype(int)
        _, m = matrix.shape
        assert isinstance(q, np.ndarray)
        return {'q': (matrix @ q) % 2}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        n = self.matrix.shape[0]
        return {CNOT(): ceil(n**2 / log2(n))}

    def adjoint(self) -> 'SynthesizeLRCircuit':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)


@attrs.frozen
class GF2Multiplication(Bloq):
    r"""Out of place multiplication over GF($2^m$).

    The bloq implements out of place multiplication of two quantum registers storing elements
    from GF($2^m$) using construction described in Ref[1], which extends the classical construction
    of Ref[2].

    To multiply two m-bit inputs $a = [a_0, a_1, ..., a_{m-1}]$ and $b = [b_0, b_1, ..., b_{m-1}]$,
    the construction computes the output vector $c$ via the following three steps:
        1. Compute $e = U.b$ where $U$ is an upper triangular matrix constructed using $a$.
        2. Compute $Q.e$ where $Q$ is an $m \times (m - 1)$ reduction matrix that depends upon the
            irreducible polynomial $P(x)$ of the galois field $GF(2^m)$. The i'th column of the
            matrix corresponds to coefficients of the polynomial $x^{m + i} % P(x)$. This matrix $Q$
            is a linear reversible circuit that can be implemented only using CNOT gates.
        3. Compute $d = L.b$ where $L$ is a lower triangular matrix constructed using $a$.
        4. Compute $c = d + Q.e$ to obtain the final product.

    Steps 1 and 3 are performed using $n^2$ Toffoli gates and step 2 is performed only using CNOT
    gates.

    Args:
        bitsize: The degree $m$ of the galois field $GF(2^m)$. Also corresponds to the number of
            qubits in each of the two input registers $a$ and $b$ that should be multiplied.
        plus_equal_prod: If True, implements the `PlusEqualProduct` version that applies the
            map $|x\rangle |y\rangle |z\rangle \rightarrow |x\rangle |y\rangle |x + z\rangle$.

    Registers:
        x: Input THRU register of size $m$ that stores elements from $GF(2^m)$.
        y: Input THRU register of size $m$ that stores elements from $GF(2^m)$.
        result: Register of size $m$ that stores the product $x * y$ in $GF(2^m)$.
            If plus_equal_prod is True - result is a THRU register and stores $result + x * y$.
            If plus_equal_prod is False - result is a RIGHT register and stores $x * y$.


    References:
        [On the Design and Optimization of a Quantum Polynomial-Time Attack on
        Elliptic Curve Cryptography](https://arxiv.org/abs/0710.1093)

        [Low complexity bit parallel architectures for polynomial basis multiplication
        over GF(2m)](https://ieeexplore.ieee.org/abstract/document/1306989)
    """

    bitsize: SymbolicInt
    plus_equal_prod: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        result_side = Side.THRU if self.plus_equal_prod else Side.RIGHT
        return Signature(
            [
                Register('x', dtype=self.qgf),
                Register('y', dtype=self.qgf),
                Register('result', dtype=self.qgf, side=result_side),
            ]
        )

    @cached_property
    def qgf(self) -> QGF:
        return QGF(characteristic=2, degree=self.bitsize)

    @cached_property
    def reduction_matrix_q(self) -> np.ndarray:
        m = int(self.bitsize)
        f = self.qgf.gf_type.irreducible_poly
        M = np.zeros((m, m))
        alpha = [1] + [0] * m
        for i in range(m - 1):
            # x ** (m + i) % f
            coeffs = (Poly(alpha, GF(2)) % f).coeffs.tolist()[::-1]
            coeffs = coeffs + [0] * (m - len(coeffs))
            M[i] = coeffs
            alpha += [0]
        M[m - 1][m - 1] = 1
        return np.transpose(M)

    @cached_property
    def synthesize_reduction_matrix_q(self) -> SynthesizeLRCircuit:
        m = self.bitsize
        return (
            SynthesizeLRCircuit(Shaped((m, m - 1)))
            if is_symbolic(m)
            else SynthesizeLRCircuit(self.reduction_matrix_q)
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'Soquet') -> Dict[str, 'Soquet']:
        if is_symbolic(self.bitsize):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self}")
        x, y = soqs['x'], soqs['y']
        result = soqs['result'] if self.plus_equal_prod else bb.allocate(dtype=self.qgf)
        x, y, result = bb.split(x)[::-1], bb.split(y)[::-1], bb.split(result)[::-1]
        m = int(self.bitsize)

        # Step-0: PlusEqualProduct special case.
        if self.plus_equal_prod:
            result = bb.add(self.synthesize_reduction_matrix_q.adjoint(), q=result)

        # Step-1: Multiply Monomials.
        for i in range(m):
            for j in range(i + 1, m):
                ctrl = np.array([x[m - j + i], y[j]])
                ctrl, result[i] = bb.add(Toffoli(), ctrl=ctrl, target=result[i])
                x[m - j + i], y[j] = ctrl[0], ctrl[1]

        # Step-2: Reduce polynomial
        result = bb.add(self.synthesize_reduction_matrix_q, q=result)

        # Step-3: Multiply Monomials
        for i in range(m):
            for j in range(i + 1):
                ctrl = np.array([x[j], y[i - j]])
                ctrl, result[i] = bb.add(Toffoli(), ctrl=ctrl, target=result[i])
                x[j], y[i - j] = ctrl[0], ctrl[1]

        # Done :)
        x, y, result = (
            bb.join(x[::-1], dtype=self.qgf),
            bb.join(y[::-1], dtype=self.qgf),
            bb.join(result[::-1], dtype=self.qgf),
        )
        return {'x': x, 'y': y, 'result': result}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        m = self.bitsize
        plus_equal_prod = (
            {self.synthesize_reduction_matrix_q.adjoint(): 1} if self.plus_equal_prod else {}
        )
        return {Toffoli(): m**2, self.synthesize_reduction_matrix_q: 1} | plus_equal_prod

    def on_classical_vals(self, **vals) -> Dict[str, 'ClassicalValT']:
        assert all(isinstance(val, self.qgf.gf_type) for val in vals.values())
        x, y = vals['x'], vals['y']
        result = vals['result'] if self.plus_equal_prod else self.qgf.gf_type(0)
        return {'x': x, 'y': y, 'result': result + x * y}


@bloq_example
def _gf16_multiplication() -> GF2Multiplication:
    gf16_multiplication = GF2Multiplication(4, plus_equal_prod=True)
    return gf16_multiplication


@bloq_example
def _gf2_multiplication_symbolic() -> GF2Multiplication:
    import sympy

    m = sympy.Symbol('m')
    gf2_multiplication_symbolic = GF2Multiplication(m, plus_equal_prod=False)
    return gf2_multiplication_symbolic


_GF2_MULTIPLICATION_DOC = BloqDocSpec(
    bloq_cls=GF2Multiplication, examples=(_gf16_multiplication, _gf2_multiplication_symbolic)
)


@attrs.frozen
class GF2MultiplyByConstantMod(Bloq):
    r"""Multiply by constant $f(x)$ modulu $m(x)$. Both $f(x)$ and $m(x)$ are constants.

    Args:
        const: The multiplication constant which is an element of the given field.
        galois_field: The galois field that defines the arithmetics.

    Registers:
        g: The polynomial coefficients (in GF(2)).

    References:
        [Space-efficient quantum multiplication of polynomials for binary finite fields with
            sub-quadratic Toffoli gate count](https://arxiv.org/abs/1910.02849v2) Algorithm 1
    """

    const: 'galois.FieldArray'
    galois_field: 'galois.FieldArrayMeta'

    def __attrs_post_init__(self):
        assert isinstance(self.const, self.galois_field)

    @cached_property
    def n(self) -> int:
        return int(self.galois_field.irreducible_poly.degree)

    @cached_property
    def qgf(self) -> QGF:
        return QGF(2, self.n, self.galois_field)

    @staticmethod
    def from_polynomials(
        f_x: Union[Poly, Sequence[int]],
        m_x: Union[Poly, Sequence[int]],
        field_representation: str = 'poly',
    ) -> 'GF2MultiplyByConstantMod':
        if not isinstance(m_x, Poly):
            m_x = Poly.Degrees(m_x)
        if not isinstance(f_x, Poly):
            f_x = Poly.Degrees(f_x)
        gf = GF(2, m_x.degree, irreducible_poly=m_x, repr=field_representation)
        return GF2MultiplyByConstantMod(
            galois_field=gf, const=gf(sum(2**i for i in f_x.nonzero_degrees))
        )

    @cached_property
    def lup(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the LUP decomposition of the matrix representing the operation.

        If m_x is irreducible, then the operation y := (y*f_x)%m_x can be represented
        by a full rank matrix that can be decomposed into PLU where L and U are lower
        and upper traingular matricies and P is a permutation matrix.
        """
        n = self.n
        matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            p = self.const * self.galois_field(2**i)
            for j, v in enumerate(reversed(self.qgf.to_bits(p))):
                matrix[j, i] = v
        P, L, U = GF(2)(matrix).plu_decompose()
        return np.asarray(L, dtype=int), np.asarray(U, dtype=int), np.asarray(P, dtype=int)

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('g', self.qgf)])

    def on_classical_vals(self, g) -> Dict[str, 'ClassicalValT']:
        assert isinstance(g, self.galois_field)
        r = g * self.const
        return {'g': g * self.const}

    def build_composite_bloq(self, bb: 'BloqBuilder', g: 'SoquetT') -> Dict[str, 'SoquetT']:
        L, U, P = self.lup
        if is_symbolic(self.n):
            raise DecomposeTypeError(f"Symbolic decomposition isn't supported for {self}")

        g_arr = bb.split(g)
        g_arr = g_arr[::-1]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if U[i, j]:
                    g_arr[j], g_arr[i] = bb.add(CNOT(), ctrl=g_arr[j], target=g_arr[i])

        for i in reversed(range(self.n)):
            for j in reversed(range(i)):
                if L[i, j]:
                    g_arr[j], g_arr[i] = bb.add(CNOT(), ctrl=g_arr[j], target=g_arr[i])

        column = [*range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if P[i, column[j]]:
                    g_arr[i], g_arr[j] = g_arr[j], g_arr[i]
                    column[i], column[j] = column[j], column[i]
        g_arr = g_arr[::-1]
        g = bb.join(g_arr, dtype=self.qgf)
        return {'g': g}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        L, U, _ = self.lup
        # The number of cnots is the number of non zero off-diagnoal entries in L and U.
        cnots = np.sum(L) + np.sum(U) - 2 * self.n
        if cnots:
            return {CNOT(): cnots}
        return {}

    def __hash__(self):
        return hash((self.const.additive_order, self.galois_field.irreducible_poly))


@bloq_example
def _gf2_multiply_by_constant_modulu() -> GF2MultiplyByConstantMod:
    import galois

    mx = galois.Poly.Degrees([0, 1, 3])  # x^3 + x + 1
    gf = galois.GF(2, 3, irreducible_poly=mx)
    const = gf(5)  # x^2 + 1
    gf2_multiply_by_constant_modulu = GF2MultiplyByConstantMod(const, gf)
    return gf2_multiply_by_constant_modulu


@bloq_example
def _gf2_poly_multiply_by_constant_modulu() -> GF2MultiplyByConstantMod:
    fx = [2, 0]  # x^2 + 1
    mx = [0, 1, 3]  # x^3 + x + 1
    gf2_poly_multiply_by_constant_modulu = GF2MultiplyByConstantMod.from_polynomials(fx, mx)
    return gf2_poly_multiply_by_constant_modulu


_MULTIPLY_BY_CONSTANT_MOD_DOC = BloqDocSpec(
    bloq_cls=GF2MultiplyByConstantMod,
    examples=(_gf2_multiply_by_constant_modulu, _gf2_poly_multiply_by_constant_modulu),
)
