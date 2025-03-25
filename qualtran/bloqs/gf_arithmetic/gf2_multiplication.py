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
from typing import Dict, Optional, Sequence, Set, TYPE_CHECKING, Union

import attrs
import galois
import numpy as np
import sympy
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
        if is_symbolic(self.matrix):
            raise ValueError(f"Cannot do classical simulation on symbolic {self}")
        matrix = GF(2)(self.matrix.astype(int))
        assert isinstance(q, np.ndarray)
        q = GF(2)(q)
        assert isinstance(matrix, np.ndarray)
        if self.is_adjoint:
            matrix = GF(2)(np.linalg.inv(matrix))
        _, m = matrix.shape
        assert isinstance(q, np.ndarray)
        return {'q': np.array(matrix @ q)}

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
class GF2MulK(Bloq):
    r"""Multiply by constant $f(x)$ modulo $m(x)$. Both $f(x)$ and $m(x)$ are constants.

    Args:
        const: The multiplication constant which is an element of the given field.
        galois_field: The galois field that defines the arithmetics.

    Registers:
        g: The polynomial coefficients (in GF(2)).

    References:
        [Space-efficient quantum multiplication of polynomials for binary finite fields with
            sub-quadratic Toffoli gate count](https://arxiv.org/abs/1910.02849v2) Algorithm 1
    """

    dtype: QGF
    const: 'SymbolicInt'

    @cached_property
    def galois_field(self):
        return self.qgf.gf_type

    def __attrs_post_init__(self):
        assert is_symbolic(self.const) or isinstance(self.const, int)

    @cached_property
    def m_x(self) -> Poly:
        return self.dtype.gf_type.irreducible_poly

    @cached_property
    def n(self) -> int:
        return self.m_x.degree

    @cached_property
    def qgf(self) -> QGF:
        return self.dtype

    @staticmethod
    def from_polynomials(
        f_x: Union[Poly, Sequence[int]], m_x: Union[Poly, Sequence[int]]
    ) -> 'GF2MulK':
        if not isinstance(m_x, Poly):
            m_x = Poly.Degrees(m_x)
        if not isinstance(f_x, Poly):
            f_x = Poly.Degrees(f_x)
        qgf = QGF(2, m_x.degree, m_x)
        return GF2MulK(dtype=qgf, const=sum(2 ** int(i) for i in f_x.nonzero_degrees))

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
            p = self._const * self.galois_field(2**i)
            for j, v in enumerate(reversed(self.qgf.to_bits(p))):
                matrix[j, i] = v
        P, L, U = GF(2)(matrix).plu_decompose()
        return np.asarray(L, dtype=int), np.asarray(U, dtype=int), np.asarray(P, dtype=int)

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('g', self.qgf)])

    @cached_property
    def _const(self) -> galois.FieldArray:
        return self.galois_field(self.const)

    def on_classical_vals(self, g) -> Dict[str, 'ClassicalValT']:
        return {'g': g * self._const}

    def build_composite_bloq(self, bb: 'BloqBuilder', g: 'Soquet') -> Dict[str, 'SoquetT']:
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


@bloq_example
def _gf2_multiply_by_constant() -> GF2MulK:
    import galois

    from qualtran import QGF

    mx = galois.Poly.Degrees([0, 1, 3])  # x^3 + x + 1
    gf = galois.GF(2, 3, irreducible_poly=mx)
    const = 5  # x^2 + 1
    gf2_multiply_by_constant = GF2MulK(QGF(2, 3, mx), const)
    return gf2_multiply_by_constant


@bloq_example
def _gf2_poly_multiply_by_constant() -> GF2MulK:
    fx = [2, 0]  # x^2 + 1
    mx = [0, 1, 3]  # x^3 + x + 1
    gf2_poly_multiply_by_constant = GF2MulK.from_polynomials(fx, mx)
    return gf2_poly_multiply_by_constant


_MULTIPLY_BY_CONSTANT_MOD_DOC = BloqDocSpec(
    bloq_cls=GF2MulK, examples=(_gf2_multiply_by_constant, _gf2_poly_multiply_by_constant)
)


@attrs.frozen
class MultiplyPolyByOnePlusXk(Bloq):
    r"""Out of place multiplication of $(1 + x^k) fg$

    Applies the transformation
    $$
    \ket{f}\ket{g}\ket{h} \rightarrow \ket{f}{\ket{g}}\ket{h \oplus (1+x^k)fg}
    $$

    Note: While this construction follows Algorithm2 of https://arxiv.org/abs/1910.02849v2,
    it has a slight modification. Namely that the original construction doesn't work in
    some cases where $k < n$. However reversing the order of the first set of CNOTs (line 2)
    makes the construction work for all $k \leq n+1$.

    Args:
        n: The degree of the polynomial ($2^n$ is the size of the galois field).
        k: An integer specifing the shift $1 + x^k$ (or $1 + 2^k$ for galois fields.)

    Registers:
        f: The first polynomial.
        g: The second polyonmial.
        h: The target polynomial.

    References:
        [Space-efficient quantum multiplication of polynomials for binary finite fields with
            sub-quadratic Toffoli gate count](https://arxiv.org/abs/1910.02849v2) Algorithm 2
    """

    n: SymbolicInt
    k: SymbolicInt

    def __attrs_post_init__(self):
        if is_symbolic(self.k):
            return
        assert self.k > 0

    @cached_property
    def l(self):
        return max(0, 2 * self.n - self.k - 1)

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('f', dtype=QBit(), shape=(self.n,)),
                Register('g', dtype=QBit(), shape=(self.n,)),
                Register('h', dtype=QBit(), shape=(2 * self.k + self.l,)),
            ]
        )

    def on_classical_vals(
        self, f: 'ClassicalValT', g: 'ClassicalValT', h: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        if is_symbolic(self.k):
            raise TypeError(f'classical action is not supported for {self=}')
        assert isinstance(f, np.ndarray)
        assert isinstance(g, np.ndarray)
        assert isinstance(h, np.ndarray)
        f_p = Poly(f[::-1])
        g_p = Poly(g[::-1])
        h_p = Poly(h[::-1])
        if self.k > 0:
            h_p += f_p * g_p * Poly.Degrees([0, self.k])
        res = h_p.coefficients().tolist()
        res = [0 for _ in range(len(h) - len(res))] + res
        res = res[::-1]
        return {'f': f, 'g': g, 'h': res}

    def build_composite_bloq(
        self, bb: 'BloqBuilder', f: 'SoquetT', g: 'SoquetT', h: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        n = self.n
        k = self.k
        l = self.l
        if is_symbolic(n) or is_symbolic(k) or is_symbolic(l):
            raise DecomposeTypeError(f"symbolic decomposition is not supported for {self}")
        assert isinstance(f, np.ndarray)
        assert isinstance(g, np.ndarray)
        assert isinstance(h, np.ndarray)
        original = len(h)
        if n > 1:
            # Note: This is the reverse order of what https://arxiv.org/abs/1910.02849v2 has.
            # The is because the reverse order to makes the construction work for k <= n+1.
            for i in reversed(range(l)):
                h[2 * k + i], h[k + i] = bb.add(CNOT(), ctrl=h[2 * k + i], target=h[k + i])
            for i in range(k):
                h[k + i], h[i] = bb.add(CNOT(), ctrl=h[k + i], target=h[i])

            f, g, h[k : 2 * k + l] = bb.add(
                BinaryPolynomialMultiplication(n), f=f, g=g, h=h[k : 2 * k + l]
            )
            for i in range(k):
                h[k + i], h[i] = bb.add(CNOT(), ctrl=h[k + i], target=h[i])
            for i in range(l):
                h[2 * k + i], h[k + i] = bb.add(CNOT(), ctrl=h[2 * k + i], target=h[k + i])
        else:
            h[k], h[0] = bb.add(CNOT(), ctrl=h[k], target=h[0])
            (f[0], g[0]), h[k] = bb.add(Toffoli(), ctrl=[f[0], g[0]], target=h[k])
            h[k], h[0] = bb.add(CNOT(), ctrl=h[k], target=h[0])

        assert len(h) == original, f'{original=} {len(h)}'
        return {'f': f, 'g': g, 'h': h}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        if not is_symbolic(self.n) and self.n == 1:
            return {CNOT(): 2, Toffoli(): 1}
        return {CNOT(): 2 * (self.l + self.k), BinaryPolynomialMultiplication(self.n): 1}


@bloq_example
def _multiplypolybyoneplusxk() -> MultiplyPolyByOnePlusXk:
    n = 5
    k = 3
    multiplypolybyoneplusxk = MultiplyPolyByOnePlusXk(n, k)
    return multiplypolybyoneplusxk


_MULTIPLY_POLY_BY_ONE_PLUS_XK_DOC = BloqDocSpec(
    bloq_cls=MultiplyPolyByOnePlusXk, examples=(_multiplypolybyoneplusxk,)
)


@attrs.frozen
class BinaryPolynomialMultiplication(Bloq):
    r"""Out of place multiplication of binary polynomial multiplication.

    Applies the transformation
    $$
    \ket{f}\ket{g}\ket{h} \rightarrow \ket{f}{\ket{g}}\ket{h \oplus fg}
    $$

    The toffoli cost of this construction is $n^{\log_2{3}}$, while CNOT count is
    upper bounded by $(10 + \frac{1}{3}) n^{\log_2{3}}$.

    Args:
        n: The degree of the polynomial ($2^n$ is the size of the galois field).

    Registers:
        f: The first polynomial.
        g: The second polyonmial.
        h: The target polynomial.

    References:
        [Space-efficient quantum multiplication of polynomials for binary finite fields with
            sub-quadratic Toffoli gate count](https://arxiv.org/abs/1910.02849v2) Algorithm 3
    """

    n: SymbolicInt

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('f', dtype=QBit(), shape=(self.n,)),
                Register('g', dtype=QBit(), shape=(self.n,)),
                Register('h', dtype=QBit(), shape=(2 * self.n - 1,)),
            ]
        )

    def on_classical_vals(
        self, f: 'ClassicalValT', g: 'ClassicalValT', h: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        assert isinstance(f, np.ndarray)
        assert isinstance(g, np.ndarray)
        assert isinstance(h, np.ndarray)
        fx = Poly(f[::-1])
        gx = Poly(g[::-1])
        hx = Poly(h[::-1])
        hx += fx * gx
        res = hx.coefficients().tolist()
        res = [0 for _ in range(len(h) - len(res))] + res
        res = res[::-1]
        return {'f': f, 'g': g, 'h': res}

    @property
    def k(self) -> 'SymbolicInt':
        if isinstance(self.n, int):
            return (self.n + 1) >> 1
        return sympy.ceiling(self.n / 2)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', f: 'SoquetT', g: 'SoquetT', h: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        k, n = self.k, self.n
        if is_symbolic(n) or is_symbolic(k):
            raise DecomposeTypeError(f"symbolic decomposition is not supported for {self}")
        assert isinstance(f, np.ndarray)
        assert isinstance(g, np.ndarray)
        assert isinstance(h, np.ndarray)

        if n == 1:
            (f[0], g[0]), h[0] = bb.add(Toffoli(), ctrl=(f[0], g[0]), target=h[0])
            return {'f': f, 'g': g, 'h': h}

        f[:k], g[:k], h[: 3 * k - 1] = bb.add(
            MultiplyPolyByOnePlusXk(k, k), f=f[:k], g=g[:k], h=h[: 3 * k - 1]
        )
        w = 2 * k + max(0, 2 * (n - k) - k - 1)
        delta = k + w - len(h)
        if delta > 0:
            # This happens for some values (e.g. n=3) where we need to add extra qubits that will always endup in zero state.
            aux = bb.split(bb.allocate(delta))
            h = np.concatenate([h, aux])
            assert isinstance(h, np.ndarray)

        f[k:n], g[k:n], h[k : k + w] = bb.add(  # type: ignore[index]
            MultiplyPolyByOnePlusXk(n - k, k), f=f[k:n], g=g[k:n], h=h[k : k + w]
        )
        if delta > 0:
            aux = h[-delta:]
            h = h[:-delta]
            bb.free(bb.join(aux))

        for i in range(n - k):
            f[k + i], f[i] = bb.add(CNOT(), ctrl=f[k + i], target=f[i])
        for i in range(n - k):
            g[k + i], g[i] = bb.add(CNOT(), ctrl=g[k + i], target=g[i])

        f[:k], g[:k], h[k : 3 * k - 1] = bb.add(  # type: ignore[index]
            BinaryPolynomialMultiplication(k), f=f[:k], g=g[:k], h=h[k : 3 * k - 1]
        )

        for i in range(n - k):
            g[k + i], g[i] = bb.add(CNOT(), ctrl=g[k + i], target=g[i])

        for i in range(n - k):
            f[k + i], f[i] = bb.add(CNOT(), ctrl=f[k + i], target=f[i])

        return {'f': f, 'g': g, 'h': h}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        if not is_symbolic(self.n) and self.n == 1:
            return {Toffoli(): 1}
        if not is_symbolic(self.n) and 2 * self.k == self.n:
            return {
                CNOT(): 4 * (self.n - self.k),
                BinaryPolynomialMultiplication(self.k): 1,
                MultiplyPolyByOnePlusXk(self.k, self.k): 2,
            }
        return {
            CNOT(): 4 * (self.n - self.k),
            BinaryPolynomialMultiplication(self.k): 1,
            MultiplyPolyByOnePlusXk(self.k, self.k): 1,
            MultiplyPolyByOnePlusXk(self.n - self.k, self.k): 1,
        }


@bloq_example
def _binarypolynomialmultiplication() -> BinaryPolynomialMultiplication:
    n = 5
    binarypolynomialmultiplication = BinaryPolynomialMultiplication(n)
    return binarypolynomialmultiplication


_BINARY_POLYNOMIAL_MULTIPLICATION_DOC = BloqDocSpec(
    bloq_cls=BinaryPolynomialMultiplication, examples=(_binarypolynomialmultiplication,)
)


def _qgf_converter(x) -> QGF:
    if isinstance(x, QGF):
        return x
    if isinstance(x, Poly):
        return QGF(2, x.degree, x)
    p = Poly.Degrees(x)
    return QGF(2, p.degree, p)


@attrs.frozen
class GF2ShiftRight(Bloq):
    r"""Multiplies by $2^k$ (or $x^k$ for polynomials) modulo the given irreducible polynomial.

    Applies the transformation
    $$
        \ket{f} \rightarrow \ket{x^k f \mod m(x)}
    $$

    Where the modulus $m(x)$ is the irreducible polynomial defining the galois field arithmetic.

    Args:
        m_x: The irreducible polynomial that defines the galois field.
        k: The number of shifts (i.e. the exponent of $2$ or $x$).

    Registers:
        f: The number (polynomial) to shift.

    References:
        [Space-efficient quantum multiplication of polynomials for binary finite fields with
            sub-quadratic Toffoli gate count](https://arxiv.org/abs/1910.02849v2) Section 3.1
    """

    qgf: QGF = attrs.field(converter=_qgf_converter)
    k: SymbolicInt = 1

    @cached_property
    def n(self):
        return self.qgf.degree

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('f', dtype=self.qgf)])

    @cached_property
    def degrees(self):
        return tuple(sorted(self.qgf.gf_type.irreducible_poly.nonzero_degrees))

    @cached_property
    def gf(self):
        return self.qgf.gf_type

    @cached_property
    def _power_2(self):
        return self.gf(2) ** self.k

    def on_classical_vals(self, f: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        k = self.k
        if is_symbolic(k):
            raise TypeError(f'classical action is not supported for {self}')
        assert isinstance(f, self.gf)
        if k == 0 or self.n == 1:
            return {'f': f}
        return {'f': f * self._power_2}

    def build_composite_bloq(self, bb: 'BloqBuilder', f: 'Soquet') -> Dict[str, 'SoquetT']:
        if is_symbolic(self.k):
            raise DecomposeTypeError(f'symbolic decomposition is not supported for {self}')
        f_arr = bb.split(f)[::-1]
        if self.n > 1:
            for _ in range(self.k):
                f_arr = np.roll(f_arr, 1)
                for i in self.degrees[1:-1]:
                    f_arr[0], f_arr[i] = bb.add(CNOT(), ctrl=f_arr[0], target=f_arr[i])
            f_arr = f_arr[::-1]
        f = bb.join(f_arr)
        return {'f': f}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        if self.k == 0 or self.n == 1:
            return {}
        return {CNOT(): max(len(self.degrees) - 2, 0) * self.k}


@bloq_example
def _gf2shiftright() -> GF2ShiftRight:
    m_x = [5, 2, 0]  # x^5 + x^2 + 1
    gf2shiftright = GF2ShiftRight(QGF(2, 5, m_x), k=3)  # shift by 3
    return gf2shiftright


_GF2_SHIFT_RIGHT_MOD_DOC = BloqDocSpec(bloq_cls=GF2ShiftRight, examples=(_gf2shiftright,))


@attrs.frozen
class _GF2MulViaKaratsubaImpl(Bloq):
    """Multiply two GF2 numbers (or binary polynomials) using quantum karatsuba algorithm."""

    qgf: QGF = attrs.field(converter=_qgf_converter)

    @cached_property
    def n(self):
        return int(self.qgf.degree)

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('f', dtype=self.qgf),
                Register('g', dtype=self.qgf),
                Register('h', dtype=self.qgf),
            ]
        )

    @cached_property
    def k(self):
        if isinstance(self.n, int):
            return (self.n + 1) // 2
        else:
            return sympy.ceiling(self.n / 2)

    @cached_property
    def m_x(self):
        return self.qgf.gf_type.irreducible_poly

    def build_composite_bloq(
        self, bb: 'BloqBuilder', f: 'Soquet', g: 'Soquet', h: 'Soquet'
    ) -> Dict[str, 'Soquet']:
        if is_symbolic(self.k, self.n):
            raise DecomposeTypeError(f"Symbolic Decomposition is not supported for {self}")

        f_arr = bb.split(f)
        g_arr = bb.split(g)
        h_arr = bb.split(h)
        f_arr = f_arr[::-1]
        g_arr = g_arr[::-1]
        h_arr = h_arr[::-1]

        for i in range(self.n - self.k):
            f_arr[self.k + i], f_arr[i] = bb.add(CNOT(), ctrl=f_arr[self.k + i], target=f_arr[i])
        for i in range(self.n - self.k):
            g_arr[self.k + i], g_arr[i] = bb.add(CNOT(), ctrl=g_arr[self.k + i], target=g_arr[i])
        f_arr[: self.k], g_arr[: self.k], h_arr[: 2 * self.k - 1] = bb.add(
            BinaryPolynomialMultiplication(self.k),
            f=f_arr[: self.k],
            g=g_arr[: self.k],
            h=h_arr[: 2 * self.k - 1],
        )

        for i in range(self.n - self.k):
            g_arr[self.k + i], g_arr[i] = bb.add(CNOT(), ctrl=g_arr[self.k + i], target=g_arr[i])
        for i in range(self.n - self.k):
            f_arr[self.k + i], f_arr[i] = bb.add(CNOT(), ctrl=f_arr[self.k + i], target=f_arr[i])

        h = bb.join(h_arr[: self.n][::-1], self.qgf)
        h = bb.add(GF2MulK.from_polynomials([0, self.k], self.m_x).adjoint(), g=h)
        h_arr[: self.n] = bb.split(h)[::-1]
        f_arr[self.k : self.n], g_arr[self.k : self.n], h_arr[: 2 * (self.n - self.k) - 1] = bb.add(
            BinaryPolynomialMultiplication(self.n - self.k),
            f=f_arr[self.k : self.n],
            g=g_arr[self.k : self.n],
            h=h_arr[: 2 * (self.n - self.k) - 1],
        )

        h = bb.join(h_arr[: self.n][::-1], self.qgf)
        h = bb.add(GF2ShiftRight(self.qgf, self.k), f=h)
        h_arr[: self.n] = bb.split(h)[::-1]

        f_arr[: self.k], g_arr[: self.k], h_arr[: 2 * self.k - 1] = bb.add(
            BinaryPolynomialMultiplication(self.k),
            f=f_arr[: self.k],
            g=g_arr[: self.k],
            h=h_arr[: 2 * self.k - 1],
        )
        h = bb.join(h_arr[: self.n][::-1], self.qgf)
        h = bb.add(GF2MulK.from_polynomials([0, self.k], self.m_x), g=h)
        h_arr[: self.n] = bb.split(h)[::-1]

        f_arr = f_arr[::-1]
        g_arr = g_arr[::-1]
        h_arr = h_arr[::-1]
        f = bb.join(f_arr, self.qgf)
        g = bb.join(g_arr, self.qgf)
        h = bb.join(h_arr, self.qgf)
        return {'f': f, 'g': g, 'h': h}


@attrs.frozen
class GF2MulViaKaratsuba(Bloq):
    r"""Multiplies two GF($2^n$) numbers (or binary polynomials) modulo $m(x)$.

    Applies the transformation
    $$
        \ket{f}\ket{g} \rightarrow \ket{f} \ket{g} \ket{f*g \mod m(x)}
    $$

    Where the modulus $m(x)$ is the irreducible polynomial defining the galois field arithmetic.
    The toffoli complexity is $n^{\log_2{3}}$

    Args:
        m_x: The irreducible polynomial that defines the galois field.
        uncompute: Whether to compute or uncompute the product.

    Registers:
        x: A TRHU register representing the first number (or polynomial).
        y: A TRHU register representing the second number (polynomial).
        result: The result (a RIGHT register).

    References:
        [Space-efficient quantum multiplication of polynomials for binary finite fields with
            sub-quadratic Toffoli gate count](https://arxiv.org/abs/1910.02849v2) Algorithm 4.
    """

    dtype: QGF = attrs.field(converter=_qgf_converter)
    uncompute: bool = False

    @cached_property
    def m_x(self):
        return self.dtype.gf_type.irreducible_poly

    def __attrs_post_init__(self):
        if self.m_x.degree < 2:
            raise ValueError(f'GF2MulViaKaratsuba is not supported for {self.m_x}')

    @cached_property
    def n(self):
        return int(self.m_x.degrees.max())

    @cached_property
    def gf(self):
        return self.qgf.gf_type

    @cached_property
    def qgf(self):
        return self.dtype

    def adjoint(self) -> 'GF2MulViaKaratsuba':
        return attrs.evolve(self, uncompute=not self.uncompute)

    @cached_property
    def signature(self) -> 'Signature':
        # C is directional
        side = Side.LEFT if self.uncompute else Side.RIGHT
        return Signature(
            [
                Register('x', dtype=self.qgf),
                Register('y', dtype=self.qgf),
                Register('result', dtype=self.qgf, side=side),
            ]
        )

    @cached_property
    def k(self):
        if isinstance(self.n, int):
            return (self.n + 1) // 2
        else:
            return sympy.ceiling(self.n / 2)

    @cached_property
    def _GF2MulViaKaratsubamod_impl(self) -> Bloq:
        impl = _GF2MulViaKaratsubaImpl(self.m_x)
        if self.uncompute:
            return impl.adjoint()
        return impl

    def build_composite_bloq(
        self, bb: 'BloqBuilder', x: 'Soquet', y: 'Soquet', **soqs: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.k, self.n):
            raise DecomposeTypeError(f"Symbolic Decomposition is not supported for {self}")

        if self.uncompute:
            result = soqs['result']
        else:
            result = bb.allocate(self.n, self.qgf)

        x, y, result = bb.add_from(self._GF2MulViaKaratsubamod_impl, f=x, g=y, h=result)

        if self.uncompute:
            bb.free(result)  # type: ignore[arg-type]
            return {'x': x, 'y': y}

        return {'x': x, 'y': y, 'result': result}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        if self.n == 1:
            return {Toffoli(): 1}
        if not is_symbolic(self.n) and 2 * self.k == self.n:
            return {
                CNOT(): 4 * (self.n - self.k),
                BinaryPolynomialMultiplication(self.k): 3,
                GF2MulK.from_polynomials([0, self.k], self.m_x): 1,
                GF2MulK.from_polynomials([0, self.k], self.m_x).adjoint(): 1,
                GF2ShiftRight(self.qgf, self.k): 1,
            }
        return {
            CNOT(): 4 * (self.n - self.k),
            BinaryPolynomialMultiplication(self.k): 2,
            BinaryPolynomialMultiplication(self.n - self.k): 1,
            GF2MulK.from_polynomials([0, self.k], self.m_x): 1,
            GF2MulK.from_polynomials([0, self.k], self.m_x).adjoint(): 1,
            GF2ShiftRight(self.qgf, self.k): 1,
        }

    def on_classical_vals(
        self, x: 'SymbolicInt', y: 'SymbolicInt', result: Optional['SymbolicInt'] = None
    ) -> Dict[str, 'ClassicalValT']:
        assert isinstance(x, self.gf)
        assert isinstance(y, self.gf)
        if self.uncompute:
            assert x * y == result
            return {'x': x, 'y': y}
        return {'x': x, 'y': y, 'result': x * y}


@bloq_example
def _gf2mulviakaratsuba() -> GF2MulViaKaratsuba:
    m_x = [5, 2, 0]  # x^5 + x^2 + 1
    gf2mulviakaratsuba = GF2MulViaKaratsuba(QGF(2, 5, m_x))
    return gf2mulviakaratsuba


_GF2_MUL_DOC = BloqDocSpec(bloq_cls=GF2MulViaKaratsuba, examples=(_gf2mulviakaratsuba,))
