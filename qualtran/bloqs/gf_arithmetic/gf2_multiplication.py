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
from qualtran.symbolics import is_symbolic, log2, Shaped, SymbolicInt, ceil

if TYPE_CHECKING:
    from qualtran import BloqBuilder, Soquet
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
        matrix: An n x m matrix describing the linear transformation.

    References:
        [Efficient Synthesis of Linear Reversible Circuits](https://arxiv.org/abs/quant-ph/0302002)
    """

    matrix: Union[Shaped, np.ndarray] = attrs.field(eq=_data_or_shape_to_tuple)

    def __attrs_post_init__(self):
        assert len(self.matrix.shape) == 2
        n, m = self.matrix.shape
        assert is_symbolic(n, m) or n >= m

    @cached_property
    def signature(self) -> 'Signature':
        n, _ = self.matrix.shape
        return Signature([Register('q', QBit(), shape=(n,))])

    def on_classical_vals(self, *, q: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        matrix = self.matrix
        assert isinstance(matrix, np.ndarray)
        _, m = matrix.shape
        assert isinstance(q, np.ndarray)
        q_in = q[:m]
        return {'q': (matrix @ q_in) % 2}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        n = self.matrix.shape[0]
        return {CNOT(): ceil(n**2 / log2(n))}


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

    Registers:
        - $x$: Input THRU register of size $m$ that stores elements from $GF(2^m)$.
        - $y$: Input THRU register of size $m$ that stores elements from $GF(2^m)$.
        - $result$: Output RIGHT register of size $m$ that stores the product $x * y$ in $GF(2^m)$.


    References:
        [On the Design and Optimization of a Quantum Polynomial-Time Attack on
        Elliptic Curve Cryptography](https://arxiv.org/abs/0710.1093)

        [Low complexity bit parallel architectures for polynomial basis multiplication
        over GF(2m)](https://ieeexplore.ieee.org/abstract/document/1306989)
    """

    bitsize: SymbolicInt

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('x', dtype=self.qgf),
                Register('y', dtype=self.qgf),
                Register('result', dtype=self.qgf, side=Side.RIGHT),
            ]
        )

    @cached_property
    def qgf(self) -> QGF:
        return QGF(characteristic=2, degree=self.bitsize)

    @cached_property
    def reduction_matrix_q(self) -> np.ndarray:
        m = int(self.bitsize)
        f = self.qgf.gf_type.irreducible_poly
        M = np.zeros((m - 1, m))
        alpha = [1] + [0] * m
        for i in range(m - 1):
            # x ** (m + i) % f
            coeffs = (Poly(alpha, GF(2)) % f).coeffs.tolist()[::-1]
            coeffs = coeffs + [0] * (m - len(coeffs))
            M[i] = coeffs
            alpha += [0]
        return np.transpose(M)

    @cached_property
    def synthesize_reduction_matrix_q(self) -> SynthesizeLRCircuit:
        m = self.bitsize
        return (
            SynthesizeLRCircuit(Shaped((m, m - 1)))
            if is_symbolic(m)
            else SynthesizeLRCircuit(self.reduction_matrix_q)
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', *, x: 'Soquet', y: 'Soquet'
    ) -> Dict[str, 'Soquet']:
        if is_symbolic(self.bitsize):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self}")
        result = bb.allocate(dtype=self.qgf)
        x, y, result = bb.split(x)[::-1], bb.split(y)[::-1], bb.split(result)[::-1]
        m = int(self.bitsize)
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
        return {Toffoli(): m**2, self.synthesize_reduction_matrix_q: 1}


@bloq_example
def _gf16_multiplication() -> GF2Multiplication:
    gf16_multiplication = GF2Multiplication(4)
    return gf16_multiplication


@bloq_example
def _gf2_multiplication_symbolic() -> GF2Multiplication:
    import sympy

    m = sympy.Symbol('m')
    gf2_multiplication_symbolic = GF2Multiplication(m)
    return gf2_multiplication_symbolic


_GF2_MULTIPLICATION_DOC = BloqDocSpec(
    bloq_cls=GF2Multiplication, examples=(_gf16_multiplication, _gf2_multiplication_symbolic)
)
