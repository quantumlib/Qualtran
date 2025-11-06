#  Copyright 2025 Google LLC
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

"""A representation of a GridOperator as defined in https://arxiv.org/abs/1403.2975 and its operations."""

import attrs
import numpy as np

import qualtran.rotation_synthesis._typing as rst
import qualtran.rotation_synthesis.rings.zsqrt2 as zsqrt2
import qualtran.rotation_synthesis.rings.zw as zw


@attrs.frozen
class GridOperator:
    r"""A scaled GridOperator.

    The grid operator is defined in section 5.3 of https://arxiv.org/abs/1403.2975 as
    $$
        G = \begin{bmatrix}
            a + \frac{a'}{\sqrt{2}} & b + \frac{b'}{\sqrt{2}} \\
            c + \frac{c'}{\sqrt{2}} & d + \frac{d'}{\sqrt{2}} \\
        \end{bmatrix}
    $$

    This class however represents a GridOperator multiplied by $\sqrt{2}$, this way each entry
    of the matrix is represented by an element of $\mathbb{Z}[\sqrt{2}$. All operations of the 
    class take this into account, for example __matmul__ returns (self @ other) / sqrt(2). 

    Attributes:
        matrix: A 2x2 matrix where each element belongs to $\mathbb{Z}[\sqrt{2}]$
    """

    matrix: np.ndarray = attrs.field(converter=np.array)

    def __attrs_post_init__(self):
        m = self.matrix
        assert isinstance(m.dtype, object)
        assert m.shape == (2, 2)
        assert all(isinstance(x, zsqrt2.ZSqrt2) for x in m.reshape(-1))
        assert sum(x.a for x in self.matrix.reshape(-1)) % 2 == 0
        assert sum(x.b for x in self.matrix.reshape(-1)) % 2 == 0

    def __pow__(self, k) -> "GridOperator":
        if k == 0:
            return ISqrt2
        x = self
        y = ISqrt2
        while k > 1:
            if k & 1:
                y = y @ x
            x = x @ x
            k >>= 1
        return x @ y

    def shift(self, k: rst.Integral) -> "GridOperator":
        """Returns the operator that equivalent to shift(k)-self-shift(k).

        Note: this method follows the mathematical formulae in Lemma A.9 which differs from
        the text above it which says shift(k)-self-shift(-k).

        Args:
            k: An integeral shift.
        """
        k = int(k)
        if k >= 0:
            lambda_value = zsqrt2.LAMBDA
            l_inv = zsqrt2.LAMBDA_INV
        else:
            l_inv = zsqrt2.LAMBDA
            lambda_value = zsqrt2.LAMBDA_INV

        k = abs(k)
        left = GridOperator(
            np.array([[zsqrt2.One, zsqrt2.Zero], [zsqrt2.Zero, l_inv**k]]) * zsqrt2.SQRT_2
        )
        right = GridOperator(
            np.array([[lambda_value**k, zsqrt2.Zero], [zsqrt2.Zero, zsqrt2.One]]) * zsqrt2.SQRT_2
        )
        return left @ self @ right

    def __matmul__(self, other) -> "GridOperator":
        assert isinstance(other, GridOperator)
        res = self.matrix @ other.matrix
        # Divide by sqrt(2) to preserve the representation.
        for i in range(2):
            for j in range(2):
                res[i, j] = res[i, j].divide_by_sqrt2()
        return GridOperator(res)

    def sqrt2_conj(self) -> "GridOperator":
        """Returns the sqrt2-conjugate of the operator"""
        res = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                res[i, j] = self.matrix[i, j].conj()
        return GridOperator(res)

    def value(self, sqrt2: rst.Real) -> np.ndarray:
        """Returns a representation of the -scaled- operator as matrix of real rst.

        Args:
            sqrt2: the sqrt(2) value to use.
        """
        ret = []
        for v in self.matrix.flat:
            ret.append(v.value(sqrt2))
        return np.array(ret).reshape((2, 2))

    def actual_value(self, sqrt2: rst.Real) -> np.ndarray:
        """Returns a representation of the operator as matrix of real rst.

        This matrix is equivalent to self.value(sqrt2) / sqrt2.

        Args:
            sqrt2: the sqrt(2) value to use.
        """
        ret = []
        for v in self.matrix.flat:
            a, b = v.a, v.b
            ret.append(a / sqrt2 + b)
        return np.array(ret).reshape((2, 2))

    def scaled_inverse(self) -> "GridOperator":
        """Returns the inverse of the operator multiplied by sqrt(2)."""
        a, b, c, d = self.matrix.reshape(-1)
        det: zsqrt2.ZSqrt2 = a * d - b * c
        assert det.b == 0 and det.a in (-2, 2)
        sgn = -1 if det.a == -2 else 1
        return GridOperator(np.array([[d, -b], [-c, a]]) * sgn)

    def apply(self, z: zw.ZW) -> zw.ZW:
        r"""Applies the operator on the given point in $\mathbb{\omega}$."""

        # The operator is a 2x2 matrix that acts on the real and imaginary parts of a point.
        # We decompose the point into its Z[sqrt(2)] representation, apply the operator and
        # return the result.
        x, y, include_w = z.to_zsqrt2()
        a, b, c, d = self.matrix.reshape(-1)

        if include_w:
            x = x * zsqrt2.SQRT_2 + zsqrt2.One
            y = y * zsqrt2.SQRT_2 + zsqrt2.One
            xp = a * x + b * y
            yp = c * x + d * y
            assert xp.a % 2 == yp.a % 2 == 0
            assert xp.b % 2 == yp.b % 2
            if xp.b % 2 == 0:
                xp = zsqrt2.ZSqrt2(xp.a // 2, xp.b // 2)
                yp = zsqrt2.ZSqrt2(yp.a // 2, yp.b // 2)
                return zw.ZW.from_pair(xp, yp, False)
            else:
                xp = zsqrt2.ZSqrt2(xp.a // 2, (xp.b - 1) // 2)
                yp = zsqrt2.ZSqrt2(yp.a // 2, (yp.b - 1) // 2)
                return zw.ZW.from_pair(xp, yp, True)
        else:
            xp = a * x + b * y
            yp = c * x + d * y
            assert (xp.a + yp.a) % 2 == 0
            if xp.a % 2 == 0:
                return zw.ZW.from_pair(xp.divide_by_sqrt2(), yp.divide_by_sqrt2(), False)
            else:
                xp = xp - zsqrt2.One
                yp = yp - zsqrt2.One
                return zw.ZW.from_pair(xp.divide_by_sqrt2(), yp.divide_by_sqrt2(), True)


####### The operators {R, K, K^\bullet, A, B, X, Z, I, Sigma} * sqrt(2) ############


RSqrt2 = GridOperator([[zsqrt2.One, -zsqrt2.One], [zsqrt2.One, zsqrt2.One]])

KSqrt2 = GridOperator([[-zsqrt2.LAMBDA_INV, -zsqrt2.One], [zsqrt2.LAMBDA, zsqrt2.One]])

KConjSqrt2 = GridOperator(
    [[-zsqrt2.LAMBDA_INV.conj(), -zsqrt2.One], [zsqrt2.LAMBDA.conj(), zsqrt2.One]]
)

ASqrt2 = GridOperator(np.array([[zsqrt2.SQRT_2, -2 * zsqrt2.SQRT_2], [zsqrt2.Zero, zsqrt2.SQRT_2]]))

BSqrt2 = GridOperator(np.array([[zsqrt2.SQRT_2, 2 * zsqrt2.One], [zsqrt2.Zero, zsqrt2.SQRT_2]]))

XSqrt2 = GridOperator(np.array([[zsqrt2.Zero, zsqrt2.SQRT_2], [zsqrt2.SQRT_2, zsqrt2.Zero]]))
ZSqrt2 = GridOperator(np.array([[zsqrt2.SQRT_2, zsqrt2.Zero], [zsqrt2.Zero, -zsqrt2.SQRT_2]]))

ISqrt2 = GridOperator(np.array([[zsqrt2.SQRT_2, zsqrt2.Zero], [zsqrt2.Zero, zsqrt2.SQRT_2]]))

HALF_SIGMA_Sqrt2 = GridOperator(
    zsqrt2.SQRT_2 * np.array([[zsqrt2.LAMBDA, zsqrt2.Zero], [zsqrt2.Zero, zsqrt2.One]])
)

HALF_SIGMA_INV_Sqrt2 = GridOperator(
    zsqrt2.SQRT_2 * np.array([[zsqrt2.LAMBDA_INV, zsqrt2.Zero], [zsqrt2.Zero, zsqrt2.One]])
)
