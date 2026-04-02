#  Copyright 2026 Google LLC
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

import itertools
from functools import cached_property
from typing import Any, Iterable, List, Sequence, TYPE_CHECKING

import attrs
import numpy as np

from qualtran.symbolics import is_symbolic, SymbolicInt

from .._base import BitEncoding, CDType, QDType
from ._gf import _GF, QGF

if TYPE_CHECKING:
    import galois

    import qualtran.dtype as qdt


@attrs.frozen
class _GFPoly(BitEncoding):
    r"""Univariate Polynomials with coefficients in a Galois Field GF($p^m$).

    Args:
        degree: The degree $n$ of the univariate polynomial $f(x)$ represented by this type.
        gf: An instance of `_GF` that represents the galois field $GF(p^m)$ over which the
            univariate polynomial $f(x)$ is defined.

    """

    degree: SymbolicInt
    gf: _GF

    @cached_property
    def bitsize(self) -> SymbolicInt:
        return self.gf.bitsize * (self.degree + 1)

    def get_domain(self) -> Iterable[Any]:
        """Yields all possible classical (computational basis state) values representable
        by this type."""
        from galois import Poly

        for it in itertools.product(self.gf.gf_type.elements, repeat=(self.degree + 1)):
            yield Poly(self.gf.gf_type(it), field=self.gf.gf_type)

    def to_gf_coefficients(self, f_x: 'galois.Poly') -> 'galois.Array':
        """Returns a big-endian array of coefficients of the polynomial f(x)."""
        f_x_coeffs = self.gf.gf_type.Zeros(self.degree + 1)
        f_x_coeffs[self.degree - f_x.degree :] = f_x.coeffs
        return f_x_coeffs

    def from_gf_coefficients(self, f_x: 'galois.Array') -> 'galois.Poly':
        """Expects a big-endian array of coefficients that represent a polynomial f(x)."""
        import galois

        return galois.Poly(f_x, field=self.gf.gf_type)

    def to_bits(self, x) -> List[int]:
        """Returns individual bits corresponding to binary representation of x"""
        import galois

        self.assert_valid_val(x)
        assert isinstance(x, galois.Poly)
        return self.gf.to_bits_array(self.to_gf_coefficients(x)).reshape(-1).tolist()

    def from_bits(self, bits: Sequence[int]):
        """Combine individual bits to form x"""
        reshaped_bits = np.array(bits).reshape((int(self.degree) + 1, int(self.gf.bitsize)))
        return self.from_gf_coefficients(self.gf.from_bits_array(reshaped_bits))  # type: ignore

    def assert_valid_val(self, val: Any, debug_str: str = 'val'):
        """Raises an exception if `val` is not a valid classical value for this type.

        Args:
            val: A classical value that should be in the domain of this QDType.
            debug_str: Optional debugging information to use in exception messages.
        """
        import galois

        if not isinstance(val, galois.Poly):
            raise ValueError(f"{debug_str} should be a {galois.Poly}, not {val!r}")
        if val.field is not self.gf.gf_type:
            raise ValueError(
                f"{debug_str} should be defined over {self.gf.gf_type}, not {val.field}"
            )
        if val.degree > self.degree:
            raise ValueError(f"{debug_str} should have a degree <= {self.degree}, not {val.degree}")


@attrs.frozen
class QGFPoly(QDType):
    r"""Quantum Univariate Polynomials with coefficients in a Galois Field GF($p^m$).

    This data type represents a degree-$n$ univariate polynomials
    $f(x)=\sum_{i=0}^{n} a_i x^{i}$ where the coefficients $a_{i}$ of the polynomial
    belong to a Galois Field $GF(p^{m})$.

    The data type uses the [Galois library](https://mhostetter.github.io/galois/latest/) to
    perform arithmetic over polynomials defined over Galois Fields using the
    [galois.Poly](https://mhostetter.github.io/galois/latest/api/galois.Poly/).

    Args:
        degree: The degree $n$ of the univariate polynomial $f(x)$ represented by this type.
        qgf: An instance of `QGF` that represents the galois field $GF(p^m)$ over which the
            univariate polynomial $f(x)$ is defined.

    References:
        [Polynomials over finite fields](https://mhostetter.github.io/galois/latest/api/galois.Poly/).
        `galois` documentation.


        [Polynomial Arithmetic](https://mhostetter.github.io/galois/latest/basic-usage/poly-arithmetic/).
        `galois` documentation.
    """

    degree: SymbolicInt
    qgf: QGF

    @cached_property
    def _bit_encoding(self) -> _GFPoly:
        return _GFPoly(self.degree, self.qgf._bit_encoding)

    @property
    def bitsize(self) -> SymbolicInt:
        return self._bit_encoding.bitsize

    def to_gf_coefficients(self, f_x: 'galois.Poly') -> 'galois.Array':
        """Returns a big-endian array of coefficients of the polynomial f(x)."""
        return self._bit_encoding.to_gf_coefficients(f_x)

    def from_gf_coefficients(self, f_x: 'galois.Array') -> 'galois.Poly':
        """Expects a big-endian array of coefficients that represent a polynomial f(x)."""
        return self._bit_encoding.from_gf_coefficients(f_x)

    @cached_property
    def _quint_equivalent(self) -> 'qdt.QUInt':
        from qualtran.dtype import QUInt

        return QUInt(self.num_qubits)

    def is_symbolic(self) -> bool:
        return is_symbolic(self.degree, self.qgf)

    def iteration_length_or_zero(self) -> SymbolicInt:
        return self.qgf.order

    def __str__(self):
        return f'QGFPoly({self.degree}, {self.qgf!s})'


@attrs.frozen
class CGFPoly(CDType):
    r"""Classical Univariate Polynomials with coefficients in a Galois Field GF($p^m$).

    This is a "classical" version of QGFPoly.
    """

    degree: SymbolicInt
    qgf: QGF

    @cached_property
    def _bit_encoding(self) -> _GFPoly:
        return _GFPoly(self.degree, self.qgf._bit_encoding)

    def is_symbolic(self) -> bool:
        return is_symbolic(self.degree, self.qgf)

    def iteration_length_or_zero(self) -> SymbolicInt:
        return self.qgf.order

    def __str__(self):
        return f'CGFPoly({self.degree}, {self.qgf!s})'
