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

from functools import cached_property
from typing import Any, Iterable, List, Optional, Sequence, TYPE_CHECKING, Union

import attrs
import numpy as np
from numpy.typing import NDArray

from qualtran.symbolics import bit_length, is_symbolic, SymbolicInt

from .._base import BitEncoding, CDType, QDType

if TYPE_CHECKING:
    import galois


def _poly_converter(p) -> Union['galois.Poly', None]:
    import galois

    if p is None:
        return None
    if isinstance(p, galois.Poly):
        return p
    return galois.Poly.Degrees(p)


@attrs.frozen
class _GF(BitEncoding['galois.FieldArray']):
    r"""Galois Field type to represent elements of a finite field."""

    characteristic: SymbolicInt
    degree: SymbolicInt
    irreducible_poly: Optional['galois.Poly'] = attrs.field(converter=_poly_converter)

    @irreducible_poly.default
    def _irreducible_poly_default(self):
        if is_symbolic(self.characteristic, self.degree):
            return None

        from galois import GF

        return GF(  # type: ignore[call-overload]
            int(self.characteristic), int(self.degree), compile='python-calculate'
        ).irreducible_poly

    @cached_property
    def order(self) -> SymbolicInt:
        return self.characteristic**self.degree

    @cached_property
    def bitsize(self) -> SymbolicInt:
        """Bitsize of qubit register required to represent a single instance of this data type."""
        return bit_length(self.order - 1)

    def get_domain(self) -> Iterable[Any]:
        yield from self.gf_type.elements

    @cached_property
    def _uint_encoder(self) -> BitEncoding[int]:
        from qualtran.dtype._uint import _UInt

        return _UInt(self.bitsize)

    @cached_property
    def gf_type(self):
        from galois import GF

        poly = self.irreducible_poly if self.degree > 1 else None

        return GF(  # type: ignore[call-overload]
            int(self.characteristic),
            int(self.degree),
            irreducible_poly=poly,
            verify=False,
            repr='poly',
            compile='python-calculate',
        )

    def to_bits(self, x) -> List[int]:
        self.assert_valid_val(x)
        return self._uint_encoder.to_bits(int(x))

    def from_bits(self, bits: Sequence[int]):
        return self.gf_type(self._uint_encoder.from_bits(bits))

    def from_bits_array(self, bits_array: NDArray[np.uint8]):
        return self.gf_type(self._uint_encoder.from_bits_array(bits_array))

    def assert_valid_val(self, val: Any, debug_str: str = 'val'):
        if not isinstance(val, self.gf_type):
            raise ValueError(f"{debug_str} should be a {self.gf_type}, not {val!r}")

    def assert_valid_val_array(self, val_array: NDArray[Any], debug_str: str = 'val'):
        if np.any(val_array < 0):
            raise ValueError(f"Negative classical values encountered in {debug_str}")
        if np.any(val_array >= self.order):
            raise ValueError(f"Too-large classical values encountered in {debug_str}")


@attrs.frozen
class QGF(QDType['galois.FieldArray']):
    r"""Galois Field type to represent elements of a finite field.

    A Finite Field or Galois Field is a field that contains finite number of elements. The order
    of a finite field is the number of elements in the field, which is either a prime number or
    a prime power. For every prime number $p$ and every positive integer $m$ there are fields of
    order $p^m$, all of which are isomorphic. When m=1, the finite field of order p can be
    constructed via integers modulo p.

    Elements of a Galois Field $GF(p^m)$ may be conveniently viewed as polynomials
    $a_{0} + a_{1}x + ... + a_{m−1}x_{m−1}$, where $a_0, a_1, ..., a_{m−1} \in F(p)$.
    $GF(p^m)$ addition is defined as the component-wise (polynomial) addition over F(p) and
    multiplication is defined as polynomial multiplication modulo an irreducible polynomial of
    degree $m$. The selection of the specific irreducible polynomial affects the representation
    of the given field, but all fields of a fixed size are isomorphic.

    The data type uses the [Galois library](https://mhostetter.github.io/galois/latest/) to
    perform arithmetic over Galois Fields. By default, the Conway polynomial $C_{p, m}$ is used
    as the irreducible polynomial.

    Args:
        characteristic: The characteristic $p$ of the field $GF(p^m)$.
            The characteristic must be prime.
        degree: The degree $m$ of the field $GF(p^{m})$. The degree must be a positive integer.
        irreducible_poly: Optional `galois.Poly` instance that defines the field arithmetic.
            This parameter is passed to `galois.GF(..., irreducible_poly=irreducible_poly, verify=False)`.

    References:
        [Finite Field](https://en.wikipedia.org/wiki/Finite_field)

        [Intro to Prime Fields](https://mhostetter.github.io/galois/latest/tutorials/intro-to-prime-fields/)

        [Intro to Extension Fields](https://mhostetter.github.io/galois/latest/tutorials/intro-to-extension-fields/)
    """

    characteristic: SymbolicInt
    degree: SymbolicInt
    irreducible_poly: Optional['galois.Poly'] = attrs.field(converter=_poly_converter)

    @irreducible_poly.default
    def _irreducible_poly_default(self):
        if is_symbolic(self.characteristic, self.degree):
            return None

        from galois import GF

        return GF(  # type: ignore[call-overload]
            int(self.characteristic), int(self.degree), compile='python-calculate'
        ).irreducible_poly

    @cached_property
    def _bit_encoding(self) -> _GF:
        return _GF(
            characteristic=self.characteristic,
            degree=self.degree,
            irreducible_poly=self.irreducible_poly,
        )

    @property
    def order(self) -> SymbolicInt:
        return self._bit_encoding.order

    @property
    def bitsize(self) -> SymbolicInt:
        """Bitsize of qubit register required to represent a single instance of this data type."""
        return self._bit_encoding.bitsize

    @property
    def gf_type(self):
        return self._bit_encoding.gf_type

    def is_symbolic(self) -> bool:
        return is_symbolic(self.characteristic, self.order)

    def iteration_length_or_zero(self) -> SymbolicInt:
        return self.order

    def __str__(self):
        return f'QGF({self.characteristic}**{self.degree})'


@attrs.frozen
class CGF(CDType['galois.FieldArray']):
    r"""Galois Field classical type to represent elements of a finite field.

    See QGF for documentation.
    """

    characteristic: SymbolicInt
    degree: SymbolicInt
    irreducible_poly: Optional['galois.Poly'] = attrs.field(converter=_poly_converter)

    @irreducible_poly.default
    def _irreducible_poly_default(self):
        if is_symbolic(self.characteristic, self.degree):
            return None

        from galois import GF

        return GF(  # type: ignore[call-overload]
            int(self.characteristic), int(self.degree), compile='python-calculate'
        ).irreducible_poly

    @cached_property
    def _bit_encoding(self) -> _GF:
        return _GF(
            characteristic=self.characteristic,
            degree=self.degree,
            irreducible_poly=self.irreducible_poly,
        )

    @property
    def order(self) -> SymbolicInt:
        return self._bit_encoding.order

    @property
    def bitsize(self) -> SymbolicInt:
        """Bitsize of qubit register required to represent a single instance of this data type."""
        return self._bit_encoding.bitsize

    @property
    def gf_type(self):
        return self._bit_encoding.gf_type

    def is_symbolic(self) -> bool:
        return is_symbolic(self.characteristic, self.order)

    def iteration_length_or_zero(self) -> SymbolicInt:
        return self.order

    def __str__(self):
        return f'CGF({self.characteristic}**{self.degree})'
