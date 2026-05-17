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
from typing import Iterable, List, Optional, Sequence

import attrs
import numpy as np
from numpy.typing import NDArray

from qualtran.symbolics import is_symbolic, SymbolicInt

from ._base import BitEncoding, CDType, QDType


@attrs.frozen
class _MontgomeryUInt(BitEncoding[int]):
    r"""Montgomery form of an unsigned integer of a given width bitsize which wraps around upon
        overflow.

    Any MontgomeryUInt can be treated as a UInt, but not
    every UInt can be treated as a MontgomeryUInt. Montgomery form is used in order to compute
    fast modular multiplication.
    """

    bitsize: SymbolicInt
    modulus: Optional[SymbolicInt] = None

    def get_domain(self) -> Iterable[int]:
        if self.modulus is None or is_symbolic(self.modulus):
            return range(2**self.bitsize)
        return range(1, int(self.modulus))

    def to_bits(self, x: int) -> List[int]:
        self.assert_valid_val(x)
        return [int(x) for x in f'{int(x):0{self.bitsize}b}']

    def from_bits(self, bits: Sequence[int]) -> int:
        return int("".join(str(x) for x in bits), 2)

    def assert_valid_val(self, val: int, debug_str: str = 'val') -> None:
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} should be an integer, not {val!r}")
        if val < 0:
            raise ValueError(f"Negative classical value encountered in {debug_str}")
        if val >= 2**self.bitsize:
            raise ValueError(f"Too-large classical value encountered in {debug_str}")

    def assert_valid_val_array(
        self, val_array: NDArray[np.integer], debug_str: str = 'val'
    ) -> None:
        if np.any(val_array < 0):
            raise ValueError(f"Negative classical values encountered in {debug_str}")
        if np.any(val_array >= 2**self.bitsize):
            raise ValueError(f"Too-large classical values encountered in {debug_str}")

    def montgomery_inverse(self, xm: int) -> int:
        """Returns the modular inverse of an integer in montgomery form.

        Args:
            xm: An integer in montgomery form.
        """
        assert self.modulus is not None and not is_symbolic(self.modulus)
        return (
            (pow(xm, -1, self.modulus)) * pow(2, 2 * self.bitsize, int(self.modulus))
        ) % self.modulus

    def montgomery_product(self, xm: int, ym: int) -> int:
        """Returns the modular product of two integers in montgomery form.

        Args:
            xm: The first montgomery form integer for the product.
            ym: The second montgomery form integer for the product.
        """
        assert self.modulus is not None and not is_symbolic(self.modulus)
        return (xm * ym * pow(2, -int(self.bitsize), int(self.modulus))) % self.modulus

    def montgomery_to_uint(self, xm: int) -> int:
        """Converts an integer in montgomery form to a normal form integer.

        Args:
            xm: An integer in montgomery form.
        """
        assert self.modulus is not None and not is_symbolic(self.modulus)
        return (xm * pow(2, -int(self.bitsize), int(self.modulus))) % self.modulus

    def uint_to_montgomery(self, x: int) -> int:
        """Converts an integer into montgomery form.

        Args:
            x: An integer.
        """
        assert self.modulus is not None and not is_symbolic(self.modulus)
        return (x * pow(2, int(self.bitsize), int(self.modulus))) % self.modulus


@attrs.frozen
class QMontgomeryUInt(QDType[int]):
    r"""Montgomery form of an unsigned integer of a given width bitsize which wraps around upon
        overflow.

    Similar to unsigned integer types in C. Any intended wrap around effect is
    expected to be handled by the developer. Any QMontgomeryUInt can be treated as a QUInt, but not
    every QUInt can be treated as a QMontgomeryUInt. Montgomery form is used in order to compute
    fast modular multiplication.

    In order to convert an unsigned integer from a finite field x % p into Montgomery form you
    first must choose a value r > p where gcd(r, p) = 1. Typically, this value is a power of 2.

    Conversion to Montgomery form is given by
    `[x] = (x * r) % p`

    Conversion from Montgomery form to normal form is given by
    `x = REDC([x])`

    Pseudocode for REDC(u) can be found in the resource below.

    Args:
        bitsize: The number of qubits used to represent the integer.
        modulus: The modulus p.

    References:
        [Montgomery modular multiplication](https://en.wikipedia.org/wiki/Montgomery_modular_multiplication).

        [Performance Analysis of a Repetition Cat Code Architecture: Computing 256-bit Elliptic Curve Logarithm in 9 Hours with 126133 Cat Qubits](https://arxiv.org/abs/2302.06639).
        Gouzien et al. 2023.
        We follow Montgomery form as described in the above paper; namely, r = 2^bitsize.
    """

    bitsize: SymbolicInt
    modulus: Optional[SymbolicInt] = None

    @cached_property
    def _bit_encoding(self) -> _MontgomeryUInt:
        return _MontgomeryUInt(self.bitsize, self.modulus)

    def montgomery_inverse(self, xm: int) -> int:
        """Returns the modular inverse of an integer in montgomery form.

        Args:
            xm: An integer in montgomery form.
        """
        return self._bit_encoding.montgomery_inverse(xm)

    def montgomery_product(self, xm: int, ym: int) -> int:
        """Returns the modular product of two integers in montgomery form.

        Args:
            xm: The first montgomery form integer for the product.
            ym: The second montgomery form integer for the product.
        """
        return self._bit_encoding.montgomery_product(xm, ym)

    def montgomery_to_uint(self, xm: int) -> int:
        """Converts an integer in montgomery form to a normal form integer.

        Args:
            xm: An integer in montgomery form.
        """
        return self._bit_encoding.montgomery_to_uint(xm)

    def uint_to_montgomery(self, x: int) -> int:
        """Converts an integer into montgomery form.

        Args:
            x: An integer.
        """
        return self._bit_encoding.uint_to_montgomery(x)

    def __str__(self):
        if self.modulus is not None:
            modstr = f', {self.modulus}'
        else:
            modstr = ''
        return f'{self.__class__.__name__}({self.bitsize}{modstr})'


@attrs.frozen
class CMontgomeryUInt(CDType[int]):
    r"""Montgomery form of an unsigned integer of a given width bitsize which wraps around upon
        overflow.

    This is a classical version of QMontgomeryUInt. See the documentation for that class.
    """

    bitsize: SymbolicInt
    modulus: Optional[SymbolicInt] = None

    @cached_property
    def _bit_encoding(self) -> _MontgomeryUInt:
        return _MontgomeryUInt(self.bitsize, self.modulus)

    def __str__(self):
        if self.modulus is not None:
            modstr = f', {self.modulus}'
        else:
            modstr = ''
        return f'{self.__class__.__name__}({self.bitsize}{modstr})'
