#  Copyright 2023 Google LLC
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

import abc
from typing import TypeVar, Union

import attrs
import sympy


class QDType:
    @property
    @abc.abstractmethod
    def num_qubits(self):
        """Number of qubits required to represent a single instance of this data type."""


QDTypeT = TypeVar("QDTypeT", bound=QDType)


@attrs.frozen
class QBit(QDType):
    """A single qubit. The smallest addressable unit of quantum data."""

    @property
    def num_qubits(self):
        return 1


@attrs.frozen
class QAny(QDType):
    """Opaque bag-of-qbits type. Should be used sparingly"""

    bitsize: Union[int, sympy.Expr]

    def __attrs_post_init__(self):
        if self.bitsize == 1:
            raise ValueError("bitsize must be greater than 1. Use QBit() instead.")

    @property
    def num_qubits(self):
        return self.bitsize


@attrs.frozen
class QInt(QDType):
    """Integer of a given width bitsize.

    Attributes:
        bitsize: The number of qubits used to represent the integer.
    """

    bitsize: Union[int, sympy.Expr]

    def __attrs_post_init__(self):
        if self.bitsize == 1:
            raise ValueError("bitsize must be greater than 1. Use QBit() instead.")

    @property
    def num_qubits(self):
        return self.bitsize


@attrs.frozen
class QUnsignedInt(QDType):
    """Integer of a given width bitsize which wraps around upon overflow.

    Similar to unsigned integer types in C.

    Attributes:
        bitsize: The number of qubits used to represent the integer.
    """

    bitsize: int

    def __attrs_post_init__(self):
        if self.bitsize == 1:
            raise ValueError("bitsize must be greater than 1. Use QBit() instead.")

    @property
    def num_qubits(self):
        return self.bitsize


@attrs.frozen
class BoundedQInt(QDType):
    """Integer whose values are bounded within a range.

    Attributes:
        bitsize: The number of qubits used to represent the integer.
        iteration_range: The
    """

    bitsize: Union[int, sympy.Expr]
    iteration_range: range

    def __attrs_post_init__(self):
        if self.iteration_range.start > self.iteration_range.stop:
            raise ValueError("iteration_range limits should be increasing in value.")

        if len(self.iteration_range) > 2**self.bitsize:
            raise ValueError(
                f"BoundedQInt iteration length is too large for given bitsize. {len(self.iteration_range)} vs {2**self.bitsize}"
            )
        if self.bitsize == 1:
            raise ValueError("bitsize must be greater than 1. Use QBit() instead.")

    @property
    def num_qubits(self):
        return self.bitsize


@attrs.frozen
class QFixedPoint(QDType):
    """Fixed point type to represent real numbers.

    The integer part of the float and the fractional part (the part after the
    decimal plase) can be chosen to have different bitsizes.

    [Question: Sign bit]

    Attributes:
        int_bitsize: The number of qubits used to represent the integer part of the float.
        frac_bitsize: The number of qubits used to represent the fractional part of the float.
    """

    int_bitsize: Union[int, sympy.Expr]
    frac_bitsize: Union[int, sympy.Expr]

    def __attrs_post_init__(self):
        if self.num_qubits == 1:
            raise ValueError("num_qubits must be greater than 1. Use QBit() instead.")

    @property
    def num_qubits(self):
        return self.int_bitsize + self.frac_bitsize
