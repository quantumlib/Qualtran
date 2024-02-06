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
"""Quantum data type definitions.

We often wish to write algorithms which operate on quantum data. One can think
of quantum data types, similar to classical data types, where a collection of
qubits can be used to represent a specific quantum data type (eg: a quantum
integer of width 32 would comprise of 32 qubits, similar to a classical uint32
type). More generally, many current primitives and algorithms in qualtran
implicitly expect registers which represent signed or unsigned integers,
fixed-point (fp) numbers , or “classical registers” which store some classical
value. Enforcing typing helps developers and users reason about algorithms, and
will also allow better type checking.

The basic principles we follow are:

1. Typing should not be too invasive for the developer / user: We got pretty far
without explicitly typing registers.
2. For algorithms or bloqs which expect registers which are meant to encode
numeric types (integers, reals, etc.) then typing should be strictly enforced.
For example, a bloq multiplying two fixed point reals should be built with an
explicit QFixedPoint dtype.
3. The smallest addressable unit is a QBit. Other types are interpretations of
collections of QBits. A QUnsigedInt(32) is intended to represent a register
encoding positive integers.
4. To avoid too much overhead we have a QAny type, which is meant to represent
an opaque bag of bits with no particular significance associated with them. A
bloq defined with a QAny register (e.g. a n-bit CSwap) will accept any other
type assuming the bitsizes match. QInt(32) == QAny(32), QInt(32) !=
QFixedPoint(16, 16). QInt(32) != QUInt(32).
5. We assume a big endian convention for addressing QBits in registers
throughout qualtran. Recall that in a big endian convention the most signficant
bit is at index 0. If you iterate through the bits in a register they will be
yielded from most significant to least significant.
6. Ones' complement integers are used extensively in quantum algorithms. We have
two types QInt and QIntOnesComp for integers using two's and ones' complement
respectively.
"""

import abc
from typing import Union

import attrs
import sympy


class QDType:
    """This defines the abstract interface for quantum data types."""

    @property
    @abc.abstractmethod
    def num_qubits(self):
        """Number of qubits required to represent a single instance of this data type."""


@attrs.frozen
class QBit(QDType):
    """A single qubit. The smallest addressable unit of quantum data."""

    @property
    def num_qubits(self):
        return 1


@attrs.frozen
class QAny(QDType):
    """Opaque bag-of-qbits type."""

    bitsize: Union[int, sympy.Expr]

    def __attrs_post_init__(self):
        if isinstance(self.bitsize, int):
            if self.num_qubits == 1:
                raise ValueError("num_qubits must be > 1.")

    @property
    def num_qubits(self):
        return self.bitsize


@attrs.frozen
class QInt(QDType):
    """Signed Integer of a given width bitsize.

    A two's complement representation is assumed for negative integers.

    Attributes:
        bitsize: The number of qubits used to represent the integer.
    """

    bitsize: Union[int, sympy.Expr]

    def __attrs_post_init__(self):
        if isinstance(self.bitsize, int):
            if self.num_qubits == 1:
                raise ValueError("num_qubits must be > 1.")

    @property
    def num_qubits(self):
        return self.bitsize


@attrs.frozen
class QIntOnesComp(QDType):
    """Signed Integer of a given width bitsize.

    A ones' complement representation is assumed for negative integers.

    Attributes:
        bitsize: The number of qubits used to represent the integer.
    """

    bitsize: Union[int, sympy.Expr]

    def __attrs_post_init__(self):
        if isinstance(self.bitsize, int):
            if self.num_qubits == 1:
                raise ValueError("num_qubits must be > 1.")

    @property
    def num_qubits(self):
        return self.bitsize


@attrs.frozen
class QUInt(QDType):
    """Unsigned integer of a given width bitsize which wraps around upon overflow.

    Similar to unsigned integer types in C. Any intended wrap around effect is
    expected to be handled by the developer.

    Attributes:
        bitsize: The number of qubits used to represent the integer.
    """

    bitsize: Union[int, sympy.Expr]

    def __attrs_post_init__(self):
        if isinstance(self.bitsize, int):
            if self.num_qubits == 1:
                raise ValueError("num_qubits must be > 1.")

    @property
    def num_qubits(self):
        return self.bitsize


@attrs.frozen
class BoundedQInt(QDType):
    """Integer whose values are bounded within a range.

    Attributes:
        bitsize: The number of qubits used to represent the integer.
        iteration_length: The length of the iteration range.
    """

    bitsize: Union[int, sympy.Expr]
    iteration_length: Union[int, sympy.Expr]

    def __attrs_post_init__(self):
        if isinstance(self.bitsize, int):
            if self.num_qubits == 1:
                raise ValueError("num_qubits must be > 1.")
            if self.iteration_length > 2**self.bitsize:
                raise ValueError(
                    "BoundedQInt iteration length is too large for given bitsize. "
                    f"{self.iteration_length} vs {2**self.bitsize}"
                )

    @property
    def num_qubits(self):
        return self.bitsize


@attrs.frozen
class QFixedPoint(QDType):
    """Fixed point type to represent real numbers.

    The integer part of the float and the fractional part (the part after the
    decimal plase) can be chosen to have different bitsizes. An additional sign bit is assumed.

    Attributes:
        int_bitsize: The number of qubits used to represent the integer part of the float.
        frac_bitsize: The number of qubits used to represent the fractional part of the float.
    """

    int_bitsize: Union[int, sympy.Expr]
    frac_bitsize: Union[int, sympy.Expr]

    @property
    def num_qubits(self):
        if isinstance(self.int_bitsize, sympy.Expr):
            return self.int_bitsize + self.frac_bitsize + sympy.symbols("1")
        else:
            return self.int_bitsize + self.frac_bitsize + 1

    def __attrs_post_init__(self):
        if isinstance(self.num_qubits, int):
            if self.num_qubits == 1:
                raise ValueError("num_qubits must be > 1.")
