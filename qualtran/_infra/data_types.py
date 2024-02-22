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
explicit QFxp dtype.
3. The smallest addressable unit is a QBit. Other types are interpretations of
collections of QBits. A QUInt(32) is intended to represent a register
encoding positive integers.
4. To avoid too much overhead we have a QAny type, which is meant to represent
an opaque bag of bits with no particular significance associated with them. A
bloq defined with a QAny register (e.g. a n-bit CSwap) will accept any other
type assuming the bitsizes match. QInt(32) == QAny(32), QInt(32) !=
QFxp(32, 16). QInt(32) != QUInt(32).
5. We assume a big endian convention for addressing QBits in registers
throughout qualtran. Recall that in a big endian convention the most signficant
bit is at index 0. If you iterate through the bits in a register they will be
yielded from most significant to least significant.
6. Ones' complement integers are used extensively in quantum algorithms. We have
two types QInt and QIntOnesComp for integers using two's and ones' complement
respectively.
"""

import abc
from typing import Any, Iterable, Union

import attrs
import numpy as np
import sympy
from numpy.typing import NDArray


class QDType(metaclass=abc.ABCMeta):
    """This defines the abstract interface for quantum data types."""

    @property
    @abc.abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits required to represent a single instance of this data type."""

    @abc.abstractmethod
    def get_classical_domain(self) -> Iterable[Any]:
        """Yields all possible classical (computational basis state) values representable
        by this type."""

    @abc.abstractmethod
    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        """Raises an exception if `val` is not a valid classical value for this type.

        Args:
            val: A classical value that should be in the domain of this QDType.
            debug_str: Optional debugging information to use in exception messages.
        """

    def assert_valid_classical_val_array(self, val_array: NDArray[Any], debug_str: str = 'val'):
        """Raises an exception if `val_array` is not a valid array of classical values
        for this type.

        Often, validation on an array can be performed faster than validating each element
        individually.

        Args:
            val_array: A numpy array of classical values. Each value should be in the domain
                of this QDType.
            debug_str: Optional debugging information to use in exception messages.
        """
        for val in val_array.reshape(-1):
            self.assert_valid_classical_val(val)


@attrs.frozen
class QBit(QDType):
    """A single qubit. The smallest addressable unit of quantum data."""

    @property
    def num_qubits(self):
        return 1

    def get_classical_domain(self) -> Iterable[int]:
        yield from (0, 1)

    def assert_valid_classical_val(self, val: int, debug_str: str = 'val'):
        if not (val == 0 or val == 1):
            raise ValueError(f"Bad {self} value {val} in {debug_str}")

    def assert_valid_classical_val_array(self, val_array: NDArray[int], debug_str: str = 'val'):
        if not np.all((val_array == 0) | (val_array == 1)):
            raise ValueError(f"Bad {self} value array in {debug_str}")


@attrs.frozen
class QAny(QDType):
    """Opaque bag-of-qbits type."""

    bitsize: Union[int, sympy.Expr]

    @property
    def num_qubits(self):
        return self.bitsize

    def get_classical_domain(self) -> Iterable[Any]:
        raise TypeError(f"Ambiguous domain for {self}. Please use a more specific type.")

    def assert_valid_classical_val(self, val, debug_str: str = 'val'):
        pass

    def assert_valid_classical_val_array(self, val_array, debug_str: str = 'val'):
        pass


@attrs.frozen
class QInt(QDType):
    """Signed Integer of a given width bitsize.

    A two's complement representation is assumed for negative integers.

    Attributes:
        bitsize: The number of qubits used to represent the integer.
    """

    bitsize: Union[int, sympy.Expr]

    @property
    def num_qubits(self):
        return self.bitsize

    def get_classical_domain(self) -> Iterable[int]:
        return range(-(2 ** (self.bitsize - 1)), 2 ** (self.bitsize - 1))

    def assert_valid_classical_val(self, val: int, debug_str: str = 'val'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} should be an integer, not {val!r}")
        if val < -(2 ** (self.bitsize - 1)):
            raise ValueError(f"Too-small classical {self}: {val} encountered in {debug_str}")
        if val >= 2 ** (self.bitsize - 1):
            raise ValueError(f"Too-large classical {self}: {val} encountered in {debug_str}")

    def assert_valid_classical_val_array(self, val_array: NDArray[int], debug_str: str = 'val'):
        if np.any(val_array < -(2 ** (self.bitsize - 1))):
            raise ValueError(f"Too-small classical {self}s encountered in {debug_str}")
        if np.any(val_array >= 2 ** (self.bitsize - 1)):
            raise ValueError(f"Too-large classical {self}s encountered in {debug_str}")


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

    def get_classical_domain(self) -> Iterable[Any]:
        raise NotImplementedError()

    def assert_valid_classical_val(self, val, debug_str: str = 'val'):
        pass  # TODO: implement


@attrs.frozen
class QUInt(QDType):
    """Unsigned integer of a given width bitsize which wraps around upon overflow.

    Similar to unsigned integer types in C. Any intended wrap around effect is
    expected to be handled by the developer.

    Attributes:
        bitsize: The number of qubits used to represent the integer.
    """

    bitsize: Union[int, sympy.Expr]

    @property
    def num_qubits(self):
        return self.bitsize

    def get_classical_domain(self) -> Iterable[Any]:
        return range(2 ** (self.bitsize))

    def assert_valid_classical_val(self, val: int, debug_str: str = 'val'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} should be an integer, not {val!r}")
        if val < 0:
            raise ValueError(f"Negative classical value encountered in {debug_str}")
        if val >= 2**self.bitsize:
            raise ValueError(f"Too-large classical value encountered in {debug_str}")

    def assert_valid_classical_val_array(self, val_array: NDArray[int], debug_str: str = 'val'):
        if np.any(val_array < 0):
            raise ValueError(f"Negative classical values encountered in {debug_str}")
        if np.any(val_array >= 2**self.bitsize):
            raise ValueError(f"Too-large classical values encountered in {debug_str}")


@attrs.frozen
class BoundedQUInt(QDType):
    """Unsigned integer whose values are bounded within a range.

    LCU methods often make use of coherent for-loops via UnaryIteration, iterating over a range
    of values stored as a superposition over the `SELECT` register. Such (nested) coherent
    for-loops can be represented using a `Tuple[Register(dtype=BoundedQUInt),
    ...]` where the i'th entry stores the bitsize and iteration length of i'th
    nested for-loop.

    One useful feature when processing such nested for-loops is to flatten out a composite index,
    represented by a tuple of indices (i, j, ...), one for each selection register into a single
    integer that can be used to index a flat target register. An example of such a mapping
    function is described in Eq.45 of https://arxiv.org/abs/1805.03662. A general version of this
    mapping function can be implemented using `numpy.ravel_multi_index` and `numpy.unravel_index`.

    For example:
        1) We can flatten a 2D for-loop as follows
        >>> import numpy as np
        >>> N, M = 10, 20
        >>> flat_indices = set()
        >>> for x in range(N):
        ...     for y in range(M):
        ...         flat_idx = x * M + y
        ...         assert np.ravel_multi_index((x, y), (N, M)) == flat_idx
        ...         assert np.unravel_index(flat_idx, (N, M)) == (x, y)
        ...         flat_indices.add(flat_idx)
        >>> assert len(flat_indices) == N * M

        2) Similarly, we can flatten a 3D for-loop as follows
        >>> import numpy as np
        >>> N, M, L = 10, 20, 30
        >>> flat_indices = set()
        >>> for x in range(N):
        ...     for y in range(M):
        ...         for z in range(L):
        ...             flat_idx = x * M * L + y * L + z
        ...             assert np.ravel_multi_index((x, y, z), (N, M, L)) == flat_idx
        ...             assert np.unravel_index(flat_idx, (N, M, L)) == (x, y, z)
        ...             flat_indices.add(flat_idx)
        >>> assert len(flat_indices) == N * M * L

    Attributes:
        bitsize: The number of qubits used to represent the integer.
        iteration_length: The length of the iteration range.
    """

    bitsize: Union[int, sympy.Expr]
    iteration_length: Union[int, sympy.Expr] = attrs.field()

    def __attrs_post_init__(self):
        if isinstance(self.bitsize, int):
            if self.iteration_length > 2**self.bitsize:
                raise ValueError(
                    "BoundedQUInt iteration length is too large for given bitsize. "
                    f"{self.iteration_length} vs {2**self.bitsize}"
                )

    @iteration_length.default
    def _default_iteration_length(self):
        return 2**self.bitsize

    @property
    def num_qubits(self):
        return self.bitsize

    def get_classical_domain(self) -> Iterable[Any]:
        return range(0, self.iteration_length)

    def assert_valid_classical_val(self, val: int, debug_str: str = 'val'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} should be an integer, not {val!r}")
        if val < 0:
            raise ValueError(f"Negative classical value encountered in {debug_str}")
        if val >= self.iteration_length:
            raise ValueError(f"Too-large classical value encountered in {debug_str}")

    def assert_valid_classical_val_array(self, val_array: NDArray[int], debug_str: str = 'val'):
        if np.any(val_array < 0):
            raise ValueError(f"Negative classical values encountered in {debug_str}")
        if np.any(val_array >= self.iteration_length):
            raise ValueError(f"Too-large classical values encountered in {debug_str}")


@attrs.frozen
class QFxp(QDType):
    r"""Fixed point type to represent real numbers.

    A real number can be approximately represented in fixed point using `num_int`
    bits for the integer part and `num_frac` bits for the fractional part. If the
    real number is signed we require an additional bit to store the sign (0 for
    +, 1 for -). In total there are `bitsize = (n_sign + num_int + num_frac)` bits used
    to represent the number. E.g. Using `(bitsize = 8, num_frac = 6, signed = False)`
    then $\pi$ \approx 3.140625 = 11.001001, where the . represents the decimal place.

    We can specify a fixed point real number by the tuple bitsize, num_frac and
    signed, with num_int determined as `(bitsize - num_frac - n_sign)`.

    Attributes:
        bitsize: The total number of qubits used to represent the integer and
            fractional part combined.
        num_frac: The number of qubits used to represent the fractional part of the real number.
        signed: Whether the number is signed or not. If signed is true the
            number of integer bits is reduced by 1.
    """

    bitsize: Union[int, sympy.Expr]
    num_frac: Union[int, sympy.Expr]
    signed: bool = False

    @property
    def num_qubits(self):
        return self.bitsize

    @property
    def num_int(self) -> Union[int, sympy.Expr]:
        return self.bitsize - self.num_frac - int(self.signed)

    @property
    def fxp_dtype_str(self) -> str:
        return f'fxp-{"us"[self.signed]}{self.bitsize}/{self.num_frac}'

    def __attrs_post_init__(self):
        if isinstance(self.num_qubits, int):
            if self.num_qubits == 1 and self.signed:
                raise ValueError("num_qubits must be > 1.")
            if self.signed and self.bitsize == self.num_frac:
                raise ValueError("num_frac must be less than bitsize if the QFxp is signed.")
            if self.bitsize < self.num_frac:
                raise ValueError("bitsize must be >= num_frac.")

    def get_classical_domain(self) -> Iterable[Any]:
        raise NotImplementedError()

    def assert_valid_classical_val(self, val, debug_str: str = 'val'):
        pass  # TODO: implement


@attrs.frozen
class QMontgomeryUInt(QDType):
    """Montgomery form of an unsigned integer of a given width bitsize which wraps around upon
        overflow.

    Similar to unsigned integer types in C. Any intended wrap around effect is
    expected to be handled by the developer. Any QMontgomeryUInt can be treated as a QUInt, but not
    every QUInt can be treated as a QMontgomeryUInt. Montgomery form is used in order to compute
    fast modular multiplication.

    In order to convert an unsigned integer from a finite field x % p into Montgomery form you
    first must choose a value r > p where gcd(r, p) = 1. Typically this value is a power of 2.

    Conversion to Montgomery form:
        [x] = (x * r) % p

    Conversion from Montgomery form to normal form:
        x = REDC([x])

    Pseudocode for REDC(u) can be found in the resource below.

    Attributes:
        bitsize: The number of qubits used to represent the integer.

    References:
        [Montgomery modular multiplication](https://en.wikipedia.org/wiki/Montgomery_modular_multiplication)
    """

    bitsize: Union[int, sympy.Expr]

    @property
    def num_qubits(self):
        return self.bitsize

    def get_classical_domain(self) -> Iterable[Any]:
        return range(2 ** (self.bitsize))

    def assert_valid_classical_val(self, val: int, debug_str: str = 'val'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} should be an integer, not {val!r}")
        if val < 0:
            raise ValueError(f"Negative classical value encountered in {debug_str}")
        if val >= 2**self.bitsize:
            raise ValueError(f"Too-large classical value encountered in {debug_str}")

    def assert_valid_classical_val_array(self, val_array: NDArray[int], debug_str: str = 'val'):
        if np.any(val_array < 0):
            raise ValueError(f"Negative classical values encountered in {debug_str}")
        if np.any(val_array >= 2**self.bitsize):
            raise ValueError(f"Too-large classical values encountered in {debug_str}")
