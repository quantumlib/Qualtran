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
from enum import Enum
from typing import Any, Iterable, List, Sequence, Union

import attrs
import numpy as np
from fxpmath import Fxp
from numpy.typing import NDArray

from qualtran.symbolics import is_symbolic, SymbolicInt


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
    def to_bits(self, x) -> List[int]:
        """Yields individual bits corresponding to binary representation of x"""

    def to_bits_array(self, x_array: NDArray[Any]) -> NDArray[np.uint8]:
        """Yields an NDArray of bits corresponding to binary representations of the input elements.

        Often, converting an array can be performed faster than converting each element individually.
        This operation accepts any NDArray of values, and the output array satisfies
        `output_shape = input_shape + (self.bitsize,)`.
        """
        return np.vectorize(
            lambda x: np.asarray(self.to_bits(x), dtype=np.uint8), signature='()->(n)'
        )(x_array)

    @abc.abstractmethod
    def from_bits(self, bits: Sequence[int]):
        """Combine individual bits to form x"""

    def from_bits_array(self, bits_array: NDArray[np.uint8]):
        """Combine individual bits to form classical values.

        Often, converting an array can be performed faster than converting each element individually.
        This operation accepts any NDArray of bits such that the last dimension equals `self.bitsize`,
        and the output array satisfies `output_shape = input_shape[:-1]`.
        """
        return np.vectorize(self.from_bits, signature='(n)->()')(bits_array)

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

    @abc.abstractmethod
    def is_symbolic(self) -> bool:
        """Returns True if this qdtype is parameterized with symbolic objects."""

    def iteration_length_or_zero(self) -> SymbolicInt:
        """Safe version of iteration length.

        Returns the iteration_length if the type has it or else zero.
        """
        return getattr(self, 'iteration_length', 0)

    def __str__(self):
        return f'{self.__class__.__name__}({self.num_qubits})'


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

    def is_symbolic(self) -> bool:
        return False

    def to_bits(self, x) -> List[int]:
        """Yields individual bits corresponding to binary representation of x"""
        self.assert_valid_classical_val(x)
        return [int(x)]

    def from_bits(self, bits: Sequence[int]) -> int:
        """Combine individual bits to form x"""
        assert len(bits) == 1
        return bits[0]

    def assert_valid_classical_val_array(
        self, val_array: NDArray[np.integer], debug_str: str = 'val'
    ):
        if not np.all((val_array == 0) | (val_array == 1)):
            raise ValueError(f"Bad {self} value array in {debug_str}")

    def __str__(self):
        return 'QBit()'


@attrs.frozen
class QAny(QDType):
    """Opaque bag-of-qbits type."""

    bitsize: SymbolicInt

    @property
    def num_qubits(self):
        return self.bitsize

    def get_classical_domain(self) -> Iterable[Any]:
        raise TypeError(f"Ambiguous domain for {self}. Please use a more specific type.")

    def to_bits(self, x) -> List[int]:
        # TODO: Raise an error once usage of `QAny` is minimized across the library
        return QUInt(self.bitsize).to_bits(x)

    def from_bits(self, bits: Sequence[int]) -> int:
        # TODO: Raise an error once usage of `QAny` is minimized across the library
        return QUInt(self.bitsize).from_bits(bits)

    def is_symbolic(self) -> bool:
        return is_symbolic(self.bitsize)

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

    bitsize: SymbolicInt

    @property
    def num_qubits(self):
        return self.bitsize

    def is_symbolic(self) -> bool:
        return is_symbolic(self.bitsize)

    def get_classical_domain(self) -> Iterable[int]:
        max_val = 1 << (self.bitsize - 1)
        return range(-max_val, max_val)

    def to_bits(self, x: int) -> List[int]:
        """Yields individual bits corresponding to binary representation of x"""
        if is_symbolic(self.bitsize):
            raise ValueError(f"cannot compute bits with symbolic {self.bitsize=}")

        self.assert_valid_classical_val(x)
        return [int(b) for b in np.binary_repr(x, width=self.bitsize)]

    def from_bits(self, bits: Sequence[int]) -> int:
        """Combine individual bits to form x"""
        sign = bits[0]
        x = (
            0
            if self.bitsize == 1
            else QUInt(self.bitsize - 1).from_bits([1 - x if sign else x for x in bits[1:]])
        )
        return ~x if sign else x

    def assert_valid_classical_val(self, val: int, debug_str: str = 'val'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} should be an integer, not {val!r}")
        if val < -(2 ** (self.bitsize - 1)):
            raise ValueError(f"Too-small classical {self}: {val} encountered in {debug_str}")
        if val >= 2 ** (self.bitsize - 1):
            raise ValueError(f"Too-large classical {self}: {val} encountered in {debug_str}")

    def assert_valid_classical_val_array(
        self, val_array: NDArray[np.integer], debug_str: str = 'val'
    ):
        if np.any(val_array < -(2 ** (self.bitsize - 1))):
            raise ValueError(f"Too-small classical {self}s encountered in {debug_str}")
        if np.any(val_array >= 2 ** (self.bitsize - 1)):
            raise ValueError(f"Too-large classical {self}s encountered in {debug_str}")

    def __str__(self):
        return f'QInt({self.bitsize})'


@attrs.frozen
class QIntOnesComp(QDType):
    """Signed Integer of a given width bitsize.

    A ones' complement representation is assumed for negative integers.

    Attributes:
        bitsize: The number of qubits used to represent the integer.
    """

    bitsize: SymbolicInt

    def __attrs_post_init__(self):
        if isinstance(self.bitsize, int):
            if self.num_qubits == 1:
                raise ValueError("num_qubits must be > 1.")

    @property
    def num_qubits(self):
        return self.bitsize

    def is_symbolic(self) -> bool:
        return is_symbolic(self.bitsize)

    def to_bits(self, x: int) -> List[int]:
        """Yields individual bits corresponding to binary representation of x"""
        self.assert_valid_classical_val(x)
        return [int(x < 0)] + [y ^ int(x < 0) for y in QUInt(self.bitsize - 1).to_bits(abs(x))]

    def from_bits(self, bits: Sequence[int]) -> int:
        """Combine individual bits to form x"""
        x = QUInt(self.bitsize).from_bits([b ^ bits[0] for b in bits[1:]])
        return (-1) ** bits[0] * x

    def get_classical_domain(self) -> Iterable[int]:
        max_val = 1 << (self.bitsize - 1)
        return range(-max_val + 1, max_val)

    def assert_valid_classical_val(self, val, debug_str: str = 'val'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} should be an integer, not {val!r}")
        max_val = 1 << (self.bitsize - 1)
        if not -max_val <= val <= max_val:
            raise ValueError(
                f"Classical value {val} must be in range [-{max_val}, +{max_val}] in {debug_str}"
            )


@attrs.frozen
class QUInt(QDType):
    """Unsigned integer of a given width bitsize which wraps around upon overflow.

    Similar to unsigned integer types in C. Any intended wrap around effect is
    expected to be handled by the developer.

    Attributes:
        bitsize: The number of qubits used to represent the integer.
    """

    bitsize: SymbolicInt

    @property
    def num_qubits(self):
        return self.bitsize

    def is_symbolic(self) -> bool:
        return is_symbolic(self.bitsize)

    def get_classical_domain(self) -> Iterable[Any]:
        return range(2**self.bitsize)

    def to_bits(self, x: int) -> List[int]:
        """Yields individual bits corresponding to binary representation of x"""
        self.assert_valid_classical_val(x)
        return [int(x) for x in f'{int(x):0{self.bitsize}b}']

    def to_bits_array(self, x_array: NDArray[np.integer]) -> NDArray[np.uint8]:
        """Returns the big-endian bitstrings specified by the given integers.

        Args:
            x_array: An integer or array of unsigned integers.
        """
        if is_symbolic(self.bitsize):
            raise ValueError(f"Cannot compute bits for symbolic {self.bitsize=}")

        if self.bitsize > 64:
            # use the default vectorized `to_bits`
            return super().to_bits_array(x_array)

        w = int(self.bitsize)
        x = np.atleast_1d(x_array)
        if not np.issubdtype(x.dtype, np.uint):
            assert np.all(x >= 0)
            assert np.iinfo(x.dtype).bits <= 64
            x = x.astype(np.uint64)
        assert w <= np.iinfo(x.dtype).bits
        mask = 2 ** np.arange(w - 1, 0 - 1, -1, dtype=x.dtype).reshape((w, 1))
        return (x & mask).astype(bool).astype(np.uint8).T

    def from_bits(self, bits: Sequence[int]) -> int:
        """Combine individual bits to form x"""
        return int("".join(str(x) for x in bits), 2)

    def from_bits_array(self, bits_array: NDArray[np.uint8]) -> NDArray[np.integer]:
        """Returns the integer specified by the given big-endian bitstrings.

        Args:
            bits_array: A bitstring or array of bitstrings, each of which has the 1s bit (LSB) at the end.
        Returns:
            An array of integers; one for each bitstring.
        """
        bitstrings = np.atleast_2d(bits_array)
        if bitstrings.shape[1] != self.bitsize:
            raise ValueError(f"Input bitsize {bitstrings.shape[1]} does not match {self.bitsize=}")

        if self.bitsize > 64:
            # use the default vectorized `from_bits`
            return super().from_bits_array(bits_array)

        basis = 2 ** np.arange(self.bitsize - 1, 0 - 1, -1, dtype=np.uint64)
        return np.sum(basis * bitstrings, axis=1, dtype=np.uint64)

    def assert_valid_classical_val(self, val: int, debug_str: str = 'val'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} should be an integer, not {val!r}")
        if val < 0:
            raise ValueError(f"Negative classical value encountered in {debug_str}")
        if val >= 2**self.bitsize:
            raise ValueError(f"Too-large classical value encountered in {debug_str}")

    def assert_valid_classical_val_array(
        self, val_array: NDArray[np.integer], debug_str: str = 'val'
    ):
        if np.any(val_array < 0):
            raise ValueError(f"Negative classical values encountered in {debug_str}")
        if np.any(val_array >= 2**self.bitsize):
            raise ValueError(f"Too-large classical values encountered in {debug_str}")

    def __str__(self):
        return f'QUInt({self.bitsize})'


@attrs.frozen
class BQUInt(QDType):
    """Unsigned integer whose values are bounded within a range.

    LCU methods often make use of coherent for-loops via UnaryIteration, iterating over a range
    of values stored as a superposition over the `SELECT` register. Such (nested) coherent
    for-loops can be represented using a `Tuple[Register(dtype=BQUInt),
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

    bitsize: SymbolicInt
    iteration_length: SymbolicInt = attrs.field()

    def __attrs_post_init__(self):
        if not self.is_symbolic():
            if self.iteration_length > 2**self.bitsize:
                raise ValueError(
                    "BQUInt iteration length is too large for given bitsize. "
                    f"{self.iteration_length} vs {2**self.bitsize}"
                )

    @iteration_length.default
    def _default_iteration_length(self):
        return 2**self.bitsize

    def is_symbolic(self) -> bool:
        return is_symbolic(self.bitsize, self.iteration_length)

    @property
    def num_qubits(self):
        return self.bitsize

    def get_classical_domain(self) -> Iterable[Any]:
        if isinstance(self.iteration_length, int):
            return range(0, self.iteration_length)
        raise ValueError(f'Classical Domain not defined for expression: {self.iteration_length}')

    def assert_valid_classical_val(self, val: int, debug_str: str = 'val'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} should be an integer, not {val!r}")
        if val < 0:
            raise ValueError(f"Negative classical value encountered in {debug_str}")
        if val >= self.iteration_length:
            raise ValueError(f"Too-large classical value encountered in {debug_str}")

    def to_bits(self, x: int) -> List[int]:
        """Yields individual bits corresponding to binary representation of x"""
        self.assert_valid_classical_val(x, debug_str='val')
        return QUInt(self.bitsize).to_bits(x)

    def from_bits(self, bits: Sequence[int]) -> int:
        """Combine individual bits to form x"""
        return QUInt(self.bitsize).from_bits(bits)

    def assert_valid_classical_val_array(
        self, val_array: NDArray[np.integer], debug_str: str = 'val'
    ):
        if np.any(val_array < 0):
            raise ValueError(f"Negative classical values encountered in {debug_str}")
        if np.any(val_array >= self.iteration_length):
            raise ValueError(f"Too-large classical values encountered in {debug_str}")

    def __str__(self):
        return f'{self.__class__.__name__}({self.bitsize}, {self.iteration_length})'


@attrs.frozen
class QFxp(QDType):
    r"""Fixed point type to represent real numbers.

    A real number can be approximately represented in fixed point using `num_int`
    bits for the integer part and `num_frac` bits for the fractional part. If the
    real number is signed we store negative values in two's complement form. The first
    bit can therefore be treated as the sign bit in such cases (0 for +, 1 for -).
    In total there are `bitsize = (num_int + num_frac)` bits used to represent the number.
    E.g. Using `(bitsize = 8, num_frac = 6, signed = False)` then
    $\pi \approx 3.140625 = 11.001001$, where the . represents the decimal place.

    We can specify a fixed point real number by the tuple bitsize, num_frac and
    signed, with num_int determined as `(bitsize - num_frac)`.


    ### Classical Simulation

    To hook into the classical simulator, we use fixed-width integers to represent
    values of this type. See `to_fixed_width_int` for details.
    In particular, the user should call `QFxp.to_fixed_width_int(float_value)`
    before passing a value to `bloq.call_classically`.

    The corresponding raw qdtype is either an QUInt (when `signed=False`) or
    QInt (when `signed=True`) of the same bitsize. This is the data type used
    to represent classical values during simulation, and convert to and from bits
    for intermediate values.

    For example, QFxp(6, 4) has 2 int bits and 4 frac bits, and the corresponding
    int type is QUInt(6). So a true classical value of `10.0011` will have a raw
    integer representation of `100011`.

    See https://github.com/quantumlib/Qualtran/issues/1219 for discussion on alternatives
    and future upgrades.


    Attributes:
        bitsize: The total number of qubits used to represent the integer and
            fractional part combined.
        num_frac: The number of qubits used to represent the fractional part of the real number.
        signed: Whether the number is signed or not.
    """

    bitsize: SymbolicInt
    num_frac: SymbolicInt
    signed: bool = False

    def __attrs_post_init__(self):
        if not is_symbolic(self.num_qubits) and self.num_qubits == 1 and self.signed:
            raise ValueError("num_qubits must be > 1.")
        if not is_symbolic(self.bitsize) and not is_symbolic(self.num_frac):
            if self.signed and self.bitsize == self.num_frac:
                raise ValueError("num_frac must be less than bitsize if the QFxp is signed.")
            if self.bitsize < self.num_frac:
                raise ValueError("bitsize must be >= num_frac.")

    @property
    def num_qubits(self):
        return self.bitsize

    @property
    def num_int(self) -> SymbolicInt:
        """Number of bits for the integral part."""
        return self.bitsize - self.num_frac

    def is_symbolic(self) -> bool:
        return is_symbolic(self.bitsize, self.num_frac)

    @property
    def _int_qdtype(self) -> Union[QUInt, QInt]:
        """The corresponding dtype for the raw integer representation.

        See class docstring section on "Classical Simulation" for more details.
        """
        return QInt(self.bitsize) if self.signed else QUInt(self.bitsize)

    def get_classical_domain(self) -> Iterable[int]:
        """Use the classical domain for the underlying raw integer type.

        See class docstring section on "Classical Simulation" for more details.
        """
        yield from self._int_qdtype.get_classical_domain()

    def to_bits(self, x) -> List[int]:
        """Use the underlying raw integer type.

        See class docstring section on "Classical Simulation" for more details.
        """
        return self._int_qdtype.to_bits(x)

    def from_bits(self, bits: Sequence[int]):
        """Use the underlying raw integer type.

        See class docstring section on "Classical Simulation" for more details.
        """
        return self._int_qdtype.from_bits(bits)

    def assert_valid_classical_val(self, val: int, debug_str: str = 'val'):
        """Verify using the underlying raw integer type.

        See class docstring section on "Classical Simulation" for more details.
        """
        self._int_qdtype.assert_valid_classical_val(val, debug_str)

    def to_fixed_width_int(
        self, x: Union[float, Fxp], *, require_exact: bool = False, complement: bool = True
    ) -> int:
        """Returns the interpretation of the binary representation of `x` as an integer.

        See class docstring section on "Classical Simulation" for more details on
        the choice of this representation.

        The returned value is an integer equal to `round(x * 2**self.num_frac)`.
        That is, the input value `x` is converted to a fixed-point binary value
        of `self.num_int` integral bits and `self.num_frac` fractional bits,
        and then re-interpreted as an integer by dropping the decimal point.

        For example, consider `QFxp(6, 4).to_fixed_width_int(1.5)`. As `1.5` is `0b01.1000`
        in this representation, the returned value would be `0b011000` = 24.

        For negative values, we use twos complement form. So in
        `QFxp(6, 4, signed=True).to_fixed_width_int(-1.5)`, the input is `0b10.1000`,
        which is interpreted as `0b101000` = -24.

        Args:
            x: input floating point value
            require_exact: Raise `ValueError` if `x` cannot be exactly represented.
            complement: Use twos-complement rather than sign-magnitude representation of negative values.
        """
        bits = self._fxp_to_bits(x, require_exact=require_exact, complement=complement)
        return self._int_qdtype.from_bits(bits)

    def float_from_fixed_width_int(self, x: int) -> float:
        """Helper to convert from the fixed-width-int representation to a true floating point value.

        Here `x` is the internal value used by the classical simulator.
        See `to_fixed_width_int` for conventions.

        See class docstring section on "Classical Simulation" for more details on
        the choice of this representation.
        """
        return x / 2**self.num_frac

    def __str__(self):
        if self.signed:
            return f'QFxp({self.bitsize}, {self.num_frac}, True)'
        else:
            return f'QFxp({self.bitsize}, {self.num_frac})'

    def fxp_dtype_template(self) -> Fxp:
        """A template of the `Fxp` data type for classical values.

        To construct an `Fxp` with this config, one can use:
        `Fxp(float_value, like=QFxp(...).fxp_dtype_template)`,
        or given an existing value `some_fxp_value: Fxp`:
        `some_fxp_value.like(QFxp(...).fxp_dtype_template)`.

        The following Fxp configuration is used:
         - op_sizing='same' and const_op_sizing='same' ensure that the returned
           object is not resized to a bigger fixed point number when doing
           operations with other Fxp objects.
         - shifting='trunc' ensures that when shifting the Fxp integer to
           left / right; the digits are truncated and no rounding occurs
         - overflow='wrap' ensures that when performing operations where result
           overflows, the overflowed digits are simply discarded.

        Support for `fxpmath.Fxp` is experimental, and does not hook into the classical
        simulator protocol. Once the library choice for fixed-point classical real
        values is finalized, the code will be updated to use the new functionality
        instead of delegating to raw integer values (see above).
        """
        if is_symbolic(self.bitsize) or is_symbolic(self.num_frac):
            raise ValueError(
                f"Cannot construct Fxp template for symbolic bitsizes: {self.bitsize=}, {self.num_frac=}"
            )

        return Fxp(
            None,
            n_word=self.bitsize,
            n_frac=self.num_frac,
            signed=self.signed,
            op_sizing='same',
            const_op_sizing='same',
            shifting='trunc',
            overflow='wrap',
        )

    def _get_classical_domain_fxp(self) -> Iterable[Fxp]:
        for x in self._int_qdtype.get_classical_domain():
            yield Fxp(x / 2**self.num_frac, like=self.fxp_dtype_template())

    def _fxp_to_bits(
        self, x: Union[float, Fxp], require_exact: bool = True, complement: bool = True
    ) -> List[int]:
        """Yields individual bits corresponding to binary representation of `x`.

        Args:
            x: The value to encode.
            require_exact: Raise `ValueError` if `x` cannot be exactly represented.
            complement: Use twos-complement rather than sign-magnitude representation of negative values.

        Raises:
            ValueError: If `x` is negative but this `QFxp` is not signed.
        """
        if require_exact:
            self._assert_valid_classical_val(x)
        if x < 0 and not self.signed:
            raise ValueError(f"unsigned QFxp cannot represent {x}.")
        if self.signed and not complement:
            sign = int(x < 0)
            x = abs(x)
        fxp = x if isinstance(x, Fxp) else Fxp(x)
        bits = [int(x) for x in fxp.like(self.fxp_dtype_template()).bin()]
        if self.signed and not complement:
            bits[0] = sign
        return bits

    def _from_bits_to_fxp(self, bits: Sequence[int]) -> Fxp:
        """Combine individual bits to form x"""
        bits_bin = "".join(str(x) for x in bits[:])
        fxp_bin = "0b" + bits_bin[: -self.num_frac] + "." + bits_bin[-self.num_frac :]
        return Fxp(fxp_bin, like=self.fxp_dtype_template())

    def _assert_valid_classical_val(self, val: Union[float, Fxp], debug_str: str = 'val'):
        fxp_val = val if isinstance(val, Fxp) else Fxp(val)
        if fxp_val.get_val() != fxp_val.like(self.fxp_dtype_template()).get_val():
            raise ValueError(
                f"{debug_str}={val} cannot be accurately represented using Fxp {fxp_val}"
            )


@attrs.frozen
class QMontgomeryUInt(QDType):
    """Montgomery form of an unsigned integer of a given width bitsize which wraps around upon
        overflow.

    Similar to unsigned integer types in C. Any intended wrap around effect is
    expected to be handled by the developer. Any QMontgomeryUInt can be treated as a QUInt, but not
    every QUInt can be treated as a QMontgomeryUInt. Montgomery form is used in order to compute
    fast modular multiplication.

    In order to convert an unsigned integer from a finite field x % p into Montgomery form you
    first must choose a value r > p where gcd(r, p) = 1. Typically, this value is a power of 2.

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

    bitsize: SymbolicInt

    @property
    def num_qubits(self):
        return self.bitsize

    def is_symbolic(self) -> bool:
        return is_symbolic(self.bitsize)

    def get_classical_domain(self) -> Iterable[Any]:
        return range(2**self.bitsize)

    def to_bits(self, x: int) -> List[int]:
        self.assert_valid_classical_val(x)
        return [int(x) for x in f'{int(x):0{self.bitsize}b}']

    def from_bits(self, bits: Sequence[int]) -> int:
        return int("".join(str(x) for x in bits), 2)

    def assert_valid_classical_val(self, val: int, debug_str: str = 'val'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} should be an integer, not {val!r}")
        if val < 0:
            raise ValueError(f"Negative classical value encountered in {debug_str}")
        if val >= 2**self.bitsize:
            raise ValueError(f"Too-large classical value encountered in {debug_str}")

    def assert_valid_classical_val_array(
        self, val_array: NDArray[np.integer], debug_str: str = 'val'
    ):
        if np.any(val_array < 0):
            raise ValueError(f"Negative classical values encountered in {debug_str}")
        if np.any(val_array >= 2**self.bitsize):
            raise ValueError(f"Too-large classical values encountered in {debug_str}")


QAnyInt = (QInt, QUInt, BQUInt, QMontgomeryUInt)
QAnyUInt = (QUInt, BQUInt, QMontgomeryUInt)


class QDTypeCheckingSeverity(Enum):
    """The level of type checking to enforce"""

    LOOSE = 0
    """Allow most type conversions between QAnyInt, QFxp and QAny."""

    ANY = 1
    """Disallow numeric type conversions but allow QAny and single bit conversion."""

    STRICT = 2
    """Strictly enforce type checking between registers. Only single bit conversions are allowed."""


def _check_uint_fxp_consistent(a: Union[QUInt, BQUInt, QMontgomeryUInt], b: QFxp) -> bool:
    """A uint / qfxp is consistent with a whole or totally fractional unsigned QFxp."""
    if b.signed:
        return False
    return a.num_qubits == b.num_qubits and (b.num_frac == 0 or b.num_int == 0)


def check_dtypes_consistent(
    dtype_a: QDType,
    dtype_b: QDType,
    type_checking_severity: QDTypeCheckingSeverity = QDTypeCheckingSeverity.LOOSE,
) -> bool:
    """Check if two types are consistent given our current definition on consistent types.

    Args:
        dtype_a: The dtype to check against the reference.
        dtype_b: The reference dtype.
        type_checking_severity: Severity of type checking to perform.

    Returns:
        True if the types are consistent.
    """
    same_dtypes = dtype_a == dtype_b
    same_n_qubits = dtype_a.num_qubits == dtype_b.num_qubits
    if same_dtypes:
        # Same types are always ok.
        return True
    elif dtype_a.num_qubits == 1 and same_n_qubits:
        # Single qubit types are ok.
        return True
    if type_checking_severity == QDTypeCheckingSeverity.STRICT:
        return False
    if isinstance(dtype_a, QAny) or isinstance(dtype_b, QAny):
        # QAny -> any dtype and any dtype -> QAny
        return same_n_qubits
    if type_checking_severity == QDTypeCheckingSeverity.ANY:
        return False
    if isinstance(dtype_a, QAnyInt) and isinstance(dtype_b, QAnyInt):
        # A subset of the integers should be freely interchangeable.
        return same_n_qubits
    elif isinstance(dtype_a, QAnyUInt) and isinstance(dtype_b, QFxp):
        # unsigned Fxp which is wholy an integer or < 1 part is a uint.
        return _check_uint_fxp_consistent(dtype_a, dtype_b)
    elif isinstance(dtype_b, QAnyUInt) and isinstance(dtype_a, QFxp):
        # unsigned Fxp which is wholy an integer or < 1 part is a uint.
        return _check_uint_fxp_consistent(dtype_b, dtype_a)
    else:
        return False
