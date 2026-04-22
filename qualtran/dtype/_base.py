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

import abc
import warnings
from typing import cast, Generic, Iterable, List, Sequence, Tuple, TypeVar

import attrs
import numpy as np
from numpy.typing import NDArray

from qualtran.symbolics import is_symbolic, SymbolicInt

T = TypeVar('T')


class BitEncoding(Generic[T], metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def bitsize(self) -> SymbolicInt: ...

    @abc.abstractmethod
    def get_domain(self) -> Iterable[T]:
        """Yields all possible classical (computational basis state) values representable
        by this type."""

    @abc.abstractmethod
    def to_bits(self, x: T) -> List[int]:
        """Yields individual bits corresponding to binary representation of x"""

    def to_bits_array(self, x_array: NDArray) -> NDArray[np.uint8]:
        """Yields an NDArray of bits corresponding to binary representations of the input elements.

        Often, converting an array can be performed faster than converting each element individually.
        This operation accepts any NDArray of values, and the output array satisfies
        `output_shape = input_shape + (self.bitsize,)`.
        """
        return np.vectorize(
            lambda x: np.asarray(self.to_bits(x), dtype=np.uint8), signature='()->(n)'
        )(x_array)

    @abc.abstractmethod
    def from_bits(self, bits: Sequence[int]) -> T:
        """Combine individual bits to form x"""

    def from_bits_array(self, bits_array: NDArray[np.uint8]) -> NDArray:
        """Combine individual bits to form classical values.

        Often, converting an array can be performed faster than converting each element individually.
        This operation accepts any NDArray of bits such that the last dimension equals `self.bitsize`,
        and the output array satisfies `output_shape = input_shape[:-1]`.
        """
        return np.vectorize(self.from_bits, signature='(n)->()')(bits_array)

    @abc.abstractmethod
    def assert_valid_val(self, val: T, debug_str: str = 'val') -> None:
        """Raises an exception if `val` is not a valid classical value for this type.

        Args:
            val: A classical value that should be in the domain of this QDType.
            debug_str: Optional debugging information to use in exception messages.
        """

    def assert_valid_val_array(self, val_array: NDArray, debug_str: str = 'val') -> None:
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
            self.assert_valid_val(val, debug_str=debug_str)


@attrs.frozen
class _BitEncodingShim(BitEncoding[T]):
    """Shim an old-style QDType to follow the BitEncoding interface.

    Before the introduction of classical data types (QCDType and CDType), QDType classes
    described how to encode values into bits (for classical simulation) and qubits (for
    quantum programs). The encoding schemes don't care whether the substrate is bits or
    qubits but the CompositeBloq type-checking does care; so we've moved the encoding
    logic to descendants of `BitEncoding`. Each `QCDType` "has a" BitEncoding and "is a"
    quantum data type or classical data type.

    This shim uses encoding logic found in the methods of an old-style QDType to satisfy
    the BitEncoding interface for backwards compatibility. Developers with custom QDTypes
    should port their custom data types to use a BitEncoding.

    """

    qdtype: 'QDType[T]'

    @property
    def bitsize(self) -> SymbolicInt:
        return self.qdtype.num_qubits

    def get_domain(self) -> Iterable[T]:
        yield from self.qdtype.get_classical_domain()

    def to_bits(self, x: T) -> List[int]:
        return self.qdtype.to_bits(x)

    def to_bits_array(self, x_array: NDArray) -> NDArray[np.uint8]:
        return np.vectorize(
            lambda x: np.asarray(self.qdtype.to_bits(x), dtype=np.uint8), signature='()->(n)'
        )(x_array)

    def from_bits(self, bits: Sequence[int]) -> T:
        return self.qdtype.from_bits(bits)

    def from_bits_array(self, bits_array: NDArray[np.uint8]) -> NDArray:
        return np.vectorize(self.qdtype.from_bits, signature='(n)->()')(bits_array)

    def assert_valid_val(self, val: T, debug_str: str = 'val') -> None:
        return self.qdtype.assert_valid_classical_val(val, debug_str=debug_str)

    def assert_valid_val_array(self, val_array: NDArray, debug_str: str = 'val') -> None:
        for val in val_array.reshape(-1):
            self.qdtype.assert_valid_classical_val(val)


@attrs.frozen
class ShapedQCDType:
    qcdtype: 'QCDType'
    shape: Tuple[int, ...] = attrs.field(
        default=tuple(), converter=lambda v: (v,) if isinstance(v, int) else tuple(v)
    )


class QCDType(Generic[T], metaclass=abc.ABCMeta):
    """The abstract interface for quantum/classical quantum computing data types."""

    @property
    @abc.abstractmethod
    def _bit_encoding(self) -> BitEncoding[T]:
        """The class describing how bits are encoded in this datatype."""

    @property
    def num_bits(self) -> int:
        """Number of bits (quantum and classical) required to represent a single instance of
        this data type."""
        return self.num_qubits + self.num_cbits

    @property
    @abc.abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits required to represent a single instance of this data type."""

    @property
    @abc.abstractmethod
    def num_cbits(self) -> int:
        """Number of classical bits required to represent a single instance of this data type."""

    def get_classical_domain(self) -> Iterable[T]:
        """Yields all possible classical (computational basis state) values representable
        by this type."""
        yield from self._bit_encoding.get_domain()

    def to_bits(self, x: T) -> List[int]:
        """Yields individual bits corresponding to binary representation of x"""
        return self._bit_encoding.to_bits(x)

    def to_bits_array(self, x_array: NDArray) -> NDArray[np.uint8]:
        """Yields an NDArray of bits corresponding to binary representations of the input elements.

        Often, converting an array can be performed faster than converting each element individually.
        This operation accepts any NDArray of values, and the output array satisfies
        `output_shape = input_shape + (self.bitsize,)`.
        """
        return self._bit_encoding.to_bits_array(x_array)

    def from_bits(self, bits: Sequence[int]) -> T:
        """Combine individual bits to form x"""
        return self._bit_encoding.from_bits(bits)

    def from_bits_array(self, bits_array: NDArray[np.uint8]) -> NDArray:
        """Combine individual bits to form classical values.

        Often, converting an array can be performed faster than converting each element individually.
        This operation accepts any NDArray of bits such that the last dimension equals `self.bitsize`,
        and the output array satisfies `output_shape = input_shape[:-1]`.
        """
        return self._bit_encoding.from_bits_array(bits_array)

    def assert_valid_classical_val(self, val: T, debug_str: str = 'val') -> None:
        """Raises an exception if `val` is not a valid classical value for this type.

        Args:
            val: A classical value that should be in the domain of this QDType.
            debug_str: Optional debugging information to use in exception messages.
        """
        return self._bit_encoding.assert_valid_val(val=val, debug_str=debug_str)

    def assert_valid_classical_val_array(self, val_array: NDArray, debug_str: str = 'val') -> None:
        """Raises an exception if `val_array` is not a valid array of classical values
        for this type.

        Often, validation on an array can be performed faster than validating each element
        individually.

        Args:
            val_array: A numpy array of classical values. Each value should be in the domain
                of this QDType.
            debug_str: Optional debugging information to use in exception messages.
        """
        return self._bit_encoding.assert_valid_val_array(val_array=val_array, debug_str=debug_str)

    def is_symbolic(self) -> bool:
        """Returns True if this dtype is parameterized with symbolic objects."""
        return is_symbolic(self._bit_encoding.bitsize)

    def iteration_length_or_zero(self) -> SymbolicInt:
        """Safe version of iteration length.

        Returns the iteration_length if the type has it or else zero.
        """
        # TODO: remove https://github.com/quantumlib/Qualtran/issues/1716
        return getattr(self, 'iteration_length', 0)

    def __getitem__(self, shape):
        """QInt(8)[20] returns a size-20 array of QInt(8)"""
        return ShapedQCDType(qcdtype=self, shape=shape)

    @classmethod
    def _pkg_(cls):
        return 'qualtran'

    def __str__(self):
        return f'{self.__class__.__name__}({self.num_bits})'


class QDType(QCDType[T], metaclass=abc.ABCMeta):
    """The abstract interface for quantum data types."""

    @property
    def _bit_encoding(self) -> BitEncoding[T]:
        """The class describing how bits are encoded in this datatype."""
        warnings.warn(
            f"{self} must provide a BitEncoding. "
            f"This shim will become an error in the future. "
            f"Omitting this may cause infinite loops.",
            DeprecationWarning,
        )
        return _BitEncodingShim(self)

    @property
    def num_qubits(self) -> int:
        return cast(int, self._bit_encoding.bitsize)

    @property
    def num_cbits(self) -> int:
        """QDTypes have zero qubits."""
        return 0

    def __str__(self):
        return f'{self.__class__.__name__}({self.num_qubits})'


class CDType(QCDType[T], metaclass=abc.ABCMeta):
    """The abstract interface for classical data types."""

    @property
    def num_qubits(self) -> int:
        """CDTypes have zero qubits."""
        return 0

    @property
    def num_cbits(self) -> int:
        return cast(int, self._bit_encoding.bitsize)

    def __str__(self):
        return f'{self.__class__.__name__}({self.num_cbits})'
