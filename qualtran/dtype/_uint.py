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
from typing import Iterable, List, Sequence

import attrs
import numpy as np
from numpy.typing import NDArray

from qualtran.symbolics import is_symbolic, SymbolicInt

from ._base import BitEncoding, CDType, QDType


@attrs.frozen
class _UInt(BitEncoding[int]):
    """Unsigned integer of a given bitsize.

    Here (and throughout Qualtran), we use a big-endian bit convention. The most significant
    bit is at index 0.
    """

    bitsize: SymbolicInt

    def get_domain(self) -> Iterable[int]:
        return range(2**self.bitsize)

    def to_bits(self, x: int) -> List[int]:
        self.assert_valid_val(x)
        return [int(x) for x in f'{int(x):0{self.bitsize}b}']

    def to_bits_array(self, x_array: NDArray[np.integer]) -> NDArray[np.uint8]:
        if is_symbolic(self.bitsize):
            raise ValueError(f"Cannot compute bits for symbolic {self.bitsize=}")

        if self.bitsize > 64:
            return np.vectorize(
                lambda x: np.asarray(self.to_bits(x), dtype=np.uint8), signature='()->(n)'
            )(x_array)

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
        return int("".join(str(x) for x in bits), 2)

    def from_bits_array(self, bits_array: NDArray[np.uint8]) -> NDArray[np.uint64]:
        bitstrings = np.atleast_2d(bits_array)
        if bitstrings.shape[1] != self.bitsize:
            raise ValueError(f"Input bitsize {bitstrings.shape[1]} does not match {self.bitsize=}")

        if self.bitsize > 64:
            # use the default vectorized `from_bits`
            return np.vectorize(self.from_bits, signature='(n)->()')(bits_array)

        basis = 2 ** np.arange(self.bitsize - 1, 0 - 1, -1, dtype=np.uint64)
        return np.sum(basis * bitstrings, axis=1, dtype=np.uint64)  # type: ignore[return-value]

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


@attrs.frozen
class QUInt(QDType[int]):
    """Unsigned quantum integer of a given bitsize.

    Here (and throughout Qualtran), we use a big-endian bit convention. The most significant
    bit is at index 0.

    Args:
        bitsize: The number of qubits used to represent the integer.
    """

    bitsize: SymbolicInt

    @cached_property
    def _bit_encoding(self) -> BitEncoding[int]:
        return _UInt(self.bitsize)


@attrs.frozen
class CUInt(CDType[int]):
    """Unsigned classical integer of a given bitsize.

    Here (and throughout Qualtran), we use a big-endian bit convention. The most significant
    bit is at index 0.

    Args:
        bitsize: The number of classical bits used to represent the integer.
    """

    bitsize: SymbolicInt

    @cached_property
    def _bit_encoding(self) -> BitEncoding[int]:
        return _UInt(self.bitsize)
