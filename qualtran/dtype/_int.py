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
from ._uint import _UInt


@attrs.frozen
class _Int(BitEncoding[int]):
    """Signed integer of a given bitsize.

    Use `QInt` or `CInt` for quantum or classical implementations, respectively.

    A two's complement representation is used for negative integers.
    Here (and throughout Qualtran), we use a big-endian bit convention.
    The most significant bit is at index 0.
    """

    bitsize: SymbolicInt

    def get_domain(self) -> Iterable[int]:
        max_val = 1 << (self.bitsize - 1)
        return range(-max_val, max_val)

    def to_bits(self, x: int) -> List[int]:
        if is_symbolic(self.bitsize):
            raise ValueError(f"cannot compute bits with symbolic {self.bitsize=}")

        self.assert_valid_val(x)
        return [int(b) for b in np.binary_repr(x, width=self.bitsize)]

    def from_bits(self, bits: Sequence[int]) -> int:
        sign = bits[0]
        x = (
            0
            if self.bitsize == 1
            else _UInt(self.bitsize - 1).from_bits([1 - x if sign else x for x in bits[1:]])
        )
        return ~x if sign else x

    def assert_valid_val(self, val: int, debug_str: str = 'val'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} should be an integer, not {val!r}")
        if val < -(2 ** (self.bitsize - 1)):
            raise ValueError(f"Too-small classical {self}: {val} encountered in {debug_str}")
        if val >= 2 ** (self.bitsize - 1):
            raise ValueError(f"Too-large classical {self}: {val} encountered in {debug_str}")

    def assert_valid_val_array(self, val_array: NDArray[np.integer], debug_str: str = 'val'):
        if np.any(val_array < -(2 ** (self.bitsize - 1))):
            raise ValueError(f"Too-small classical {self}s encountered in {debug_str}")
        if np.any(val_array >= 2 ** (self.bitsize - 1)):
            raise ValueError(f"Too-large classical {self}s encountered in {debug_str}")


@attrs.frozen
class QInt(QDType[int]):
    """Signed quantum integer of a given bitsize.

    A two's complement representation is used for negative integers.
    Here (and throughout Qualtran), we use a big-endian bit convention.
    The most significant bit is at index 0.

    Args:
        bitsize: The number of qubits used to represent the integer.
    """

    bitsize: SymbolicInt

    @cached_property
    def _bit_encoding(self) -> BitEncoding[int]:
        return _Int(self.bitsize)


@attrs.frozen
class CInt(CDType[int]):
    """Signed classical integer of a given bitsize.

    A two's complement representation is used for negative integers.
    Here (and throughout Qualtran), we use a big-endian bit convention.
    The most significant bit is at index 0.

    Args:
        bitsize: The number of qubits used to represent the integer.
    """

    bitsize: SymbolicInt

    @cached_property
    def _bit_encoding(self) -> BitEncoding[int]:
        return _Int(self.bitsize)
