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

from ._base import BitEncoding, CDType, QDType


@attrs.frozen
class _Bit(BitEncoding[int]):
    """A single quantum or classical bit. The smallest addressable unit of data.

    Use either `QBit()` or `CBit()` for quantum or classical implementations, respectively.
    """

    @property
    def bitsize(self) -> int:
        return 1

    def get_domain(self) -> Iterable[int]:
        yield from (0, 1)

    def assert_valid_val(self, val: int, debug_str: str = 'val'):
        if not (val == 0 or val == 1):
            raise ValueError(f"Bad bit value: {val} in {debug_str}")

    def to_bits(self, x: int) -> List[int]:
        """Yields individual bits corresponding to binary representation of x"""
        self.assert_valid_val(x)
        return [int(x)]

    def from_bits(self, bits: Sequence[int]) -> int:
        """Combine individual bits to form x"""
        assert len(bits) == 1
        return bits[0]

    def assert_valid_val_array(
        self, val_array: NDArray[np.integer], debug_str: str = 'val'
    ) -> None:
        if not np.all((val_array == 0) | (val_array == 1)):
            raise ValueError(f"Bad bit value array in {debug_str}")


@attrs.frozen
class QBit(QDType[int]):
    """A single qubit. The smallest addressable unit of quantum data."""

    @cached_property
    def _bit_encoding(self) -> BitEncoding[int]:
        return _Bit()

    def is_symbolic(self) -> bool:
        return False

    def __str__(self) -> str:
        return 'QBit()'


@attrs.frozen
class CBit(CDType[int]):
    """A single classical bit. The smallest addressable unit of classical data."""

    @cached_property
    def _bit_encoding(self) -> BitEncoding[int]:
        return _Bit()

    def is_symbolic(self) -> bool:
        return False

    def __str__(self) -> str:
        return 'CBit()'
