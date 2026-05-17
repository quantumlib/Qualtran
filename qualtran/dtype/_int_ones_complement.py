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

from qualtran.symbolics import SymbolicInt

from ._base import BitEncoding, CDType, QDType
from ._uint import _UInt


@attrs.frozen
class _IntOnesComp(BitEncoding[int]):
    """Ones' complement signed integer of a given bitsize.

    This contrasts with `_Int` by using the ones' complement representation for negative
    integers.
    Here (and throughout Qualtran), we use a big-endian bit convention.
    The most significant bit is at index 0.
    """

    bitsize: SymbolicInt

    def __attrs_post_init__(self):
        if isinstance(self.bitsize, int):
            if self.bitsize == 1:
                raise ValueError("bitsize must be > 1.")

    def to_bits(self, x: int) -> List[int]:
        self.assert_valid_val(x)
        return [int(x < 0)] + [y ^ int(x < 0) for y in _UInt(self.bitsize - 1).to_bits(abs(x))]

    def from_bits(self, bits: Sequence[int]) -> int:
        x = _UInt(self.bitsize).from_bits([b ^ bits[0] for b in bits[1:]])
        return (-1) ** int(bits[0]) * x

    def get_domain(self) -> Iterable[int]:
        max_val = 1 << (self.bitsize - 1)
        return range(-max_val + 1, max_val)

    def assert_valid_val(self, val: int, debug_str: str = 'val') -> None:
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} should be an integer, not {val!r}")
        max_val = 1 << (self.bitsize - 1)
        if not -max_val <= val <= max_val:
            raise ValueError(
                f"Classical value {val} must be in range [-{max_val}, +{max_val}] in {debug_str}"
            )


@attrs.frozen
class QIntOnesComp(QDType[int]):
    """Ones' complement signed quantum integer of a given bitsize.

    This contrasts with `QInt` by using the ones' complement representation for negative
    integers.
    Here (and throughout Qualtran), we use a big-endian bit convention.
    The most significant bit is at index 0.

    Args:
        bitsize: The number of qubits used to represent the integer.
    """

    bitsize: SymbolicInt

    def __attrs_post_init__(self):
        if isinstance(self.bitsize, int):
            if self.bitsize == 1:
                raise ValueError("bitsize must be > 1.")

    @cached_property
    def _bit_encoding(self) -> BitEncoding[int]:
        return _IntOnesComp(self.bitsize)


@attrs.frozen
class CIntOnesComp(CDType[int]):
    """Ones' complement signed classical integer of a given bitsize.

    This contrasts with `CInt` by using the ones' complement representation for negative
    integers.
    Here (and throughout Qualtran), we use a big-endian bit convention.
    The most significant bit is at index 0.

    Args:
        bitsize: The number of classical bits used to represent the integer.
    """

    bitsize: SymbolicInt

    def __attrs_post_init__(self):
        if isinstance(self.bitsize, int):
            if self.bitsize == 1:
                raise ValueError("bitsize must be > 1.")

    @cached_property
    def _bit_encoding(self) -> BitEncoding[int]:
        return _IntOnesComp(self.bitsize)
