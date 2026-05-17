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
class _IntSignMag(BitEncoding[int]):
    """Sign-magnitude encoding.

    The most significant bit is the sign bit (0=positive, 1=negative).
    The remaining bits encode the absolute value.
    """

    bitsize: SymbolicInt

    def get_domain(self) -> Iterable[int]:
        max_val = 1 << (self.bitsize - 1)
        return range(-max_val + 1, max_val)

    def to_bits(self, x: int) -> List[int]:
        if is_symbolic(self.bitsize):
            raise ValueError(f"cannot compute bits with symbolic {self.bitsize=}")
        self.assert_valid_val(x)
        return [1 if x < 0 else 0] + [
            int(b) for b in np.binary_repr(np.abs(x), width=self.bitsize - 1)
        ]

    def from_bits(self, bits: Sequence[int]) -> int:
        sign = bits[0]
        if self.bitsize == 1:
            return 0
        magnitude = 0
        for b in bits[1:]:
            magnitude = (magnitude << 1) | b
        return -magnitude if sign else magnitude

    def assert_valid_val(self, val: int, debug_str: str = 'val'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} should be an integer, not {val!r}")
        max_val = 1 << (self.bitsize - 1)
        if val <= -max_val:
            raise ValueError(f"Too-small classical {self}: {val} encountered in {debug_str}")
        if val >= max_val:
            raise ValueError(f"Too-large classical {self}: {val} encountered in {debug_str}")

    def assert_valid_val_array(self, val_array: NDArray[np.integer], debug_str: str = 'val'):
        max_val = 1 << (self.bitsize - 1)
        if np.any(val_array <= -max_val):
            raise ValueError(f"Too-small classical {self}s encountered in {debug_str}")
        if np.any(val_array >= max_val):
            raise ValueError(f"Too-large classical {self}s encountered in {debug_str}")


@attrs.frozen
class QIntSignMag(QDType[int]):
    """Sign-magnitude signed quantum integer.

    The most significant bit is the sign bit (0=positive, 1=negative),
    and the remaining bits encode the absolute value.

    For an n-bit QSignInt, the representable range is [-(2^(n-1)-1), 2^(n-1)-1].
    Note that this means +0 and -0 are distinct representations.

    Args:
        bitsize: The number of qubits used to represent the integer.
    """

    bitsize: SymbolicInt

    def __attrs_post_init__(self):
        if isinstance(self.bitsize, int):
            if self.bitsize < 2:
                raise ValueError("bitsize must be >= 2.")

    @cached_property
    def _bit_encoding(self) -> BitEncoding[int]:
        return _IntSignMag(self.bitsize)

    def is_symbolic(self) -> bool:
        return is_symbolic(self.bitsize)

    def __str__(self):
        return f'QIntSignMag({self.bitsize})'


@attrs.frozen
class CIntSignMag(CDType[int]):
    """Sign-magnitude signed classical integer.

    The most significant bit is the sign bit (0=positive, 1=negative),
    and the remaining bits encode the absolute value.

    Args:
        bitsize: The number of qubits used to represent the integer.
    """

    bitsize: SymbolicInt

    def __attrs_post_init__(self):
        if isinstance(self.bitsize, int):
            if self.bitsize < 2:
                raise ValueError("bitsize must be >= 2.")

    @cached_property
    def _bit_encoding(self) -> BitEncoding[int]:
        return _IntSignMag(self.bitsize)

    def is_symbolic(self) -> bool:
        return is_symbolic(self.bitsize)

    def __str__(self):
        return f'CIntSignMag({self.bitsize})'
