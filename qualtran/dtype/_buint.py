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

from typing import Iterable, List, Sequence

import attrs
import numpy as np
from numpy.typing import NDArray

from qualtran.symbolics import is_symbolic, SymbolicInt

from ._base import BitEncoding, CDType, QDType
from ._uint import _UInt


@attrs.frozen
class _BUInt(BitEncoding[int]):
    """Unsigned integer whose values are bounded within a range.

    Args:
        bitsize: The number of bits used to represent the integer.
        bound: The bound (exclusive)
    """

    bitsize: SymbolicInt
    bound: SymbolicInt

    def __attrs_post_init__(self):
        if is_symbolic(self.bitsize) or is_symbolic(self.bound):
            return

        if self.bound > 2**self.bitsize:
            raise ValueError(
                "BUInt value bound is too large for given bitsize. "
                f"{self.bound} vs {2 ** self.bitsize}"
            )

    def get_domain(self) -> Iterable[int]:
        if isinstance(self.bound, int):
            return range(0, self.bound)
        raise ValueError(f'Classical domain not defined for {self}')

    def assert_valid_val(self, val: int, debug_str: str = 'val') -> None:
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} should be an integer, not {val!r}")
        if val < 0:
            raise ValueError(f"Negative classical value encountered in {debug_str}")
        if val >= self.bound:
            raise ValueError(f"Too-large classical value encountered in {debug_str}")

    def to_bits(self, x: int) -> List[int]:
        """Yields individual bits corresponding to binary representation of x"""
        self.assert_valid_val(x)
        return _UInt(self.bitsize).to_bits(x)

    def from_bits(self, bits: Sequence[int]) -> int:
        """Combine individual bits to form x"""
        val = _UInt(self.bitsize).from_bits(bits)
        self.assert_valid_val(val)
        return val

    def assert_valid_val_array(self, val_array: NDArray[np.integer], debug_str: str = 'val'):
        if np.any(val_array < 0):
            raise ValueError(f"Negative classical values encountered in {debug_str}")
        if np.any(val_array >= self.bound):
            raise ValueError(f"Too-large classical values encountered in {debug_str}")


@attrs.frozen
class BQUInt(QDType[int]):
    """Unsigned quantum integer whose values are bounded within a range.

    LCU methods often make use of coherent for-loops via UnaryIteration, iterating over a range
    of values stored as a superposition over the `SELECT` register. Such (nested) coherent
    for-loops can be represented using a `Tuple[Register(dtype=BQUInt), ...]` where the
    i'th entry stores the bitsize and iteration length of i'th
    nested for-loop.

    One useful feature when processing such nested for-loops is to flatten out a composite index,
    represented by a tuple of indices (i, j, ...), one for each selection register into a single
    integer that can be used to index a flat target register. An example of such a mapping
    function is described in Eq.45 of https://arxiv.org/abs/1805.03662. A general version of this
    mapping function can be implemented using `numpy.ravel_multi_index` and `numpy.unravel_index`.

    Examples:
        We can flatten a 2D for-loop as follows

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

        Similarly, we can flatten a 3D for-loop as follows
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

    Args:
        bitsize: The number of qubits used to represent the integer.
        iteration_length: The length of the iteration range.
    """

    bitsize: SymbolicInt
    iteration_length: SymbolicInt = attrs.field()

    def __attrs_post_init__(self):
        if not self.is_symbolic():
            if self.iteration_length > 2**self.bitsize:
                raise ValueError(
                    f"{self} iteration length is too large for given bitsize. "
                    f"{self.iteration_length} vs {2 ** self.bitsize}"
                )

    @iteration_length.default
    def _default_iteration_length(self):
        return 2**self.bitsize

    @property
    def bound(self) -> SymbolicInt:
        return self.iteration_length

    def is_symbolic(self) -> bool:
        return is_symbolic(self.bitsize, self.iteration_length)

    @property
    def _bit_encoding(self) -> BitEncoding[int]:
        return _BUInt(self.bitsize, self.iteration_length)

    def __str__(self):
        return f'{self.__class__.__name__}({self.bitsize}, {self.iteration_length})'


@attrs.frozen
class BCUInt(CDType[int]):
    """Unsigned classical integer whose values are bounded within a range.

    Args:
        bitsize: The number of bits used to represent the integer.
        bound: The value bound (exclusive).
    """

    bitsize: SymbolicInt
    bound: SymbolicInt = attrs.field()

    def __attrs_post_init__(self):
        if not self.is_symbolic():
            if self.bound > 2**self.bitsize:
                raise ValueError(
                    f"{self} bound is too large for given bitsize. "
                    f"{self.bound} vs {2 ** self.bitsize}"
                )

    @bound.default
    def _default_bound(self):
        return 2**self.bitsize

    def is_symbolic(self) -> bool:
        return is_symbolic(self.bitsize, self.bound)

    @property
    def _bit_encoding(self) -> BitEncoding[int]:
        return _BUInt(self.bitsize, self.bound)

    def __str__(self):
        return f'{self.__class__.__name__}({self.bitsize}, {self.bound})'
