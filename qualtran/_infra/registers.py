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

"""Classes for specifying `Bloq.registers`."""
import enum
import itertools
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, overload, Tuple, Union

import attrs
import numpy as np
from attrs import field, frozen

from .data_types import QAny, QBit, QDType


class Side(enum.Flag):
    """Denote LEFT, RIGHT, or THRU registers.

    LEFT registers serve as input lines (only) to the Bloq. RIGHT registers are output
    lines (only) from the Bloq. THRU registers are both input and output.

    Traditional unitary operations will have THRU registers that operate on a collection of
    qubits which are then made available to following operations. RIGHT and LEFT registers
    imply allocation, deallocation, or reshaping of the registers.
    """

    LEFT = enum.auto()
    RIGHT = enum.auto()
    THRU = LEFT | RIGHT


@frozen
class Register:
    """A data type describing a register of qubits.

    Each register has a name as well as attributes describing the quantum data expected
    to be passed to the register. A collection of `Register` objects can be used to define
    a bloq's signature, see the `Signature` class.

    Attributes:
        name: The string name of the register
        _bitsize: The number of (qu)bits in the register OR the quantum data type of the register.
            If an integer is given it will be converted into either a QAny
            dtype or QBit dtype (_bitsize = 1).
        shape: A tuple of integer dimensions to declare a multidimensional register. The
            total number of bits is the product of entries in this tuple times `bitsize`.
        side: Whether this is a left, right, or thru register. See the documentation for `Side`
            for more information.
    """

    name: str
    _bitsize: Union[int, QDType] = field(
        converter=lambda v: v if isinstance(v, QDType) else QBit() if v == 1 else QAny(v)
    )
    shape: Tuple[int, ...] = field(
        default=tuple(), converter=lambda v: (v,) if isinstance(v, int) else tuple(v)
    )
    side: Side = Side.THRU

    @property
    def dtype(self) -> QDType:
        return self._bitsize

    @property
    def bitsize(self) -> int:
        return self.dtype.num_qubits

    def all_idxs(self) -> Iterable[Tuple[int, ...]]:
        """Iterate over all possible indices of a multidimensional register."""
        yield from itertools.product(*[range(sh) for sh in self.shape])

    def total_bits(self) -> int:
        """The total number of bits in this register.

        This is the product of bitsize and each of the dimensions in `shape`.
        """
        return self.bitsize * int(np.prod(self.shape))

    def adjoint(self) -> 'Register':
        """Return the 'adjoint' of this register by switching RIGHT and LEFT registers."""
        if self.side is Side.THRU:
            return self
        if self.side is Side.LEFT:
            return attrs.evolve(self, side=Side.RIGHT)
        if self.side is Side.RIGHT:
            return attrs.evolve(self, side=Side.LEFT)
        raise ValueError(f"Unknown side {self.side}")


@frozen
class SelectionRegister(Register):
    """Register used to represent SELECT register for various LCU methods.

    `SelectionRegister` extends the `Register` class to store the iteration length
    corresponding to that register along with its size.

    LCU methods often make use of coherent for-loops via UnaryIteration, iterating over a range
    of values stored as a superposition over the `SELECT` register. Such (nested) coherent
    for-loops can be represented using a `Tuple[SelectionRegister, ...]` where the i'th entry
    stores the bitsize and iteration length of i'th nested for-loop.

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
    """

    name: str
    _bitsize: Union[int, QDType] = field(
        converter=lambda v: v if isinstance(v, QDType) else QBit() if v == 1 else QAny(v)
    )
    iteration_length: int = field()
    shape: Tuple[int, ...] = field(
        converter=lambda v: (v,) if isinstance(v, int) else tuple(v), default=()
    )
    side: Side = Side.THRU

    @iteration_length.default
    def _default_iteration_length(self):
        return 2**self.bitsize

    @iteration_length.validator
    def validate_iteration_length(self, attribute, value):
        if len(self.shape) != 0:
            raise ValueError(f'Selection register {self.name} should be flat. Found {self.shape=}')
        if not (0 <= value <= 2**self.bitsize):
            raise ValueError(f'iteration length must be in range [0, 2^{self.bitsize}]')


def _dedupe(kv_iter: Iterable[Tuple[str, Register]]) -> Dict[str, Register]:
    """Construct a dictionary, but check that there are no duplicate keys."""
    # throw ValueError if duplicate keys are provided.
    d = {}
    for k, v in kv_iter:
        if k in d:
            raise ValueError(f"Register {k} is specified more than once per side.") from None
        d[k] = v
    return d


class Signature:
    """An ordered sequence of `Register`s that follow the rules for a bloq signature.

    `Bloq.signature` is a property of all bloqs, and should be an object of this type.
    It is analogous to a function signature in traditional computing where we specify the
    names and types of the expected inputs and outputs.

    Each LEFT (including thru) register must have a unique name. Each RIGHT (including thru)
    register must have a unique name.

    Args:
        registers: The registers comprising the signature.
    """

    def __init__(self, registers: Iterable[Register]):
        self._registers = tuple(registers)
        self._lefts = _dedupe((reg.name, reg) for reg in self._registers if reg.side & Side.LEFT)
        self._rights = _dedupe((reg.name, reg) for reg in self._registers if reg.side & Side.RIGHT)

    @classmethod
    def build(cls, **registers: int) -> 'Signature':
        """Construct a Signature comprised of simple thru registers.

        Args:
            registers: keyword arguments mapping register name to bitsize. All registers
                will be 0-dimensional and THRU.
        """
        return cls(Register(name=k, bitsize=v) for k, v in registers.items() if v)

    def lefts(self) -> Iterable[Register]:
        """Iterable over all registers that appear on the LEFT as input."""
        yield from self._lefts.values()

    def rights(self) -> Iterable[Register]:
        """Iterable over all registers that appear on the RIGHT as output."""
        yield from self._rights.values()

    def get_left(self, name: str) -> Register:
        """Get a left register by name."""
        return self._lefts[name]

    def get_right(self, name: str) -> Register:
        """Get a right register by name."""
        return self._rights[name]

    def groups(self) -> Iterable[Tuple[str, List[Register]]]:
        """Iterate over register groups by name.

        Registers with shared names (but differing `side` attributes) can be implicitly grouped.
        """
        groups = defaultdict(list)
        for reg in self._registers:
            groups[reg.name].append(reg)

        yield from groups.items()

    def adjoint(self) -> 'Signature':
        """Swap all RIGHT and LEFT registers in this collection."""
        return Signature(reg.adjoint() for reg in self._registers)

    def __repr__(self):
        return f'Signature({repr(self._registers)})'

    @overload
    def __getitem__(self, key: int) -> Register:
        pass

    @overload
    def __getitem__(self, key: slice) -> Tuple[Register, ...]:
        pass

    def __getitem__(self, key):
        return self._registers[key]

    def __contains__(self, item: Register) -> bool:
        return item in self._registers

    def __iter__(self) -> Iterator[Register]:
        yield from self._registers

    def __len__(self) -> int:
        return len(self._registers)

    def __hash__(self):
        return hash(self._registers)

    def __eq__(self, other) -> bool:
        return self._registers == other._registers
