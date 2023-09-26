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
from typing import Dict, Iterable, Iterator, List, overload, Tuple, TYPE_CHECKING

import numpy as np
from attr import frozen
from numpy.typing import NDArray

if TYPE_CHECKING:
    import cirq


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
        bitsize: The number of (qu)bits in the register.
        shape: A tuple of integer dimensions to declare a multidimensional register. The
            total number of bits is the product of entries in this tuple times `bitsize`.
        side: Whether this is a left, right, or thru register. See the documentation for `Side`
            for more information.
    """

    name: str
    bitsize: int
    shape: Tuple[int, ...] = tuple()
    side: Side = Side.THRU

    def all_idxs(self) -> Iterable[Tuple[int, ...]]:
        """Iterate over all possible indices of a multidimensional register."""
        yield from itertools.product(*[range(sh) for sh in self.shape])

    def total_bits(self) -> int:
        """The total number of bits in this register.

        This is the product of bitsize and each of the dimensions in `shape`.
        """
        return self.bitsize * int(np.product(self.shape))


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
        return cls(Register(name=k, bitsize=v) for k, v in registers.items())

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

    def get_cirq_quregs(self) -> Dict[str, 'NDArray[cirq.Qid]']:
        """Get arrays of cirq qubits for these registers."""
        import cirq_ft

        cirq_regs = [
            cirq_ft.Register(name=reg.name, shape=reg.shape + (reg.bitsize,))
            for reg in self.lefts()
        ]
        return cirq_ft.Registers(cirq_regs).get_named_qubits()

    def __hash__(self):
        return hash(self._registers)

    def __eq__(self, other) -> bool:
        return self._registers == other._registers
