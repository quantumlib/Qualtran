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
from typing import cast, Dict, Iterable, Iterator, List, overload, Tuple, Union

import attrs
import sympy
from attrs import field, frozen

from qualtran.symbolics import is_symbolic, prod, smax, ssum, SymbolicInt

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
    """A register serves as the input/output quantum data specifications in a bloq's `Signature`.

    Each register has a name and a quantum data type. A collection of `Register` objects are used
    to define a bloq's signature, see the `Signature` class.

    Attributes:
        name: The string name of the register. This name is used to 'wire up' quantum inputs
            by name, analogous to Python's keyword-arguments.
        dtype: The quantum data type of the register, for example `QBit()`, `QUInt(n)`, `QAny(n)`,
            or any of the data types provided in the top-level `qualtran` namespace.
        shape: An optional tuple of integer dimensions to declare a multidimensional register. The
            total number of bits is the product of entries in this tuple times `bitsize`.
        side: Whether this is a left, right, or thru register. See the documentation for `Side`
            for more information.
    """

    name: str
    dtype: QDType
    _shape: Tuple[SymbolicInt, ...] = field(
        default=tuple(), converter=lambda v: (v,) if isinstance(v, int) else tuple(v)
    )
    side: Side = Side.THRU

    def __attrs_post_init__(self):
        if not isinstance(self.dtype, QDType):
            raise ValueError(f'dtype must be a QDType: found {type(self.dtype)}')

    def is_symbolic(self) -> bool:
        return is_symbolic(self.dtype, *self._shape)

    @property
    def shape_symbolic(self) -> Tuple[SymbolicInt, ...]:
        return self._shape

    @property
    def shape(self) -> Tuple[int, ...]:
        if is_symbolic(*self._shape):
            raise ValueError(f"{self} is symbolic. Cannot get real-valued shape.")
        return cast(Tuple[int, ...], self._shape)

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
        return self.bitsize * prod(self.shape_symbolic)

    def adjoint(self) -> 'Register':
        """Return the 'adjoint' of this register by switching RIGHT and LEFT registers."""
        if self.side is Side.THRU:
            return self
        if self.side is Side.LEFT:
            return attrs.evolve(self, side=Side.RIGHT)
        if self.side is Side.RIGHT:
            return attrs.evolve(self, side=Side.LEFT)
        raise ValueError(f"Unknown side {self.side}")


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
    def build(cls, **registers: Union[int, sympy.Expr]) -> 'Signature':
        """Construct a Signature comprised of simple thru registers given the register bitsizes.

        Args:
            registers: keyword arguments mapping register name to bitsize. All registers
                will be 0-dimensional and THRU.
        """
        return cls(
            Register(name=k, dtype=QBit() if v == 1 else QAny(v)) for k, v in registers.items() if v
        )

    @classmethod
    def build_from_dtypes(cls, **registers: QDType) -> 'Signature':
        """Construct a Signature comprised of simple thru registers given the register dtypes.

        Args:
            registers: keyword arguments mapping register name to QDType. All registers
                will be 0-dimensional and THRU.
        """
        return cls(Register(name=k, dtype=v) for k, v in registers.items() if v.num_qubits)

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

    def n_qubits(self) -> int:
        """The number of qubits in the signature.

        If the signature has LEFT and RIGHT registers, the number of qubits in the signature
        is taken to be the greater of the number of left or right qubits. A bloq with this
        signature uses at least this many qubits.
        """
        left_size = ssum(reg.total_bits() for reg in self.lefts())
        right_size = ssum(reg.total_bits() for reg in self.rights())
        return smax(left_size, right_size)

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
