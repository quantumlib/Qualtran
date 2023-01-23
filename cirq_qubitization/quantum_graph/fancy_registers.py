import enum
import itertools
from collections import defaultdict
from typing import Dict, Iterable, Iterator, overload, Tuple

import cirq
import numpy as np
from attr import frozen
from numpy.typing import NDArray


class Side(enum.Flag):
    """Denote LEFT, RIGHT, or THRU registers.

    LEFT registers serve as input lines (only) to the Bloq. RIGHT registers are output
    lines (only) from the Bloq. THRU registers are both input and output.

    Traditional unitary operations will have THRU registers that operate on a colleciton of
    qubits which are then made available to following operations. RIGHT and LEFT registers
    imply allocation, deallocation, or reshaping of the registers.
    """

    LEFT = enum.auto()
    RIGHT = enum.auto()
    THRU = LEFT | RIGHT


@frozen
class FancyRegister:
    """A quantum register.

    This sets a bloq's "function signature": its input and output types.

    Args:
        name: The string name of the register
        bitsize: The number of (qu)bits in the register.
        wireshape: A tuple of integer dimensions to declare a multidimensional register. The
            total number of bits is the product of entries in this tuple times `bitsize`.
        side: Whether this is a left, right, or thru register. See the documentation for `Side`
            for more information.
    """

    name: str
    bitsize: int
    wireshape: Tuple[int, ...] = tuple()
    side: Side = Side.THRU

    def wire_idxs(self) -> Iterable[Tuple[int, ...]]:
        """Iterate over all possible indices of a multidimensional register."""
        yield from itertools.product(*[range(sh) for sh in self.wireshape])


class FancyRegisters:
    """An ordered collection of `FancyRegister`.

    Args:
        registers: an iterable of the contained `FancyRegister`.
    """

    def __init__(self, registers: Iterable[FancyRegister]):
        self._registers = tuple(registers)
        self._lefts = {reg.name: reg for reg in self._registers if reg.side & Side.LEFT}
        self._rights = {reg.name: reg for reg in self._registers if reg.side & Side.RIGHT}

    @classmethod
    def build(cls, **registers: int) -> 'FancyRegisters':
        """Convenience method for building a collection of simple registers.

        Args:
            registers: keyword arguments mapping register name to bitsize. All registers
                will be 0-dimensional and THRU.
        """
        return cls(FancyRegister(name=k, bitsize=v) for k, v in registers.items())

    def lefts(self) -> Iterable[FancyRegister]:
        """Iterable over all registers that appear on the LEFT as input."""
        yield from self._lefts.values()

    def rights(self) -> Iterable[FancyRegister]:
        """Iterable over all registers that appear on the RIGHT as output."""
        yield from self._rights.values()

    def get_left(self, name: str) -> FancyRegister:
        """Get a left register by name."""
        return self._lefts[name]

    def get_right(self, name: str) -> FancyRegister:
        """Get a right register by name."""
        return self._rights[name]

    def groups(self) -> Iterable[Tuple[str, 'FancyRegisters']]:
        """Iterate over register groups by name.

        Registers with shared names (but differing `side` attributes) can be implicitly grouped.
        """
        groups = defaultdict(list)
        for reg in self._registers:
            groups[reg.name].append(reg)

        yield from ((name, FancyRegisters(grp)) for name, grp in groups.items())

    def __repr__(self):
        return f'FancyRegisters({repr(self._registers)})'

    @overload
    def __getitem__(self, key: int) -> FancyRegister:
        pass

    @overload
    def __getitem__(self, key: str) -> FancyRegister:
        pass

    @overload
    def __getitem__(self, key: slice) -> 'FancyRegisters':
        pass

    def __getitem__(self, key):
        if isinstance(key, slice):
            return FancyRegisters(self._registers[key])
        elif isinstance(key, int):
            return self._registers[key]
        elif isinstance(key, str):
            left = self._lefts[key]
            right = self._rights[key]
            if left != right:
                raise KeyError(f"`{key}` is not a thru register and cannot be indexed by name")
            return left
        else:
            raise IndexError(f"key {key} must be of the type str/int/slice.")

    def __contains__(self, item: FancyRegister) -> bool:
        return item in self._registers

    def __iter__(self) -> Iterator[FancyRegister]:
        yield from self._registers

    def __len__(self) -> int:
        return len(self._registers)

    def get_named_qubits(self) -> Dict[str, NDArray[cirq.Qid]]:
        def _qubit_array(reg: FancyRegister):
            qubits = np.empty(reg.wireshape + (reg.bitsize,), dtype=object)
            for ii in reg.wire_idxs():
                for j in range(reg.bitsize):
                    qubits[ii + (j,)] = cirq.NamedQubit(
                        f'{reg.name}[{", ".join(str(i) for i in ii+(j,))}]'
                    )
            return qubits

        def _qubits_for_reg(reg: FancyRegister):
            if reg.wireshape:
                return _qubit_array(reg)

            return (
                [cirq.NamedQubit(f"{reg.name}")]
                if reg.bitsize == 1
                else cirq.NamedQubit.range(reg.bitsize, prefix=reg.name)
            )

        return {reg.name: _qubits_for_reg(reg) for reg in self}

    def __hash__(self):
        return hash(self._registers)

    def __eq__(self, other) -> bool:
        return self._registers == other._registers
