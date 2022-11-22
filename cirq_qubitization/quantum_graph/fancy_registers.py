import enum
import itertools
from collections import defaultdict
from typing import Dict, Iterable, List, overload, Tuple

import cirq
from attr import frozen


class Side(enum.Flag):
    LEFT = enum.auto()
    RIGHT = enum.auto()
    THRU = LEFT | RIGHT


@frozen
class FancyRegister:
    name: str
    bitsize: int
    wireshape: Tuple[int, ...] = tuple()
    side: Side = Side.THRU

    def wire_idxs(self) -> Iterable[Tuple[int, ...]]:
        yield from itertools.product(*[range(sh) for sh in self.wireshape])


class FancyRegisters:
    def __init__(self, registers: Iterable[FancyRegister]):
        self._registers = tuple(registers)
        self._lefts = {reg.name: reg for reg in self._registers if reg.side & Side.LEFT}
        self._rights = {reg.name: reg for reg in self._registers if reg.side & Side.RIGHT}

    @classmethod
    def build(cls, **registers: int) -> 'FancyRegisters':
        return cls(FancyRegister(name=k, bitsize=v) for k, v in registers.items())

    def lefts(self) -> Iterable[FancyRegister]:
        yield from (reg for reg in self._registers if reg.side & Side.LEFT)

    def rights(self) -> Iterable[FancyRegister]:
        yield from (reg for reg in self._registers if reg.side & Side.RIGHT)

    def get_left(self, name: str) -> FancyRegister:
        return self._lefts[name]

    def get_right(self, name: str) -> FancyRegister:
        return self._rights[name]

    def groups(self) -> Iterable[Tuple[str, 'FancyRegisters']]:
        groups = defaultdict(list)
        for soq in self._registers:
            groups[soq.name].append(soq)

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

    def __iter__(self) -> Iterable[FancyRegister]:
        yield from self._registers

    def __len__(self) -> int:
        return len(self._registers)

    def get_named_qubits(self) -> Dict[str, List[cirq.Qid]]:
        def qubits_for_reg(name: str, bitsize: int):
            return (
                [cirq.NamedQubit(f"{name}")]
                if bitsize == 1
                else cirq.NamedQubit.range(bitsize, prefix=name)
            )

        return {reg.name: qubits_for_reg(reg.name, reg.bitsize) for reg in self}

    def __hash__(self):
        return hash(self._registers)

    def __eq__(self, other) -> bool:
        return self._registers == other._registers
