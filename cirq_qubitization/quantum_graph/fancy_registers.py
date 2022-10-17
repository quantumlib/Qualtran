import abc
import itertools
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, overload, Tuple
from typing import Optional

import enum
import cirq
from attrs import frozen

assert sys.version_info > (3, 6), "https://docs.python.org/3/whatsnew/3.6.html#whatsnew36-pep468"


class Side(enum.Flag):
    LEFT = enum.auto()
    RIGHT = enum.auto()
    THRU = LEFT | RIGHT


class FancyRegister(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def bitsize(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def wireshape(self) -> Tuple[int, ...]:
        ...

    def wire_idxs(self):
        yield from itertools.product(*[range(sh) for sh in self.wireshape])

    @property
    @abc.abstractmethod
    def side(self) -> Side:
        ...


@frozen
class CustomRegister(FancyRegister):
    name: str
    bitsize: int
    wireshape: Tuple[int, ...]
    side: Side


@frozen
class ThruRegister(FancyRegister):
    name: str
    bitsize: int

    @property
    def wireshape(self) -> Tuple[int, ...]:
        return tuple()

    @property
    def side(self) -> Side:
        return Side.THRU


# @frozen
# class SplitRegister(FancyRegister):
#     name: str
#     n: int
#
#     @property
#     def left_shape(self) -> Tuple[int, ...]:
#         return (self.n,)
#
#     @property
#     def right_shape(self) -> Tuple[int, ...]:
#         return (self.n, 1)
#
#
# @frozen
# class JoinRegister(FancyRegister):
#     name: str
#     n: int
#
#     @property
#     def left_shape(self) -> Tuple[int, ...]:
#         return (self.n, 1)
#
#     @property
#     def right_shape(self) -> Tuple[int, ...]:
#         return (self.n,)
#
#
# @frozen
# class AllocRegister(FancyRegister):
#     name: str
#     n: int
#
#     @property
#     def left_shape(self) -> Tuple[int, ...]:
#         return tuple()
#
#     @property
#     def right_shape(self) -> Tuple[int, ...]:
#         return (self.n,)
#
#
# @frozen
# class ApplyFRegister(IThruRegister):
#     name: str
#     bitsize: int
#     out_name: str
#     in_text: Optional[str] = None
#     out_text: Optional[str] = None
#
#     @property
#     def shape(self) -> Tuple[int, ...]:
#         return (self.bitsize,)


class Soquets:
    def __init__(self, registers: Iterable[FancyRegister]):
        self._registers = tuple(registers)
        self._lefts = {reg.name: reg for reg in self._registers if reg.side & Side.LEFT}
        self._rights = {reg.name: reg for reg in self._registers if reg.side & Side.RIGHT}
        # self._register_dict = {r.name: r for r in self._registers}
        # if len(self._registers) != len(self._register_dict):
        #     raise ValueError("Please provide unique register names.")

    def lefts(self) -> Iterable[FancyRegister]:
        yield from (reg for reg in self._registers if reg.side & Side.LEFT)

    def rights(self) -> Iterable[FancyRegister]:
        yield from (reg for reg in self._registers if reg.side & Side.RIGHT)

    def get_left(self, name: str) -> FancyRegister:
        return self._lefts[name]

    def get_right(self, name: str) -> FancyRegister:
        return self._rights[name]

    def groups(self):
        groups = defaultdict(list)
        for soq in self._registers:
            groups[soq.name].append(soq)

        return [(name, Soquets(grp)) for name, grp in groups.items()]

    def __repr__(self):
        return f'Soquets({repr(self._registers)})'

    @overload
    def __getitem__(self, key: int) -> FancyRegister:
        pass

    @overload
    def __getitem__(self, key: str) -> FancyRegister:
        pass

    @overload
    def __getitem__(self, key: slice) -> 'Soquets':
        pass

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Soquets(self._registers[key])
        elif isinstance(key, int):
            return self._registers[key]
        elif isinstance(key, str):
            # TODO: check
            left = self._lefts[key]
            right = self._rights[key]
            if left != right:
                raise KeyError(f"Not a thru register for {key}")
            return left
        else:
            raise IndexError(f"key {key} must be of the type str/int/slice.")

    def __contains__(self, item: str) -> bool:
        return item in self._register_dict

    def __iter__(self):
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

    def __eq__(self, other) -> bool:
        return self._registers == other._registers
