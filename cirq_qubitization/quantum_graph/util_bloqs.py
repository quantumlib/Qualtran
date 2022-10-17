from dataclasses import dataclass
from functools import cached_property
from typing import Union, Sequence, TYPE_CHECKING, Dict, Tuple

from cirq_qubitization.quantum_graph.bloq import Bloq, NoCirqEquivalent
from cirq_qubitization.quantum_graph.fancy_registers import Soquets, CustomRegister, Side

from attrs import frozen

if TYPE_CHECKING:
    import cirq


@dataclass(frozen=True)
class Split(Bloq):
    n: int

    @cached_property
    def soquets(self) -> Soquets:
        return Soquets(
            [
                CustomRegister(name='sss', bitsize=self.n, wireshape=tuple(), side=Side.LEFT),
                CustomRegister(name='sss', bitsize=1, wireshape=(self.n,), side=Side.RIGHT),
            ]
        )

    def on_registers(
        self, **qubit_regs: Union['cirq.Qid', Sequence['cirq.Qid']]
    ) -> 'cirq.GateOperation':
        raise NoCirqEquivalent()


@dataclass(frozen=True)
class Partition(Bloq):
    sizes: Tuple[int, ...]

    @cached_property
    def soquets(self) -> Soquets:
        tot = sum(self.sizes)
        return Soquets(
            [CustomRegister(name='x', bitsize=tot, wireshape=tuple(), side=Side.LEFT)]
            + [
                CustomRegister(name=f'y{i}', bitsize=n, wireshape=tuple(), side=Side.RIGHT)
                for i, n in enumerate(self.sizes)
            ]
        )

    def on_registers(
        self, **qubit_regs: Union['cirq.Qid', Sequence['cirq.Qid']]
    ) -> 'cirq.GateOperation':
        raise NoCirqEquivalent()


@dataclass(frozen=True)
class Join(Bloq):
    sizes: Tuple[int, ...]

    @cached_property
    def soquets(self) -> Soquets:
        tot = sum(self.sizes)
        return Soquets(
            [
                CustomRegister(name=f'x{i}', bitsize=n, wireshape=tuple(), side=Side.LEFT)
                for i, n in enumerate(self.sizes)
            ]
            + [CustomRegister(name='y', bitsize=tot, wireshape=tuple(), side=Side.RIGHT)]
        )

    def on_registers(
        self, **qubit_regs: Union['cirq.Qid', Sequence['cirq.Qid']]
    ) -> 'cirq.GateOperation':
        raise NoCirqEquivalent()


@frozen
class FirstAndRest(Bloq):
    insize: int

    @cached_property
    def soquets(self) -> Soquets:
        return Soquets(
            [
                CustomRegister('x', bitsize=self.insize, wireshape=tuple(), side=Side.LEFT),
                CustomRegister('first', bitsize=1, wireshape=tuple(), side=Side.RIGHT),
                CustomRegister('rest', bitsize=self.insize - 1, wireshape=tuple(), side=Side.RIGHT),
            ]
        )

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', **soquets: 'Wire'
    ) -> Dict[str, 'Wire']:
        first, *rest = bb.split(soquets['x'], self.insize)
        return {'first': first, 'rest': bb.join(rest)}
