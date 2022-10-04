import abc
from typing import TYPE_CHECKING, Dict

from cirq_qubitization.gate_with_registers import Registers

if TYPE_CHECKING:
    from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq
    from cirq_qubitization.quantum_graph.bloq_builder import BloqBuilder
    from cirq_qubitization.quantum_graph.quantum_graph import Soquet


class Bloq(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def registers(self) -> Registers:
        ...

    def pretty_name(self) -> str:
        return self.__class__.__name__

    def short_name(self) -> str:
        name = self.pretty_name()
        if len(name) <= 6:
            return name

        return name[:4] + '..'

    def build_decomposition(self, bb: 'BloqBuilder', **soquets: 'Soquet') -> Dict[str, 'Soquet']:
        return NotImplemented

    def decompose(self) -> 'CompositeBloq':
        from cirq_qubitization.quantum_graph.bloq_builder import BloqBuilder

        bb = BloqBuilder(self.registers)
        stuff = self.build_decomposition(bb=bb, **bb.initial_soquets())
        return bb.finalize(**stuff)
