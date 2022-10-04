import abc
from typing import TYPE_CHECKING

from cirq_qubitization.gate_with_registers import Registers

if TYPE_CHECKING:
    from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq


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

    def decompose(self) -> 'CompositeBloq':
        return NotImplemented
