import abc
from typing import TYPE_CHECKING, Sequence

from cirq_qubitization.gate_with_registers import Registers, GateWithRegisters

if TYPE_CHECKING:
    import cirq
    from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq


class Bloq(GateWithRegisters, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def registers(self) -> Registers:
        ...

    def pretty_name(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def decompose_bloq(self) -> 'CompositeBloq':
        ...

    # ----- cirq stuff -----

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        qubit_regs = self.registers.split_qubits(qubits)
        yield from self.decompose_bloq().to_cirq_circuit(**qubit_regs)

    def decompose_from_registers(self, **qubit_regs: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        yield from self.decompose_bloq().to_cirq_circuit(**qubit_regs)


class NoCirqEquivalent(NotImplementedError):
    """Raise this in `Bloq.on_registers` to signify that it should be omitted from Cirq circuits.

    For example, this would apply for qubit bookkeeping operations.
    """
