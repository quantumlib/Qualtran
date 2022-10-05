import abc
from typing import TYPE_CHECKING, Dict, Sequence, Union

import cirq

from cirq_qubitization.gate_with_registers import Registers

if TYPE_CHECKING:
    from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq, CompositeBloqBuilder
    from cirq_qubitization.quantum_graph.quantum_graph import Soquet


class Bloq(cirq.Gate, metaclass=abc.ABCMeta):
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

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', **soquets: 'Soquet'
    ) -> Dict[str, 'Soquet']:
        return NotImplemented

    def to_composite_bloq(self) -> 'CompositeBloq':
        from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder

        bb = CompositeBloqBuilder(self.registers)
        ret_soqs = self.build_composite_bloq(bb=bb, **bb.initial_soquets())
        if ret_soqs is NotImplemented:
            raise NotImplementedError(f"Cannot decompose {self}.")

        return bb.finalize(**ret_soqs)

    ## ----- cirq stuff

    def _num_qubits_(self) -> int:
        return self.registers.bitsize

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        qubit_regs = self.registers.split_qubits(qubits)
        yield from self.to_composite_bloq().to_cirq_circuit(**qubit_regs)

    def on_registers(self, **qubit_regs: Union[cirq.Qid, Sequence[cirq.Qid]]) -> cirq.Operation:
        return self.on(*self.registers.merge_qubits(**qubit_regs))

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.registers:
            wire_symbols += [reg.name] * reg.bitsize

        wire_symbols[0] = self.__class__.__name__
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)


class NoCirqEquivalent(NotImplementedError):
    pass
