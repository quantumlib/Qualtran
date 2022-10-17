import abc
from typing import TYPE_CHECKING, Dict, Sequence

import cirq


if TYPE_CHECKING:
    from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq, CompositeBloqBuilder
    from cirq_qubitization.quantum_graph.fancy_registers import Soquets
    from cirq_qubitization.quantum_graph.quantum_graph import Wire


class Bloq(cirq.Gate, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def soquets(self) -> 'Soquets':
        ...

    def pretty_name(self) -> str:
        return self.__class__.__name__

    def short_name(self) -> str:
        name = self.pretty_name()
        if len(name) <= 6:
            return name

        return name[:4] + '..'

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', **soquets: 'Wire'
    ) -> Dict[str, 'Wire']:
        return NotImplemented

    def decompose_bloq(self) -> 'CompositeBloq':
        from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder

        bb = CompositeBloqBuilder(self.soquets)
        ret_soqs = self.build_composite_bloq(bb=bb, **bb.initial_soquets())
        if ret_soqs is NotImplemented:
            raise NotImplementedError(f"Cannot decompose {self}.")

        return bb.finalize(**ret_soqs)

    def as_composite_bloq(self) -> 'CompositeBloq':
        from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder

        bb = CompositeBloqBuilder(self.soquets)
        ret_soqs_tuple = bb.add(self, **bb.initial_soquets())
        assert len(list(self.soquets.rights())) == len(ret_soqs_tuple)
        ret_soqs = {reg.name: v for reg, v in zip(self.soquets.rights(), ret_soqs_tuple)}
        return bb.finalize(**ret_soqs)

    # ----- cirq stuff -----

    def _num_qubits_(self) -> int:
        return self.soquets.total_size

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        qubit_regs = self.soquets.split_qubits(qubits)
        yield from self.decompose_bloq().to_cirq_circuit(**qubit_regs)

    def on_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.Operation:
        return self.on(*self.soquets.merge_qubits(**qubit_regs))

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.soquets:
            wire_symbols += [reg.name] * reg.bitsize

        wire_symbols[0] = self.__class__.__name__
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)


class NoCirqEquivalent(NotImplementedError):
    """Raise this in `Bloq.on_registers` to signify that it should be omitted from Cirq circuits.

    For example, this would apply for qubit bookkeeping operations.
    """
