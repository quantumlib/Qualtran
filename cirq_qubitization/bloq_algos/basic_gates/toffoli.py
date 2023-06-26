from functools import cached_property
from typing import TYPE_CHECKING

from attrs import frozen
from cirq_ft.infra.t_complexity_protocol import TComplexity

from cirq_qubitization.bloq_algos.basic_gates.t_gate import TGate
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters

if TYPE_CHECKING:
    import cirq

    from cirq_qubitization.quantum_graph.cirq_conversion import CirqQuregT


@frozen
class ToffoliGate(Bloq):
    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(c0=1, c1=1, t=1)

    def t_complexity(self) -> 't_complexity_protocol.TComplexity':
        return t_complexity_protocol.t_complexity(cirq.CCNOT)

    def rough_decompose(self, mgr):
        return [(7, TGate())]
