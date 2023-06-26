from functools import cached_property
from typing import Dict, Tuple, TYPE_CHECKING

from attrs import frozen
from cirq_ft.infra.t_complexity_protocol import TComplexity

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters
from cirq_qubitization.quantum_graph.util_bloqs import ArbitraryClifford

if TYPE_CHECKING:
    import cirq

    from cirq_qubitization.quantum_graph.cirq_conversion import CirqQuregT


@frozen
class HGate(Bloq):
    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(q=1)

    def t_complexity(self) -> 'TComplexity':
        return TComplexity(t=0, clifford=1)

    def rough_decompose(self, mgr):
        return [(1, ArbitraryClifford(n=1))]

    def as_cirq_op(self, q: 'CirqQuregT') -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.H(q), {'q': [q]}
