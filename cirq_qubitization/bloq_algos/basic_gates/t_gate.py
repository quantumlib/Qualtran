from functools import cached_property
from typing import Dict, Tuple, TYPE_CHECKING

from attrs import frozen
from cirq_ft import TComplexity

from cirq_qubitization import Bloq, FancyRegisters

if TYPE_CHECKING:
    import cirq

    from cirq_qubitization.quantum_graph.cirq_conversion import CirqQuregT


@frozen
class TGate(Bloq):
    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(q=1)

    def t_complexity(self) -> 'TComplexity':
        return TComplexity(t=1)

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.T(q), {'q': [q]}
