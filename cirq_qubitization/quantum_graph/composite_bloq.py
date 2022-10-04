from typing import Sequence, TYPE_CHECKING, Tuple

from cirq_qubitization.gate_with_registers import Registers
from cirq_qubitization.quantum_graph.bloq import Bloq

if TYPE_CHECKING:
    from cirq_qubitization.quantum_graph.quantum_graph import Wiring


class CompositeBloq(Bloq):
    def __init__(self, wires: Sequence['Wiring'], registers: Registers):
        self._wires = tuple(wires)
        self._registers = registers

    @property
    def registers(self) -> Registers:
        return self._registers

    @property
    def wires(self) -> Tuple['Wiring', ...]:
        return self._wires
