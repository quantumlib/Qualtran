from functools import cached_property
from typing import Dict, Sequence, TYPE_CHECKING

import cirq
import numpy as np
from attrs import frozen
from numpy.typing import NDArray

from cirq_qubitization.cirq_algos import multi_control_multi_target_pauli
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder, SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters
from cirq_qubitization.quantum_graph.musical_score import directional_text_box
from cirq_qubitization.quantum_graph.quantum_graph import Soquet
from cirq_qubitization.t_complexity_protocol import TComplexity

if TYPE_CHECKING:
    from cirq_qubitization.quantum_graph.classical_sim import ClassicalValT


@frozen
class SetConstant(Bloq):
    k: int
    bitsize: int

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(x=self.bitsize)

    def short_name(self) -> str:
        return f'{self.k}'

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        assert soq.reg.name == 'x'
        return directional_text_box(f'{self.k}', side=soq.reg.side)
