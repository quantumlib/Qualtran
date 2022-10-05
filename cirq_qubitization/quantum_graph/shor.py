from dataclasses import dataclass
from functools import cached_property
from typing import Dict, TYPE_CHECKING, List

from cirq_qubitization.gate_with_registers import Registers, Register
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder
from cirq_qubitization.quantum_graph.fancy_registers import ApplyFRegister
from cirq_qubitization.quantum_graph.quantum_graph import Soquet

if TYPE_CHECKING:
    import cirq


@dataclass(frozen=True)
class SingleControlModMultiply(Bloq):
    x_bitsize: int
    mul_constant: int

    def short_name(self) -> str:
        return 'SingleCtrlMM'

    @cached_property
    def registers(self) -> Registers:
        return Registers(
            [
                Register('control', 1),
                ApplyFRegister(
                    'x', self.x_bitsize, 'x_out', in_text='x', out_text=f'{self.mul_constant}*x'
                ),
            ]
        )

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> List[str]:
        return ['@'] + ['x'] * self.x_bitsize


@dataclass(frozen=True)
class ModMultiply(Bloq):
    exponent_bitsize: int
    x_bitsize: int
    mul_constant: int
    mod_N: int

    def short_name(self) -> str:
        return 'ModMult'

    @cached_property
    def registers(self) -> Registers:
        return Registers.build(exponent=self.exponent_bitsize, x=self.x_bitsize)

    def build_composite_bloq(
        self, bb: CompositeBloqBuilder, *, exponent: Soquet, x: Soquet
    ) -> Dict[str, Soquet]:
        ctls = bb.split(exponent, self.exponent_bitsize)
        out_ctls = []

        for j in range(self.exponent_bitsize - 1, 0 - 1, -1):
            c = self.mul_constant**2**j % self.mod_N
            ctl, x = bb.add(
                SingleControlModMultiply(x_bitsize=self.x_bitsize, mul_constant=c),
                control=ctls[j],
                x=x,
            )
            out_ctls.append(ctl)

        return {'exponent': bb.join(out_ctls), 'x': x}
