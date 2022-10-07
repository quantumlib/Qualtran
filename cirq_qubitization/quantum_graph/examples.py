from functools import cached_property
from typing import Dict, List, Sequence

import cirq
from attrs import frozen

from cirq_qubitization.gate_with_registers import Registers, ThruRegister
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder
from cirq_qubitization.quantum_graph.fancy_registers import ApplyFRegister, SplitRegister
from cirq_qubitization.quantum_graph.quantum_graph import Soquet


@frozen
class SingleControlModMultiply(Bloq):
    x_bitsize: int
    mul_constant: int

    def short_name(self) -> str:
        return 'SingleCtrlMM'

    @cached_property
    def registers(self) -> Registers:
        return Registers(
            [
                ThruRegister('control', 1),
                ApplyFRegister(
                    'x', self.x_bitsize, 'x_out', in_text='x', out_text=f'{self.mul_constant}*x'
                ),
            ]
        )

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> List[str]:
        return ['@'] + ['x'] * self.x_bitsize


@frozen
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


class CNOT(Bloq):
    @cached_property
    def registers(self) -> Registers:
        return Registers.build(control=1, target=1)

    def on_registers(
        self, control: Sequence[cirq.Qid], target: Sequence[cirq.Qid]
    ) -> cirq.Operation:
        (control,) = control
        (target,) = target
        return cirq.CNOT(control, target)


@frozen
class MultiCNOTSplit(Bloq):
    n_targets: int

    @cached_property
    def registers(self) -> Registers:
        return Registers([ThruRegister('control', 1), SplitRegister('target', self.n_targets)])

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', control: Soquet, target: Soquet
    ) -> Dict[str, 'Soquet']:
        targets = list(bb.split(target, self.n_targets))
        for i in range(self.n_targets):
            control, trg = bb.add(CNOT(), control=control, target=targets[i])
            targets[i] = trg

        targ_outs = dict(zip(self.registers['target'].right_names(), targets))
        return {'control': control, **targ_outs}
