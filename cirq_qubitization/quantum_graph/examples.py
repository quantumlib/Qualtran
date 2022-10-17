from functools import cached_property
from typing import Dict, List, Sequence, Tuple

import cirq
import numpy as np
from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder
from cirq_qubitization.quantum_graph.fancy_registers import (
    ThruRegister,
    Soquets,
    CustomRegister,
    Side,
)
from cirq_qubitization.quantum_graph.quantum_graph import Wire
from cirq_qubitization.quantum_graph.util_bloqs import Partition


@frozen
class SplitJoin(Bloq):
    nn: int

    @cached_property
    def soquets(self) -> Soquets:
        return Soquets([ThruRegister('xx', self.nn)])

    def build_composite_bloq(self, bb: 'CompositeBloqBuilder', *, xx: 'Wire') -> Dict[str, 'Wire']:
        xxs = bb.split(xx, self.nn)
        xx2 = bb.join(xxs)
        return {'xx': xx2}


@frozen
class And(Bloq):
    cv1: int = 1
    cv2: int = 1
    adjoint: bool = False

    @cached_property
    def soquets(self) -> Soquets:
        return Soquets(
            [
                ThruRegister('control', 2),
                CustomRegister('target', 1, wireshape=tuple(), side=Side.RIGHT),
            ]
        )

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        controls = ["(0)", "@"]
        target = "And†" if self.adjoint else "And"
        wire_symbols = [controls[c] for c in (self.cv1, self.cv2)]
        wire_symbols += [target]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _has_unitary_(self) -> bool:
        return not self.adjoint


@frozen
class MultiAnd(Bloq):
    cvs: Tuple[int, ...]
    adjoint: bool = False

    @cached_property
    def soquets(self) -> Soquets:
        return Soquets(
            [
                ThruRegister('control', len(self.cvs)),
                CustomRegister('ancilla', len(self.cvs) - 2, wireshape=tuple(), side=Side.RIGHT),
                CustomRegister('target', 1, wireshape=tuple(), side=Side.RIGHT),
            ]
        )

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        controls = ["(0)", "@"]
        target = "And†" if self.adjoint else "And"
        wire_symbols = [controls[c] for c in self.cvs]
        wire_symbols += [target]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _has_unitary_(self) -> bool:
        return not self.adjoint

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', *, control: 'Wire'
    ) -> Dict[str, 'Wire']:
        if len(self.cvs) <= 2:
            raise ValueError()
            control, target = bb.add(And(*self.cvs, adjoint=self.adjoint), control=control)
            return {'control': control, 'target': target}

        assert not self.adjoint

        assert control.soq.bitsize == self.soquets['control'].bitsize
        actrl, sub_ctrl = bb.add(Partition(sizes=(2, len(self.cvs) - 2)), x=control)
        acvs, sub_cvs = self.cvs[:2], self.cvs[2:]
        actrl, new_anc = bb.add(And(*acvs, adjoint=self.adjoint), control=actrl)

        sub_ctrls = bb.join([sub_ctrl[tuple()], new_anc[tuple()]])
        sub_cvs = (1, *sub_cvs)

        if len(sub_cvs) == 2:
            new_ctrls, trg = bb.add(And(*sub_cvs), control=sub_ctrls)
            sc1, ac1 = bb.add(Partition(sizes=(1, 1)), x=new_ctrls)
            all_ctrls = bb.join(np.concatenate((actrl[np.newaxis], sc1[np.newaxis])))
            return {'control': all_ctrls, 'ancilla': ac1, 'target': trg}

        new_ctrls, anc, trg = bb.add(MultiAnd(sub_cvs), control=sub_ctrls)
        sc1, ac1 = bb.add(Partition(sizes=(len(sub_cvs) - 1, 1)), x=new_ctrls)
        anc = bb.join([ac1[tuple()], anc[tuple()]])
        all_ctrls = bb.join(np.concatenate((actrl[np.newaxis], sc1[np.newaxis])))
        return {'control': all_ctrls, 'ancilla': anc, 'target': trg}


@frozen
class SingleControlModMultiply(Bloq):
    x_bitsize: int
    mul_constant: int

    def short_name(self) -> str:
        return 'SingleCtrlMM'

    @cached_property
    def soquets(self) -> Soquets:
        return Soquets(
            [
                ThruRegister('control', 1),
                ThruRegister('x', self.x_bitsize),
                # ApplyFRegister(
                #     'x', self.x_bitsize, 'x_out', in_text='x', out_text=f'{self.mul_constant}*x'
                # ),
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
    def soquets(self) -> Soquets:
        return Soquets(
            [ThruRegister('exponent', self.exponent_bitsize), ThruRegister('x', self.x_bitsize)]
        )

    def build_composite_bloq(
        self, bb: CompositeBloqBuilder, *, exponent: Wire, x: Wire
    ) -> Dict[str, Wire]:
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
    def soquets(self) -> Soquets:
        return Soquets([ThruRegister('control', 1), ThruRegister('target', 1)])

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
    def soquets(self) -> Soquets:
        return Soquets(
            [
                ThruRegister('control', 1),
                CustomRegister('target', bitsize=self.n_targets, wireshape=tuple(), side=Side.LEFT),
                CustomRegister('target', bitsize=1, wireshape=(self.n_targets,), side=Side.RIGHT),
            ]
        )

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', control: Wire, target: Wire
    ) -> Dict[str, 'Wire']:
        targets = list(bb.split(target, self.n_targets))
        for i in range(self.n_targets):
            control, trg = bb.add(CNOT(), control=control, target=targets[i])
            targets[i] = trg

        return {'control': control, 'target': targets}
