from functools import cached_property
from typing import Dict, List, Sequence, Tuple

import cirq
import numpy as np
from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq, CompositeBloqBuilder
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import Soquet
from cirq_qubitization.quantum_graph.util_bloqs import Join, Partition, Split, Unpartition


@frozen
class TestBloq(Bloq):
    """A test bloq with one control and one target."""

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(control=1, target=1)

    def on_registers(
        self, control: Sequence[cirq.Qid], target: Sequence[cirq.Qid]
    ) -> cirq.Operation:
        (control,) = control
        (target,) = target
        return cirq.CNOT(control, target)



@frozen
class SplitJoin(Bloq):
    nn: int

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters([FancyRegister('xx', self.nn)])

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', *, xx: 'Soquet'
    ) -> Dict[str, 'Soquet']:
        xxs = bb.split(xx, self.nn)
        xx2 = bb.join(xxs)
        return {'xx': xx2}


@frozen
class And(Bloq):
    cv1: int = 1
    cv2: int = 1
    adjoint: bool = False

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [
                FancyRegister('control', 2),
                FancyRegister('target', 1, wireshape=tuple(), side=Side.RIGHT),
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

    def on_registers(
        self, *, control: Sequence['cirq.Qid'], target: Sequence['cirq.Qid']
    ) -> 'cirq.OP_TREE':
        return cirq.CCNOT(control + target)


@frozen
class MultiAnd(Bloq):
    cvs: Tuple[int, ...]
    adjoint: bool = False

    @cached_property
    def registers(self) -> FancyRegisters:
        # TODO: respond to "adjoint"
        return FancyRegisters(
            [
                FancyRegister('control', len(self.cvs)),
                FancyRegister('ancilla', len(self.cvs) - 2, wireshape=tuple(), side=Side.RIGHT),
                FancyRegister('target', 1, wireshape=tuple(), side=Side.RIGHT),
            ]
        )

    def short_name(self) -> str:
        return '*And'

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        controls = ["(0)", "@"]
        target = "And†" if self.adjoint else "And"
        wire_symbols = [controls[c] for c in self.cvs]
        wire_symbols += [target]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _has_unitary_(self) -> bool:
        return not self.adjoint

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', *, control: 'Soquet'
    ) -> Dict[str, 'Soquet']:
        if len(self.cvs) <= 2:
            raise ValueError()

        assert not self.adjoint

        actrl, sub_ctrl = bb.add(Partition(sizes=(2, len(self.cvs) - 2)), x=control)
        acvs, sub_cvs = self.cvs[:2], self.cvs[2:]
        actrl, new_anc = bb.add(And(*acvs, adjoint=self.adjoint), control=actrl)

        (sub_ctrls,) = bb.add(Unpartition(sizes=(len(self.cvs) - 2, 1)), y0=sub_ctrl, y1=new_anc)
        sub_cvs = (1, *sub_cvs)

        if len(sub_cvs) == 2:
            new_ctrls, trg = bb.add(And(*sub_cvs), control=sub_ctrls)
            sc1, ac1 = bb.add(Partition(sizes=(1, 1)), x=new_ctrls)
            (all_ctrls,) = bb.add(Unpartition(sizes=(1, 1)), y0=actrl, y1=sc1)
            return {'control': all_ctrls, 'ancilla': ac1, 'target': trg}

        new_ctrls, anc, trg = bb.add(MultiAnd(sub_cvs), control=sub_ctrls)
        sc1, ac1 = bb.add(Partition(sizes=(len(sub_cvs) - 1, 1)), x=new_ctrls)
        (anc,) = bb.add(Unpartition(sizes=(1, anc.reg.bitsize)), y0=ac1, y1=anc)
        (all_ctrls,) = bb.add(Unpartition(sizes=(2, sc1.reg.bitsize)), y0=actrl, y1=sc1)
        return {'control': all_ctrls, 'ancilla': anc, 'target': trg}


@frozen
class SingleControlModMultiply(Bloq):
    x_bitsize: int
    mul_constant: int

    def short_name(self) -> str:
        return 'SingleCtrlMM'

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters([FancyRegister('control', 1), FancyRegister('x', self.x_bitsize)])

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> List[str]:
        return ['@'] + ['x'] * self.x_bitsize

    def on_registers(
        self, control: Sequence['cirq.Qid'], x: Sequence['cirq.Qid']
    ) -> 'cirq.OP_TREE':
        yield GateWithRegistersWrapper(bloq=self).on_registers(control=control, x=x)


@frozen
class ModMultiply(Bloq):
    exponent_bitsize: int
    x_bitsize: int
    mul_constant: int
    mod_N: int

    def short_name(self) -> str:
        return 'ModMult'

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(exponent=self.exponent_bitsize, x=self.x_bitsize)

    def build_composite_bloq(
        self, bb: CompositeBloqBuilder, *, exponent: Soquet, x: Soquet
    ) -> Dict[str, Soquet]:
        ctls = bb.split(exponent, self.exponent_bitsize)

        for j in range(self.exponent_bitsize - 1, 0 - 1, -1):
            c = self.mul_constant**2**j % self.mod_N
            ctls[j], x = bb.add(
                SingleControlModMultiply(x_bitsize=self.x_bitsize, mul_constant=c),
                control=ctls[j],
                x=x,
            )
            # ctls[j], x = SingleControlModMultiply(x_bitsize=self.x_bitsize, mul_constant=c).wire_up(
            #     control=ctls[j], x=x
            # )

        return {'exponent': bb.join(ctls), 'x': x}


class CNOT(Bloq):
    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(control=1, target=1)

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
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [
                FancyRegister('control', 1),
                FancyRegister('target', bitsize=self.n_targets, wireshape=tuple(), side=Side.LEFT),
                FancyRegister('target', bitsize=1, wireshape=(self.n_targets,), side=Side.RIGHT),
            ]
        )

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', control: Soquet, target: Soquet
    ) -> Dict[str, 'Soquet']:
        targets = list(bb.split(target, self.n_targets))
        for i in range(self.n_targets):
            control, trg = bb.add(CNOT(), control=control, target=targets[i])
            targets[i] = trg

        return {'control': control, 'target': targets}
