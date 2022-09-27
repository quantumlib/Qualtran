from dataclasses import dataclass
from functools import cached_property
from typing import Sequence, Union, List

import cirq

from cirq_qubitization import MultiTargetCSwap
from cirq_qubitization.atoms import Split, Join
from cirq_qubitization.gate_with_registers import GateWithRegisters, Registers


@dataclass(frozen=True)
class ScaledModAdd(GateWithRegisters):
    xy_bitsize: int
    scaling_constant: int

    @cached_property
    def registers(self) -> Registers:
        return Registers.build(
            control=1,
            src=self.xy_bitsize,
            dest=self.xy_bitsize,
        )

    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        raise NotImplementedError()


@dataclass(frozen=True)
class SingleControlModMultiply(GateWithRegisters):
    x_bitsize: int
    mul_constant: int

    @cached_property
    def registers(self) -> Registers:
        return Registers.build(
            control=1,
            x=self.x_bitsize
        )

    def pretty_name(self) -> str:
        return '× (mod)'

    def decompose_from_registers(self, control: Sequence[cirq.Qid],
                                 x: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        control, = control
        y = cirq.NamedQubit.range(self.x_bitsize, prefix='y')
        yield ScaledModAdd(xy_bitsize=self.x_bitsize,
                           scaling_constant=self.mul_constant).on_registers(control=[control],
                                                                            src=x, dest=y)

        # TODO: aren't these ints?
        yield ScaledModAdd(xy_bitsize=self.x_bitsize,
                           scaling_constant=-1 / self.mul_constant).on_registers(control=[control],
                                                                                 src=y, dest=x)
        yield MultiTargetCSwap(target_bitsize=self.x_bitsize).on_registers(control=[control],
                                                                           target_x=x, target_y=y)
        # // dealloc y

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=['@'] + ['mm'] * self.x_bitsize
        )





@dataclass(frozen=True)
class ModMultiply(GateWithRegisters):
    exponent_bitsize: int
    x_bitsize: int
    mul_constant: int
    mod_N: int

    @cached_property
    def registers(self) -> Registers:
        return Registers.build(
            exponent=self.exponent_bitsize,
            x=self.x_bitsize
        )

    def compute_graph(self):
        from cirq_qubitization.quantum_graph import Wire, LeftDangle, RightDangle
        nodes: List[GateWithRegisters] = []
        edges: List[Wire] = []

        split = Split(self.exponent_bitsize)
        nodes.append(split)
        edges.append(Wire(LeftDangle, 'exponent', split, 'x'))

        join = Join(self.exponent_bitsize)

        prev_node = LeftDangle
        for j in range(self.exponent_bitsize - 1, 0 - 1, -1):
            c = self.mul_constant ** 2 ** j % self.mod_N
            node = SingleControlModMultiply(
                x_bitsize=self.x_bitsize,
                mul_constant=c)
            nodes.append(node)
            edges.append(Wire(split, f'y{j}', node, 'control'))
            edges.append(Wire(node, 'control', join, f'x{j}'))

            edges.append(Wire(prev_node, 'x', node, 'x'))
            prev_node = node

        edges.append(Wire(prev_node, 'x', RightDangle, 'x'))

        nodes.append(join)
        edges.append(Wire(join, 'x', RightDangle, 'exponent'))
        return nodes, edges

    def decompose_from_registers(
            self, exponent: Sequence[cirq.Qid], x: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        for j in range(self.exponent_bitsize - 1, 0 - 1, -1):
            c = self.mul_constant ** 2 ** j % self.mod_N
            yield SingleControlModMultiply(
                x_bitsize=self.x_bitsize,
                mul_constant=c).on_registers(control=exponent[j:j+1], x=x)
