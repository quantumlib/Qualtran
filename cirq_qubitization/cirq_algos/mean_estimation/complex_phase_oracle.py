from functools import cached_property
from typing import Sequence

import cirq
from attrs import frozen

from cirq_qubitization import cirq_infra
from cirq_qubitization.cirq_algos import select_and_prepare as sp
from cirq_qubitization.cirq_algos.mean_estimation import arctan


@frozen
class ComplexPhaseOracle(cirq_infra.GateWithRegisters):
    r"""Applies $ROT_{y}|l>|garbage_{l}> = exp(i * -2arctan{y_{l}})|l>|garbage_{l}>$.

    TODO: This currently assumes that the random variable `y_{l}` only takes integer
    values. This constraint can be removed by using a standardized floating point to
    binary encoding, like IEEE 754, to encode arbitrary floats in the binary target
    register and use them to compute the more accurate $-2arctan{y_{l}}$ for any arbitrary
    $y_{l}$.
    """

    encoder: sp.SelectOracle
    arctan_bitsize: int = 32

    @cached_property
    def control_registers(self) -> cirq_infra.Registers:
        return self.encoder.control_registers

    @cached_property
    def selection_registers(self) -> cirq_infra.SelectionRegisters:
        return self.encoder.selection_registers

    @cached_property
    def registers(self) -> cirq_infra.Registers:
        return cirq_infra.Registers([*self.control_registers, *self.selection_registers])

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **qubit_regs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        qm = context.qubit_manager
        target_reg = {reg.name: qm.qalloc(reg.bitsize) for reg in self.encoder.target_registers}
        target_qubits = self.encoder.target_registers.merge_qubits(**target_reg)
        encoder_op = self.encoder.on_registers(**qubit_regs, **target_reg)

        arctan_sign, arctan_target = qm.qalloc(1), qm.qalloc(self.arctan_bitsize)
        arctan_op = arctan.ArcTan(len(target_qubits), self.arctan_bitsize).on(
            *target_qubits, *arctan_sign, *arctan_target
        )

        yield encoder_op
        yield arctan_op
        for i, q in enumerate(arctan_target):
            yield (cirq.Z(q) ** (1 / 2 ** (1 + i))).controlled_by(*arctan_sign, control_values=[0])
            yield (cirq.Z(q) ** (-1 / 2 ** (1 + i))).controlled_by(*arctan_sign, control_values=[1])

        yield cirq.inverse(arctan_op)
        yield cirq.inverse(encoder_op)

        qm.qfree([*arctan_sign, *arctan_target, *target_qubits])

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['@'] * self.control_registers.bitsize
        wire_symbols += ['ROTy'] * self.selection_registers.bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)
