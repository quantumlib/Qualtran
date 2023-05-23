from functools import cached_property
from typing import Sequence

import cirq
from attrs import frozen

from cirq_qubitization import cirq_infra
from cirq_qubitization.cirq_algos import select_and_prepare as sp
from cirq_qubitization.cirq_algos.mean_estimation import arctan


@frozen
class ComplexPhaseOracle(sp.SelectOracle):
    """Applies ROT_{y}|l>|garbage_{l}> = exp(i -2arctan{y_{l}})|l>|garbage_{l}>"""

    Y: sp.SelectOracle
    arctan_bitsize: int

    @cached_property
    def control_registers(self) -> cirq_infra.Registers:
        return self.Y.control_registers

    @cached_property
    def selection_registers(self) -> cirq_infra.SelectionRegisters:
        return self.Y.selection_registers

    @cached_property
    def target_registers(self) -> cirq_infra.Registers:
        return self.Y.target_registers

    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        target_reg = {reg.name: qubit_regs[reg.name] for reg in self.target_registers}
        target_qubits = self.target_registers.merge_qubits(**target_reg)

        arctan_sign, arctan_target = cirq_infra.qalloc(1), cirq_infra.qalloc(self.arctan_bitsize)
        arctan_op = arctan.ArcTan(len(target_qubits), self.arctan_bitsize).on(
            *target_qubits, *arctan_sign, *arctan_target
        )

        yield self.Y.on_registers(**qubit_regs)
        yield arctan_op
        for i, q in enumerate(arctan_target):
            yield (cirq.Z(q) ** (1 / 2 ** (1 + i))).controlled_by(*arctan_sign, control_values=[0])
            yield (cirq.Z(q) ** (-1 / 2 ** (1 + i))).controlled_by(*arctan_sign, control_values=[1])

        yield arctan_op**-1
        yield self.Y.on_registers(**qubit_regs) ** -1

        cirq_infra.qfree([*arctan_sign, *arctan_target])
