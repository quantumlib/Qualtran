from functools import cached_property
from typing import Sequence, Tuple

import cirq
import numpy as np
from attrs import field, frozen

from cirq_qubitization import cirq_infra
from cirq_qubitization.cirq_algos.and_gate import And
from cirq_qubitization.cirq_algos.arithmetic_gates import LessThanGate


@frozen
class PrepareUniformSuperposition(cirq_infra.GateWithRegisters):
    n: int
    cv: Tuple[int, ...] = field(converter=tuple, default=())

    @cached_property
    def registers(self) -> cirq_infra.Registers:
        return cirq_infra.Registers.build(controls=len(self.cv), target=(self.n - 1).bit_length())

    def __repr__(self) -> str:
        return f"cirq_qubitization.PrepareUniformSuperposition({self.n}, cv={self.cv})"

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        control_symbols = ["@" if cv else "@(0)" for cv in self.cv]
        target_symbols = ['target'] * self.registers['target'].bitsize
        target_symbols[0] = f"UNIFORM({self.n})"
        return cirq.CircuitDiagramInfo(wire_symbols=control_symbols + target_symbols)

    def decompose_from_registers(
        self, controls: Sequence[cirq.Qid], target: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        # Find K and L as per https://arxiv.org/abs/1805.03662 Fig 12.
        n, k = self.n, 0
        while n > 1 and n % 2 == 0:
            k += 1
            n = n // 2
        l, logL = int(n), self.registers['target'].bitsize - k
        logL_qubits, k_qubits = target[:logL], target[logL:]

        yield [
            op.controlled_by(*controls, control_values=self.cv) for op in cirq.H.on_each(*target)
        ]
        if not logL_qubits:
            return

        ancilla = cirq_infra.qalloc(1)
        theta = np.arccos(1 - (2 ** np.floor(np.log2(l))) / l)
        yield LessThanGate([2] * logL, l).on(*logL_qubits, *ancilla)
        yield cirq.Rz(rads=theta)(*ancilla)
        yield LessThanGate([2] * logL, l).on(*logL_qubits, *ancilla)

        yield cirq.H.on_each(*logL_qubits)

        and_gate = And((0,) * logL + self.cv)
        and_ancilla = cirq_infra.qalloc(and_gate.registers['ancilla'].bitsize)
        yield and_gate.on_registers(
            control=[*logL_qubits, *controls], ancilla=and_ancilla, target=ancilla
        )
        yield cirq.Rz(rads=theta)(*ancilla)
        yield (and_gate**-1).on_registers(
            control=[*logL_qubits, *controls], ancilla=and_ancilla, target=ancilla
        )

        yield cirq.H.on_each(*logL_qubits)
        cirq_infra.qfree([*ancilla, *and_ancilla])
