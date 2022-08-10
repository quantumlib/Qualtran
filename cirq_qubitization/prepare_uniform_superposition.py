from functools import cached_property
from typing import Sequence

import cirq
import numpy as np

from cirq_qubitization.arithmetic_gates import LessThanGate
from cirq_qubitization.gate_with_registers import GateWithRegisters, Registers


class PrepareUniformSuperposition(GateWithRegisters):
    def __init__(self, n: int, *, num_controls: int = 0):
        target_bitsize = (n - 1).bit_length()
        self._K = 0
        while n > 1 and n % 2 == 0:
            self._K += 1
            n = n // 2
        self._L = int(n)
        self._logL = target_bitsize - self._K
        self._num_controls = num_controls

    @cached_property
    def registers(self) -> Registers:
        return Registers.build(
            controls=self._num_controls, logL_qubits=self._logL, k_qubits=self._K, ancilla=1
        )

    def __repr__(self) -> str:
        return (
            f"cirq_qubitization.PrepareUniformSuperposition("
            f"{(2**self._K) * self._L}, "
            f"num_controls={self._num_controls}"
            f")"
        )

    def decompose_from_registers(
        self,
        controls: Sequence[cirq.Qid],
        logL_qubits: Sequence[cirq.Qid],
        k_qubits: Sequence[cirq.Qid],
        ancilla: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        (ancilla,) = ancilla
        yield [op.controlled_by(*controls) for op in cirq.H.on_each(*(k_qubits + logL_qubits))]
        if not logL_qubits:
            return
        theta = np.arccos(1 - (2 ** np.floor(np.log2(self._L))) / self._L)

        yield LessThanGate([2] * self._logL, self._L).on(*logL_qubits, ancilla)
        yield cirq.Rz(rads=theta)(ancilla)
        yield LessThanGate([2] * self._logL, self._L).on(*logL_qubits, ancilla)

        yield cirq.H.on_each(*logL_qubits)
        yield cirq.X(ancilla).controlled_by(
            *logL_qubits, *controls, control_values=[0] * self._logL + [1] * self._num_controls
        )
        yield cirq.Rz(rads=theta)(ancilla)
        yield cirq.X(ancilla).controlled_by(
            *logL_qubits, *controls, control_values=[0] * self._logL + [1] * self._num_controls
        )
        yield cirq.H.on_each(*logL_qubits)
