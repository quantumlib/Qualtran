from typing import List

import numpy as np

from cirq_qubitization.arithmetic_gates import LessThanGate
import cirq


class PrepareUniformSuperposition(cirq.Gate):
    def __init__(self, n: int, *, num_controls: int = 0):
        target_register = (n - 1).bit_length()
        self._K = 0
        while n > 1 and n % 2 == 0:
            self._K += 1
            n = n / 2
        self._L = n
        self._logL = target_register - self._K
        self._num_controls = num_controls

    def _num_qubits_(self) -> int:
        return self._num_controls + self._K + self._logL + 1

    def __repr__(self) -> str:
        return (
            f"cirq_qubitization.PrepareUniformSuperposition("
            f"{(2**self._K) * self._L}, "
            f"num_controls={self._num_controls}"
            f")"
        )

    def _decompose_(self, qubits: List[cirq.Qid]) -> cirq.OP_TREE:
        controls = qubits[: self._num_controls]
        k_qubits = qubits[self._num_controls : self._num_controls + self._K]
        logL_qubits = qubits[self._num_controls + self._K : -1]
        ancilla = qubits[-1]
        yield [
            op.controlled_by(*controls)
            for op in cirq.H.on_each(*(k_qubits + logL_qubits))
        ]
        if not logL_qubits:
            return
        A = (2**self._K) * self._L
        theta = np.arccos(1 - (2 ** np.floor(np.log2(self._L))) / self._L)

        yield LessThanGate([2] * self._logL, A).on(*logL_qubits, ancilla)
        yield cirq.Rz(rads=theta)(ancilla)
        yield LessThanGate([2] * self._logL, A).on(*logL_qubits, ancilla)

        yield cirq.H.on_each(*logL_qubits)
        yield cirq.X(ancilla).controlled_by(
            *logL_qubits,
            *controls,
            control_values=[0] * self._logL + [1] * self._num_controls,
        )
        yield cirq.Rz(rads=theta)(ancilla)
        yield cirq.X(ancilla).controlled_by(
            *logL_qubits,
            *controls,
            control_values=[0] * self._logL + [1] * self._num_controls,
        )
        yield cirq.H.on_each(*logL_qubits)
