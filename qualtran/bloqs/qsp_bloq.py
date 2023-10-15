from functools import cached_property
from typing import Any, Dict, Tuple

import cirq
import numpy as np
from attrs import frozen
from cirq_ft.infra import GateWithRegisters, Register, Signature
from numpy.typing import NDArray


def _arbitrary_SU2_rotation(theta, phi, lambd):
    el = np.exp(lambd * 1j)
    ep = np.exp(phi * 1j)
    cs = np.cos(theta)
    sn = np.sin(theta)
    return np.array([[el * ep * cs, ep * sn], [el * sn, -cs]])


def qsp_phase_factors(P: np.ndarray, Q: np.ndarray) -> Dict[str, Any]:
    P = np.array(P)
    Q = np.array(Q)
    if P.ndim != 1 or Q.ndim != 1 or P.shape != Q.shape:
        raise ValueError("Polynomials P and Q must be arrays of the same length.")

    S = np.array([P, Q])
    n = S.shape[1]

    theta = np.zeros(n)
    phi = np.zeros(n)
    lambd = 0

    for d in range(n - 1, 0, -1):
        assert S.shape == (2, d + 1)

        a, b = S[:, d]
        theta[d] = np.arctan2(np.abs(a), np.abs(b))
        phi[d] = np.angle(a / b)

        if d == 0:
            lambd = np.angle(b)
        else:
            S = _arbitrary_SU2_rotation(theta[d], phi[d], 0) @ S
            S = np.array([S[0][1:d], S[1][0 : d - 1]])

    return {'theta': theta, 'phi': phi, 'lambda': lambd}


@frozen
class QEVTCircuit(GateWithRegisters):
    U: GateWithRegisters
    P: Tuple[float]
    Q: Tuple[float]

    @property
    def signature(self) -> Signature:
        return Signature((Register(name='signal', bitsize=1),) + tuple(self.U.signature))

    @cached_property
    def _qsp_phases(self):
        return qsp_phase_factors(self.P, self.Q)

    @cached_property
    def _theta(self):
        return self._qsp_phases['theta']

    @cached_property
    def _phi(self):
        return self._qsp_phases['phi']

    @cached_property
    def _lambda(self):
        return self._qsp_phases['lambda']

    def _signal_gate(self, signal_qubit, theta, phi, lambd) -> cirq.OP_TREE:
        gates = cirq.single_qubit_matrix_to_gates(_arbitrary_SU2_rotation(theta, phi, lambd))
        for gate in gates:
            yield gate.on(signal_qubit)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, signal, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        assert len(signal) == 1
        signal_qubit = signal[0]

        yield from self._signal_gate(signal_qubit, self._theta[0], self._phi[0], self._lambda)
        for theta, phi in zip(self._theta[1:], self._phi[1:]):
            yield self.U.on_registers(**quregs).controlled_by(signal_qubit, control_values=[0])
            yield from self._signal_gate(signal_qubit, theta, phi, 0)
