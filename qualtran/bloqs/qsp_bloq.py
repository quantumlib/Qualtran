from functools import cached_property
from typing import Any, Dict, Tuple

import cirq
import numpy as np
from attrs import frozen
from cirq_ft.infra import GateWithRegisters, Register, Signature
from numpy.typing import NDArray


def _arbitrary_SU2_rotation(theta: float, phi: float, lambd: float):
    r"""Implements an arbitrary SU(2) rotation defined by

    .. math::

        \begin{matrix}
        e^{i(\lambda + \phi)} \cos(\theta) & e^{i\phi} \sin(\theta) \\
        e^{i\phi} \sin(\theta) & - \cos(\theta)
        \end{matrix}

    Returns:
        A 2x2 rotation matrix

    References:
        [Generalized Quantum Signal Processing](https://arxiv.org/abs/2308.01501)
            Motlagh and Wiebe. (2023). Equation 7.
    """
    return np.array(
        [
            [np.exp(1j * (lambd + phi)) * np.cos(theta), np.exp(1j * phi) * np.sin(theta)],
            [np.exp(1j * lambd) * np.sin(theta), -np.cos(theta)],
        ]
    )


def qsp_phase_factors(P: np.ndarray, Q: np.ndarray) -> Dict[str, Any]:
    """Computes the QSP signal rotations for a given pair of polynomials.
    The QSP transformation is described in Theorem 3, and the algorithm for computing co-efficients is described in Algorithm 1.

    Args:
        P: Co-efficients of a complex polynomial.
        Q: Co-efficients of a complex polynomial.

    References:
        [Generalized Quantum Signal Processing](https://arxiv.org/abs/2308.01501)
            Motlagh and Wiebe. (2023). Theorem 3; Algorithm 1.
    """
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
    """Applies a QSP sequence described by a pair of polynomials $P, Q$, to a unitary $U$ to obtain a block-encoding of $P(U)$.
    The exact circuit is described in Figure 2.

    Args:
        U: Unitary operation (encoding some Hermitian matrix $H$ as $e^{iH}$).
        P: Co-efficients of a complex polynomial.
        Q: Co-efficients of a complex polynomial.

    References:
        [Generalized Quantum Signal Processing](https://arxiv.org/abs/2308.01501)
            Motlagh and Wiebe. (2023). Theorem 3; Figure 2.
    """

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
