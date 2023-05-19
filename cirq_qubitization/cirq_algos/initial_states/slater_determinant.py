from typing import Sequence

import cirq
import numpy as np

from cirq_qubitization import cirq_infra, t_complexity_protocol


class SlaterDeterminant(cirq_infra.GateWithRegisters):
    r"""Gate to prepare an initial state as a Slater determinant.

    Registers:

    References:
        [Improved techniques for preparing eigenstates of fermionic
        Hamiltonians](https://www.nature.com/articles/s41534-018-0071-5)
        [Fault-Tolerant Quantum Simulations of Chemistry in First
        Quantization](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332)
        [Quantum simulation of exact electron dynamics can be more efficient
        than classical mean-field methods](https://arxiv.org/pdf/2301.01203.pdf)
    """

    def __init__(self, eta, N):
        """
        """
        self._eta = eta
        self._N = N
        self._bitsize = max(np.ceil(np.log2(N)), 1)
        self._n_eta = max(np.ceil(np.log2(self._eta + 1)), 1)

    @property
    def registers(self):
        return cirq_infra.Registers.build(p=self._eta * self._bitsize)

    def decompose_from_registers(
        self, selection: Sequence[cirq.Qid], **targets: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        raise NotImplementedError
        # yield FromSecondQuantization
        # yield AntiSymmetrize

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["PrepareSlaterDet"] * self._bitsize * self._eta
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _t_complexity_(self) -> t_complexity_protocol.TComplexity:
        num_toff = self._N * (3*self._eta + self._n_eta - 2)
        num_toff += np.ceil(self._eta * self._n_eta * self._bitsize)
        num_cliff = self._N * self._eta * self._bitsize
        return t_complexity_protocol.TComplexity(
            t=num_toff,
            clifford=num_ciff
        )


class AntiSymmetrize(cirq_infra.GateWithRegisters):
    r"""Gate to antisymmetrize a set of eta labelled registers each of size log N.

    $$
    \mathcal{A}|12\rangle = |12\rangle - |21\rangle,
    $$
    where each register $|p\rangle$ is of size $\log N$.

    Registers:

    References:
        [Improved techniques for preparing eigenstates of fermionic
        Hamiltonians](https://www.nature.com/articles/s41534-018-0071-5)
    """

    def __init__(self, num_registers: int, bitsize: int):
        """ """
        self._n = num_registers
        self._bitsize = bitsize

    @property
    def registers(self):
        return cirq_infra.Registers.build(p=self._n * self._bitsize)

    def decompose_from_registers(
        self, selection: Sequence[cirq.Qid], **targets: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        raise NotImplementedError

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["AntiSymm"] * self._bitsize * self._n
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _t_complexity_(self) -> t_complexity_protocol.TComplexity:
        # This is big(O) but negligibly small so taking prefactor of 1.
        return t_complexity_protocol.TComplexity(
            t=
        )
