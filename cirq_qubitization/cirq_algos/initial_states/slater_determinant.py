from typing import Sequence

import cirq
import numpy as np

from cirq_qubitization import cirq_infra, t_complexity_protocol


class SlaterDeterminant(cirq_infra.GateWithRegisters):
    r"""Gate to prepare an initial state as a Slater determinant in first quantization.

    Registers:

    References:
        [Improved techniques for preparing eigenstates of fermionic
        Hamiltonians](https://www.nature.com/articles/s41534-018-0071-5)
        [Fault-Tolerant Quantum Simulations of Chemistry in First
        Quantization](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332)
        [Quantum simulation of exact electron dynamics can be more efficient
        than classical mean-field methods](https://arxiv.org/pdf/2301.01203.pdf)
    """

    def __init__(self, eta: int, N: int):
        """ """
        self._eta = eta
        self._N = N
        self._bitsize = max(np.ceil(np.log2(N)), 1)
        self._n_eta = max(np.ceil(np.log2(self._eta + 1)), 1)

    @property
    def registers(self):
        return cirq_infra.Registers.build(q=self._N, p=self._eta * self._bitsize)

    def decompose_from_registers(
        self, selection: Sequence[cirq.Qid], **targets: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        # This is pseudocode
        # yield GivensRotation
        # yield PrepareFromSecondQuantization
        # yield AntiSymmetrize
        raise NotImplementedError

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["PrepareSlaterDet"] * (self._bitsize * self._eta + self._N)
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _t_complexity_(self) -> t_complexity_protocol.TComplexity:
        # N additions into xi
        num_toff = self._N * (self._n_eta - 1)
        # Controlled Unary iteration
        num_toff += self._N * (self._eta - 1)
        # Unary iteration
        num_toff += self._N * self._eta
        # Multi-Controlled not on q register
        num_toff += self._N * self._eta
        # CX cost for setting first qunatized registers + controls
        num_cliff = self._N * self._eta * self._bitsize
        # TODO Givens cost?
        return t_complexity_protocol.TComplexity(t=4 * num_toff, clifford=num_cliff)


class PrepareFromSecondQuantization(cirq_infra.GateWithRegisters):
    r"""Gate to copy an initial state from second quantized to first quantization.

    A slater determinant in second quantization can be represented as a set of occupations
    e.g. [1,1,1,1,0,0,0] means occupy the lowest 4 molecular orbitals. This Gate
    copies this information to the first quantized registers in the following
    way. Registers in first quantization are represented by eta, log N sized registers.

    Registers:

    References:
        [Quantum simulation of exact electron dynamics can be more efficient
        than classical mean-field methods](https://arxiv.org/pdf/2301.01203.pdf)
    """

    def __init__(self, eta: int, N: int):
        self._eta = eta
        self._N = N
        self._bitsize = max(np.ceil(np.log2(N)), 1)
        self._n_eta = max(np.ceil(np.log2(self._eta + 1)), 1)

    @property
    def registers(self):
        return cirq_infra.Registers.build(q=self._N, p=self._eta * self._bitsize)

    def decompose_from_registers(
        self, selection: Sequence[cirq.Qid], **targets: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        raise NotImplementedError

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["From2NDQNT"] * (self._bitsize * self._eta + self._N)
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _t_complexity_(self) -> t_complexity_protocol.TComplexity:
        # TODO: This is big(O), should not be impossible to work out constants.
        num_toff = self._N * (self._n_eta - 1)
        # Controlled Unary iteration
        num_toff += self._N * (self._eta - 1)
        # Unary iteration
        num_toff += self._N * self._eta
        # Multi-Controlled not on q register
        num_toff += self._N * self._eta
        # CX cost for setting first qunatized registers + controls
        num_cliff = self._N * self._eta * self._bitsize
        return t_complexity_protocol.TComplexity(t=4 * num_toff, clifford=num_cliff)


@frozen
class AntiSymmetrize(cirq_infra.GateWithRegisters):
    r"""Gate to antisymmetrize a set of eta labelled registers each of size log N.

    The antisymmetrization operator produces all permutations of eta labels with
    the appropriate sign, e.g. for eta = 2
    $$
    \mathcal{A}|12\rangle = |12\rangle - |21\rangle,
    $$
    where each register $|p\rangle$ is of size $\log N$.

    Registers:

    References:
        [Improved techniques for preparing eigenstates of fermionic
        Hamiltonians](https://www.nature.com/articles/s41534-018-0071-5)
    """

    n: int
    bitsize: int

    @property
    def registers(self):
        return cirq_infra.Registers.build(p=self.n * self.bitsize)

    def decompose_from_registers(
        self, selection: Sequence[cirq.Qid], **targets: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        raise NotImplementedError

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["AntiSymm"] * self.bitsize * self._n
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _t_complexity_(self) -> t_complexity_protocol.TComplexity:
        # TODO: This is big(O), should not be impossible to work out constants.
        num_t = 4 * self.n * np.log2(self.n) * self.bitsize
        return t_complexity_protocol.TComplexity(t=num_t)
