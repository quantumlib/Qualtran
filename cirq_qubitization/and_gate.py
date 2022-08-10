from typing import Tuple
import cirq


class And(cirq.Gate):
    """And gate optimized for T-count.

    Assumptions:
        * And(cv).on(c1, c2, target) assumes that target is initially in the |0> state.
        * And(cv, adjoint=True).on(c1, c2, target) will always leave the target in |0> state.

    References:
        (Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity)[https://arxiv.org/abs/1805.03662].
            Babbush et. al. 2018.
        (Verifying Measurement Based Uncomputation)[https://algassert.com/post/1903].
            Gidney, C. 2019.
    """

    def __init__(self, cv: Tuple[int, int] = (1, 1), *, adjoint: bool = False) -> None:
        self.cv = cv
        self.adjoint = adjoint

    def _num_qubits_(self) -> int:
        return 3

    def __pow__(self, power: int) -> "And":
        if power == 1:
            return self
        if power == -1:
            return And(self.cv, adjoint=self.adjoint ^ True)
        return NotImplemented

    def __str__(self) -> str:
        suffix = "" if self.cv == (1, 1) else str(self.cv)
        return f"And†{suffix}" if self.adjoint else f"And{suffix}"

    def __repr__(self) -> str:
        return f"cirq_qubitization.And({self.cv}, adjoint={self.adjoint})"

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        controls = ["(0)", "@"]
        target = "And†" if self.adjoint else "And"
        return cirq.CircuitDiagramInfo(wire_symbols=[controls[c] for c in self.cv] + [target])

    def _has_unitary_(self) -> bool:
        return not self.adjoint

    def _decompose_(self, qubits) -> cirq.OP_TREE:
        c1, c2, target = qubits
        pre_post_ops = [cirq.X(q) for (q, v) in zip([c1, c2], self.cv) if v == 0]
        yield pre_post_ops
        if self.adjoint:
            yield cirq.H(target)
            yield cirq.measure(target, key=f"{target}")
            yield cirq.CZ(c1, c2).with_classical_controls(f"{target}")
            yield cirq.reset(target)
        else:
            yield [cirq.H(target), cirq.T(target)]
            yield [cirq.CNOT(c1, target), cirq.CNOT(c2, target)]
            yield [cirq.CNOT(target, c1), cirq.CNOT(target, c2)]
            yield [cirq.T(c1) ** -1, cirq.T(c2) ** -1, cirq.T(target)]
            yield [cirq.CNOT(target, c1), cirq.CNOT(target, c2)]
            yield [cirq.H(target), cirq.S(target)]
        yield pre_post_ops
