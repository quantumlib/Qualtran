from typing import Sequence
import cirq


class MultiTargetCNOT(cirq.Gate):
    """Implements single control, multi-target CNOT_{n} gate in 2*log(n) + 1 CNOT depth.

    Implements CNOT_{n} = |0><0| I + |1><1| X^{n} using a circuit of depth 2*log(n) + 1
    containing only CNOT gates. See Appendix B.1 of https://arxiv.org/abs/1812.00954 for
    reference.
    """

    def __init__(self, num_targets: int):
        self._num_targets = num_targets

    def _num_qubits_(self) -> int:
        return 1 + self._num_targets

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        def cnots_for_depth_i(i: int, q: Sequence[cirq.Qid]) -> cirq.OP_TREE:
            for c, t in zip(q[: 2**i], q[2**i : min(len(q), 2 ** (i + 1))]):
                yield cirq.CNOT(c, t)

        control, targets = qubits[0], list(qubits[1:])
        depth = len(targets).bit_length()
        for i in range(depth):
            yield cirq.Moment(cnots_for_depth_i(depth - i - 1, targets))
        yield cirq.CNOT(control, targets[0])
        for i in range(depth):
            yield cirq.Moment(cnots_for_depth_i(i, targets))

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=["@"] + ["X"] * (self._num_qubits_() - 1)
        )
