from functools import cached_property
from typing import Sequence

import cirq

from cirq_qubitization.cirq_infra import qubit_manager
from cirq_qubitization.cirq_infra.gate_with_registers import GateWithRegisters, Registers
from cirq_qubitization.t_complexity_protocol import t_complexity, TComplexity


class MultiTargetCNOT(GateWithRegisters):
    """Implements single control, multi-target CNOT_{n} gate in 2*log(n) + 1 CNOT depth.

    Implements CNOT_{n} = |0><0| I + |1><1| X^{n} using a circuit of depth 2*log(n) + 1
    containing only CNOT gates. See Appendix B.1 of https://arxiv.org/abs/1812.00954 for
    reference.
    """

    def __init__(self, num_targets: int):
        self._num_targets = num_targets

    @cached_property
    def registers(self) -> Registers:
        return Registers.build(control=1, targets=self._num_targets)

    def decompose_from_registers(self, control: Sequence[cirq.Qid], targets: Sequence[cirq.Qid]):
        def cnots_for_depth_i(i: int, q: Sequence[cirq.Qid]) -> cirq.OP_TREE:
            for c, t in zip(q[: 2**i], q[2**i : min(len(q), 2 ** (i + 1))]):
                yield cirq.CNOT(c, t)

        (control,) = control
        depth = len(targets).bit_length()
        for i in range(depth):
            yield cirq.Moment(cnots_for_depth_i(depth - i - 1, targets))
        yield cirq.CNOT(control, targets[0])
        for i in range(depth):
            yield cirq.Moment(cnots_for_depth_i(i, targets))

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=["@"] + ["X"] * self._num_targets)


class MultiControlPauli(GateWithRegisters):
    """Implements multi-control, single-target C^{n}P gate.

    Implements $C^{n}P = (1 - |1^{n}><1^{n}|) I + |1^{n}><1^{n}| P^{n}$ using $n-2$
    dirty ancillas and 4n - 8 TOFFOLI gates. See Appendix B.1 of https://arxiv.org/abs/1812.00954 for
    reference.

    References:
        [Factoring with $n+2$ clean qubits and $n-1$ dirty qubits](https://arxiv.org/abs/1706.07884).
        Craig Gidney (2018). Figure 25.
        [Constructing Large Controlled Nots]
        (https://algassert.com/circuits/2015/06/05/Constructing-Large-Controlled-Nots.html)
    """

    def __init__(self, num_controls: int, *, target_gate: cirq.Pauli = cirq.X):
        self._num_controls = num_controls
        self._target_gate = target_gate

    @cached_property
    def registers(self) -> Registers:
        return Registers.build(controls=self._num_controls, target=1)

    def decompose_from_registers(self, controls: Sequence[cirq.Qid], target: Sequence[cirq.Qid]):
        if len(controls) == 2:
            return self._target_gate(*target).controlled_by(*controls)
        anc = qubit_manager.qborrow(len(controls) - 2)
        ops = [cirq.CCNOT(anc[-i], controls[-i], anc[-i + 1]) for i in range(2, len(anc) + 1)]
        inverted_v_ladder = ops + [cirq.CCNOT(*controls[:2], anc[0])] + ops[::-1]

        yield self._target_gate(*target).controlled_by(anc[-1], controls[-1])
        yield inverted_v_ladder
        yield self._target_gate(*target).controlled_by(anc[-1], controls[-1])
        yield inverted_v_ladder

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=["@"] * self._num_controls + [str(self._target_gate)]
        )

    def _t_complexity_(self) -> TComplexity:
        toffoli_complexity = t_complexity(cirq.CCNOT)
        controlled_pauli_complexity = t_complexity(self._target_gate.controlled(2))
        return (4 * self._num_controls - 10) * toffoli_complexity + 2 * controlled_pauli_complexity
