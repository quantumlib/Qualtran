import enum
from functools import cached_property
from typing import Sequence, Tuple

import cirq
import numpy as np
from attrs import field, frozen

from cirq_qubitization.cirq_algos import and_gate
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

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ):
        control, targets = quregs['control'], quregs['targets']

        def cnots_for_depth_i(i: int, q: Sequence[cirq.Qid]) -> cirq.OP_TREE:
            for c, t in zip(q[: 2**i], q[2**i : min(len(q), 2 ** (i + 1))]):
                yield cirq.CNOT(c, t)

        depth = len(targets).bit_length()
        for i in range(depth):
            yield cirq.Moment(cnots_for_depth_i(depth - i - 1, targets))
        yield cirq.CNOT(*control, targets[0])
        for i in range(depth):
            yield cirq.Moment(cnots_for_depth_i(i, targets))

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=["@"] + ["X"] * self._num_targets)


def _to_tuple(x: Sequence[int]) -> Tuple[int, ...]:
    return tuple(x)


@frozen
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

    class DecomposeMode(enum.Enum):
        CLEAN_ANCILLA = 1
        DIRTY_ANCILLA = 2

    cvs: Tuple[int, ...] = field(converter=_to_tuple)
    target_gate: cirq.Pauli = cirq.X
    mode: DecomposeMode = DecomposeMode.CLEAN_ANCILLA

    @cached_property
    def registers(self) -> Registers:
        return Registers.build(controls=len(self.cvs), target=1)

    def _decompose_dirty(
        self,
        context: cirq.DecompositionContext,
        controls: Sequence[cirq.Qid],
        target: Sequence[cirq.Qid],
    ) -> cirq.ops.op_tree.OpTree:
        pre_post_x = [cirq.X(controls[i]) for i, b in enumerate(self.cvs) if not b]
        if len(controls) == 2:
            return [pre_post_x, self.target_gate(*target).controlled_by(*controls), pre_post_x]
        anc = context.qubit_manager.qborrow(len(controls) - 2)
        ops = [cirq.CCNOT(anc[-i], controls[-i], anc[-i + 1]) for i in range(2, len(anc) + 1)]
        inverted_v_ladder = ops + [cirq.CCNOT(*controls[:2], anc[0])] + ops[::-1]

        yield pre_post_x
        yield self.target_gate(*target).controlled_by(anc[-1], controls[-1])
        yield inverted_v_ladder
        yield self.target_gate(*target).controlled_by(anc[-1], controls[-1])
        yield inverted_v_ladder
        yield pre_post_x
        context.qubit_manager.qfree(anc)

    def _decompose_clean(
        self,
        context: cirq.DecompositionContext,
        controls: Sequence[cirq.Qid],
        target: Sequence[cirq.Qid],
    ) -> cirq.ops.op_tree.OpTree:
        qm = context.qubit_manager
        and_ancilla, and_target = qm.qalloc(len(self.cvs) - 2), qm.qalloc(1)
        yield and_gate.And(self.cvs).on_registers(
            control=controls, ancilla=and_ancilla, target=and_target
        )
        yield self.target_gate.on(*target).controlled_by(*and_target)
        yield and_gate.And(self.cvs, adjoint=True).on_registers(
            control=controls, ancilla=and_ancilla, target=and_target
        )
        qm.qfree(and_ancilla + and_target)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: Sequence['cirq.Qid']
    ) -> cirq.OP_TREE:
        controls, target = quregs['controls'], quregs['target']
        if self.mode == self.DecomposeMode.CLEAN_ANCILLA:
            yield from self._decompose_clean(context=context, controls=controls, target=target)
        elif self.mode == self.DecomposeMode.DIRTY_ANCILLA:
            yield from self._decompose_dirty(context=context, controls=controls, target=target)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@" if b else "@(0)" for b in self.cvs]
        wire_symbols += [str(self.target_gate)]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _t_complexity_dirty(self) -> TComplexity:
        toffoli_complexity = t_complexity(cirq.CCNOT)
        controlled_pauli_complexity = t_complexity(self.target_gate.controlled(2))
        pre_post_x_complexity = (len(self.cvs) - sum(self.cvs)) * t_complexity(cirq.X)
        return (
            (4 * len(self.cvs) - 10) * toffoli_complexity
            + 2 * controlled_pauli_complexity
            + 2 * pre_post_x_complexity
        )

    def _t_complexity_clean(self) -> TComplexity:
        and_cost = t_complexity(and_gate.And(self.cvs))
        controlled_pauli_cost = t_complexity(self.target_gate.controlled(1))
        and_inv_cost = t_complexity(and_gate.And(self.cvs, adjoint=True))
        return and_cost + controlled_pauli_cost + and_inv_cost

    def _t_complexity_(self) -> TComplexity:
        if self.mode == self.DecomposeMode.CLEAN_ANCILLA:
            return self._t_complexity_clean()
        elif self.mode == self.DecomposeMode.DIRTY_ANCILLA:
            return self._t_complexity_dirty()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs') -> np.ndarray:
        return cirq.apply_unitary(self.target_gate.controlled(control_values=self.cvs), args)

    def _has_unitary_(self) -> bool:
        return True
