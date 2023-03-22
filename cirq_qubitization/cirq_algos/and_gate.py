from functools import cached_property
from typing import Any, Sequence

import cirq

from cirq_qubitization.cirq_infra.gate_with_registers import GateWithRegisters, Registers
from cirq_qubitization.t_complexity_protocol import TComplexity


@cirq.value_equality
class And(GateWithRegisters):
    """And gate optimized for T-count.

    Assumptions:
        * And(cv).on(c1, c2, target) assumes that target is initially in the |0> state.
        * And(cv, adjoint=True).on(c1, c2, target) will always leave the target in |0> state.

    Multi-controlled AND version decomposes into an AND ladder with `#controls - 2` ancillas.

    References:
        (Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity)[https://arxiv.org/abs/1805.03662].
            Babbush et. al. 2018.
        (Verifying Measurement Based Uncomputation)[https://algassert.com/post/1903].
            Gidney, C. 2019.
    """

    def __init__(self, cv: Sequence[int] = (1, 1), *, adjoint: bool = False) -> None:
        assert len(cv) >= 2, "And gate needs at-least 2 qubits to compute the AND of."
        self.cv = tuple(cv)
        self.adjoint = adjoint

    @cached_property
    def registers(self) -> Registers:
        return Registers.build(control=len(self.cv), ancilla=len(self.cv) - 2, target=1)

    def __pow__(self, power: int) -> "And":
        if power == 1:
            return self
        if power == -1:
            return And(self.cv, adjoint=self.adjoint ^ True)
        return NotImplemented

    def __str__(self) -> str:
        suffix = "" if self.cv == (1,) * len(self.cv) else str(self.cv)
        return f"And†{suffix}" if self.adjoint else f"And{suffix}"

    def __repr__(self) -> str:
        return f"cirq_qubitization.And({self.cv}, adjoint={self.adjoint})"

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        controls = ["(0)", "@"]
        target = "And†" if self.adjoint else "And"
        wire_symbols = [controls[c] for c in self.cv]
        wire_symbols += ["Anc"] * (len(self.cv) - 2)
        wire_symbols += [target]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _has_unitary_(self) -> bool:
        return not self.adjoint

    def _decompose_single_and(
        self, cv1: int, cv2: int, c1: cirq.Qid, c2: cirq.Qid, target: cirq.Qid
    ) -> cirq.OP_TREE:
        """Decomposes a single `And` gate on 2 controls and 1 target in terms of Clifford+T gates.

        * And(cv).on(c1, c2, target) uses 4 T-gates and assumes target is in |0> state.
        * And(cv, adjoint=True).on(c1, c2, target) uses measurement based un-computation
            (0 T-gates) and will always leave the target in |0> state.
        """
        pre_post_ops = [cirq.X(q) for (q, v) in zip([c1, c2], [cv1, cv2]) if v == 0]
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

    def _decompose_via_tree(
        self,
        controls: Sequence[cirq.Qid],
        control_values: Sequence[int],
        ancillas: Sequence[cirq.Qid],
        target: cirq.Qid,
    ) -> cirq.OP_TREE:
        """Decomposes multi-controlled `And` in-terms of an `And` ladder of size #controls- 2."""
        if len(controls) == 2:
            yield And(control_values, adjoint=self.adjoint).on(*controls, target)
            return
        new_controls = (ancillas[0], *controls[2:])
        new_control_values = (1, *control_values[2:])
        and_op = And(control_values[:2], adjoint=self.adjoint).on(*controls[:2], ancillas[0])
        if self.adjoint:
            yield from self._decompose_via_tree(
                new_controls, new_control_values, ancillas[1:], target
            )
            yield and_op
        else:
            yield and_op
            yield from self._decompose_via_tree(
                new_controls, new_control_values, ancillas[1:], target
            )

    def decompose_from_registers(
        self, control: Sequence[cirq.Qid], ancilla: Sequence[cirq.Qid], target: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        (target,) = target
        if len(control) == 2:
            yield from self._decompose_single_and(*self.cv, *control, target)
        else:
            yield from self._decompose_via_tree(control, self.cv, ancilla, target)

    def __eq__(self, other: 'And'):
        return self.cv == other.cv and self.adjoint == other.adjoint

    def _value_equality_values_(self) -> Any:
        return (self.cv, self.adjoint)

    def _t_complexity_(self) -> TComplexity:
        pre_post_cliffords = len(self.cv) - sum(self.cv)  # number of zeros in self.cv
        num_single_and = len(self.cv) - 1
        if self.adjoint:
            return TComplexity(clifford=4 * num_single_and + 2 * pre_post_cliffords)
        else:
            return TComplexity(
                t=4 * num_single_and, clifford=9 * num_single_and + 2 * pre_post_cliffords
            )
