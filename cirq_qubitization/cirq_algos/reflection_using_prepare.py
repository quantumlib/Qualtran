import itertools
from functools import cached_property
from typing import Sequence

import cirq

from cirq_qubitization import cirq_infra
from cirq_qubitization.cirq_algos import multi_control_multi_target_pauli as mcmt
from cirq_qubitization.cirq_algos import state_preparation


class ReflectionUsingPrepare(cirq_infra.GateWithRegisters):
    """Applies $R_{s} = I - 2|S><S|$ using $R_{s} = P^†(I - 2|0><0|)P$ s.t. $P|0> = |S>$

    The composite gate corresponds to implementing the following circuit:

    |control> ------------------ Z -------------------
                                 |
    |L>       ---- PREPARE^† --- o --- PREPARE -------


    Args:
        prepare_gate: An instance of `cq.StatePreparationAliasSampling` gate the corresponds to
                      `PREPARE`.
        num_controls: If 1, a controlled version of the reflection operator is constructed.
                      Defaults to 0.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Figure 1.
    """

    def __init__(
        self,
        prepare_gate: state_preparation.StatePreparationAliasSampling,
        *,
        num_controls: int = 0,
    ):
        self.prepare_gate = prepare_gate
        self._num_controls = num_controls
        if self._num_controls > 1:
            raise NotImplementedError("num_controls > 1 is not yet implemented.")

    @cached_property
    def control_registers(self) -> cirq_infra.Registers:
        return (
            cirq_infra.Registers.build(control=self._num_controls)
            if self._num_controls
            else cirq_infra.Registers([])
        )

    @cached_property
    def target_registers(self) -> cirq_infra.Registers:
        return self.prepare_gate.selection_registers

    @cached_property
    def registers(self) -> cirq_infra.Registers:
        return cirq_infra.Registers([*self.control_registers, *self.target_registers])

    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        if self._num_controls:
            phase_ancilla = qubit_regs.pop('control')
        else:
            phase_ancilla = cirq_infra.qalloc(1)[0]
            yield cirq.X(phase_ancilla)
        state_prep_ancilla = {}
        for reg in self.prepare_gate.temp_registers:
            state_prep_ancilla[reg.name] = cirq_infra.qalloc(reg.bitsize)
        phase_controls = [*itertools.chain(*qubit_regs.values())]

        yield self.prepare_gate.on_registers(**qubit_regs, **state_prep_ancilla) ** -1
        # Phase the all zero state.
        yield cirq.X.on_each(*phase_controls)
        yield mcmt.MultiControlPauli(len(phase_controls), target_gate=cirq.Z).on_registers(
            controls=phase_controls, target=phase_ancilla
        )
        yield cirq.X.on_each(*phase_controls)

        yield self.prepare_gate.on_registers(**qubit_regs, **state_prep_ancilla)

        cirq_infra.qfree([q for anc in state_prep_ancilla.values() for q in anc])
        if not self._num_controls:
            yield cirq.X(phase_ancilla)
            cirq_infra.qfree(phase_ancilla)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['@'] * self.control_registers.bitsize
        wire_symbols += ['R_L'] * self.target_registers.bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)
