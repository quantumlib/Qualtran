from functools import cached_property
from typing import Collection, Optional, Sequence, Tuple, Union

import cirq

from cirq_qubitization import cirq_infra
from cirq_qubitization.cirq_algos import multi_control_multi_target_pauli as mcmt
from cirq_qubitization.cirq_algos import state_preparation


@cirq.value_equality()
class ReflectionUsingPrepare(cirq_infra.GateWithRegisters):
    """Applies reflection around a state prepared by `prepare_gate`

    Applies $R_{s} = I - 2|s><s|$ using $R_{s} = P^†(I - 2|0><0|)P$ s.t. $P|0> = |s>$.
    Here
        $|s>$: The state along which we want to reflect.
        $P$: Unitary that prepares that state $|s>$ from the zero state $|0>$
        $R_{s}$: Reflection operator that adds a `-1` phase to all states in the subspace
            spanned by $|s>$.

    The composite gate corresponds to implementing the following circuit:

    |control> ------------------ Z -------------------
                                 |
    |L>       ---- PREPARE^† --- o --- PREPARE -------


    Args:
        prepare_gate: An instance of `cq.StatePreparationAliasSampling` gate the corresponds to
            `PREPARE`.
        control_val: If 0/1, a controlled version of the reflection operator is constructed.
            Defaults to None, in which case the resulting reflection operator is not controlled.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Figure 1.
    """

    def __init__(
        self, prepare_gate: state_preparation.StatePreparationAliasSampling, *, control_val=None
    ):
        self._prepare_gate = prepare_gate
        self._control_val = control_val

    @property
    def prepare_gate(self):
        return self._prepare_gate

    @cached_property
    def control_registers(self) -> cirq_infra.Registers:
        registers = [] if self._control_val is None else [cirq_infra.Register('control', 1)]
        return cirq_infra.Registers(registers)

    @cached_property
    def target_registers(self) -> cirq_infra.Registers:
        return self.prepare_gate.selection_registers

    @cached_property
    def registers(self) -> cirq_infra.Registers:
        return cirq_infra.Registers([*self.control_registers, *self.target_registers])

    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        # 0. Allocate new ancillas, if needed.
        phase_target = (
            cirq_infra.qalloc(1)[0] if self._control_val is None else qubit_regs.pop('control')[0]
        )
        state_prep_ancilla = {
            reg.name: cirq_infra.qalloc(reg.bitsize) for reg in self.prepare_gate.temp_registers
        }
        state_prep_target_regs = qubit_regs
        # 1. PREPARE†
        yield self.prepare_gate.on_registers(**state_prep_target_regs, **state_prep_ancilla) ** -1
        # 2. MultiControlled Z, controlled on |000..00> state.
        phase_control = self.target_registers.merge_qubits(**state_prep_target_regs)
        yield cirq.X.on_each(*phase_control, phase_target if not self._control_val else [])
        yield mcmt.MultiControlPauli(len(phase_control), target_gate=cirq.Z).on_registers(
            controls=phase_control, target=phase_target
        )
        yield cirq.X.on_each(*phase_control, phase_target if not self._control_val else [])
        # 3. PREPARE
        yield self.prepare_gate.on_registers(**state_prep_target_regs, **state_prep_ancilla)

        # 4. Deallocate ancilla.
        cirq_infra.qfree([q for anc in state_prep_ancilla.values() for q in anc])
        if self._control_val is None:
            cirq_infra.qfree(phase_target)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['@' if self._control_val else '@(0)'] * self.control_registers.bitsize
        wire_symbols += ['R_L'] * self.target_registers.bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def controlled(
        self,
        num_controls: int = None,
        control_values: Sequence[Union[int, Collection[int]]] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> 'ReflectionUsingPrepare':
        if num_controls is None:
            num_controls = 1
        if control_values is None:
            control_values = [1] * num_controls
        if len(control_values) == 1 and self._control_val is None:
            return ReflectionUsingPrepare(self.prepare_gate, control_val=control_values[-1])
        raise NotImplementedError(f'Cannot create a controlled version of {self}')

    def _value_equality_values_(self):
        return self.prepare_gate, self._control_val


ReflectionUsingPrepare.__hash__ = cirq._compat.cached_method(ReflectionUsingPrepare.__hash__)
