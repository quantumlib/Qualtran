from functools import cached_property
from typing import Collection, Optional, Sequence, Tuple, Union

import cirq

from cirq_qubitization import cirq_infra, generic_select
from cirq_qubitization.cirq_algos import reflection_using_prepare, state_preparation


@cirq.value_equality()
class QubitizationWalkOperator(cirq_infra.GateWithRegisters):
    def __init__(
        self,
        select: generic_select.GenericSelect,
        prepare: state_preparation.StatePreparationAliasSampling,
        *,
        control_val: Optional[int] = None,
        power: int = 1,
    ):
        self._select = select
        self._reflect = reflection_using_prepare.ReflectionUsingPrepare(
            prepare, control_val=control_val
        )
        self._control_val = control_val
        assert self._select.control_registers == self._reflect.control_registers
        self.power = power

    @cached_property
    def control_registers(self) -> cirq_infra.Registers:
        return self._select.control_registers

    @cached_property
    def selection_registers(self) -> cirq_infra.Registers:
        return self._reflect.target_registers

    @cached_property
    def target_registers(self) -> cirq_infra.Registers:
        return self._select.target_registers

    @cached_property
    def registers(self) -> cirq_infra.Registers:
        return cirq_infra.Registers(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )

    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        select_reg = {k: v for k, v in qubit_regs.items() if k in self._select.registers}
        reflect_reg = {k: v for k, v in qubit_regs.items() if k in self._reflect.registers}
        select_op = self._select.on_registers(**select_reg)
        reflect_op = self._reflect.on_registers(**reflect_reg)
        for _ in range(self.power):
            yield select_op
            yield reflect_op

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['@' if self._control_val else '@(0)'] * self.control_registers.bitsize
        wire_symbols += ['W'] * (self.registers.bitsize - self.control_registers.bitsize)
        wire_symbols[-1] = f'W^{self.power}' if self.power != 1 else 'W'
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def controlled(
        self,
        num_controls: int = None,
        control_values: Sequence[Union[int, Collection[int]]] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> 'QubitizationWalkOperator':
        if num_controls is None:
            num_controls = 1
        if control_values is None:
            control_values = [1] * num_controls
        if len(control_values) == 1 and self._control_val is None:
            return QubitizationWalkOperator(
                self._select.controlled(control_values=control_values),
                self._reflect.prepare_gate,
                control_val=control_values[-1],
            )
        raise NotImplementedError(f'Cannot create a controlled version of {self}')

    def with_power(self, new_power: int) -> 'QubitizationWalkOperator':
        return QubitizationWalkOperator(
            self._select, self._reflect.prepare_gate, control_val=self._control_val, power=new_power
        )

    def _value_equality_values_(self):
        return self._select, self._reflect, self._control_val, self.power

    def __pow__(self, power: int):
        return self.with_power(self.power * power)
