from functools import cached_property
from typing import Collection, Optional, Sequence, Tuple, Union

import cirq
from attrs import field, frozen

from cirq_qubitization import cirq_infra
from cirq_qubitization.cirq_algos import reflection_using_prepare as rup
from cirq_qubitization.cirq_algos import select_and_prepare as sp
from cirq_qubitization.cirq_algos.mean_estimation import arctan


@frozen
class CodeForRandomVariable:
    r"""We say we have the "code" for a random variable y iff we have unitaries P and Y s.t.
    $$
    P|0> = \sum_{w \in W} \sqrt{p(w)} |w> |garbage_{w}>
    $$
    and
    $$
    Y|w>|0^b> = |w>|y(w)>
    $$
    where b is the number of bits required to encode the real range of y.

    References:
        https://arxiv.org/abs/2208.07544, Definition 2.2 for P and Definition 2.10 for Y.
    """

    P: sp.PrepareOracle
    Y: sp.SelectOracle


@frozen
class PhaseOracle(sp.SelectOracle):
    """Applies ROT_{y}|l>|garbage_{l}> = exp(i -2arctan{y_{l}})|l>|garbage_{l}>"""

    Y: sp.SelectOracle
    arctan_bitsize: int

    @cached_property
    def control_registers(self) -> cirq_infra.Registers:
        return self.Y.control_registers

    @cached_property
    def selection_registers(self) -> cirq_infra.SelectionRegisters:
        return self.Y.selection_registers

    @cached_property
    def target_registers(self) -> cirq_infra.Registers:
        return self.Y.target_registers

    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        target_reg = {reg.name: qubit_regs[reg.name] for reg in self.target_registers}
        target_qubits = self.target_registers.merge_qubits(**target_reg)

        arctan_sign, arctan_target = cirq_infra.qalloc(1), cirq_infra.qalloc(self.arctan_bitsize)
        arctan_op = arctan.ArcTan(len(target_qubits), self.arctan_bitsize).on(
            *target_qubits, *arctan_sign, *arctan_target
        )

        yield self.Y.on_registers(**qubit_regs)
        yield arctan_op
        for i, q in enumerate(arctan_target):
            yield (cirq.Z(q) ** (1 / 2 ** (1 + i))).controlled_by(*arctan_sign, control_values=[0])
            yield (cirq.Z(q) ** (-1 / 2 ** (1 + i))).controlled_by(*arctan_sign, control_values=[1])

        yield arctan_op**-1
        yield self.Y.on_registers(**qubit_regs) ** -1

        cirq_infra.qfree([*arctan_sign, *arctan_target])


@frozen
class MeanEstimationWalk(cirq_infra.GateWithRegisters):
    code: CodeForRandomVariable
    arctan_bitsize: int
    cv: Tuple[int, ...] = field(default=())
    power: int = 1

    @cv.validator
    def _validate_cv(self, attribute, value):
        assert value in [(), (0,), (1,)]

    @cached_property
    def reflect(self) -> rup.ReflectionUsingPrepare:
        return rup.ReflectionUsingPrepare(
            self.code.P, control_val=None if self.cv == () else self.cv[0]
        )

    @cached_property
    def select(self) -> PhaseOracle:
        return PhaseOracle(self.code.Y, self.arctan_bitsize)

    @cached_property
    def control_registers(self) -> cirq_infra.Registers:
        return self.code.Y.control_registers

    @cached_property
    def selection_registers(self) -> cirq_infra.SelectionRegisters:
        return self.code.Y.selection_registers

    @cached_property
    def target_registers(self) -> cirq_infra.Registers:
        return self.code.Y.target_registers

    @cached_property
    def registers(self) -> cirq_infra.Registers:
        return cirq_infra.Registers(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )

    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        select_reg = {reg.name: qubit_regs[reg.name] for reg in self.select.registers}
        reflect_reg = {reg.name: qubit_regs[reg.name] for reg in self.reflect.registers}
        select_op = self.select.on_registers(**select_reg)
        reflect_op = self.reflect.on_registers(**reflect_reg)
        for _ in range(self.power):
            yield select_op
            yield reflect_op

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = [] if self.cv == () else [["@(0)", "@"][self.cv[0]]]
        wire_symbols += ['W'] * (self.registers.bitsize - self.control_registers.bitsize)
        wire_symbols[-1] = f'MW^{self.power}' if self.power != 1 else 'MW'
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def controlled(
        self,
        num_controls: int = None,
        control_values: Sequence[Union[int, Collection[int]]] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> 'MeanEstimationWalk':
        if num_controls is None:
            num_controls = 1
        if control_values is None:
            control_values = [1] * num_controls
        if len(control_values) == 1 and self._control_val is None:
            return MeanEstimationWalk(
                CodeForRandomVariable(
                    Y=self.code.Y.controlled(control_values=control_values), P=self.code.P
                ),
                self.arctan_bitsize,
                cv=control_values,
                power=self.power,
            )
        raise NotImplementedError(f'Cannot create a controlled version of {self}')

    def with_power(self, new_power: int) -> 'QubitizationWalkOperator':
        return MeanEstimationWalk(self.code, cv=self.cv, power=new_power)

    def __pow__(self, power: int):
        return self.with_power(self.power * power)
