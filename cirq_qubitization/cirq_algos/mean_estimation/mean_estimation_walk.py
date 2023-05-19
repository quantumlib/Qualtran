from functools import cached_property
from typing import Collection, Optional, Sequence, Tuple, Union

import cirq
from attrs import field, frozen

from cirq_qubitization import cirq_infra
from cirq_qubitization.cirq_algos import reflection_using_prepare as rup
from cirq_qubitization.cirq_algos import select_and_prepare as sp
from cirq_qubitization.cirq_algos.mean_estimation import complex_phase_oracle


@frozen
class CodeForRandomVariable:
    r"""We say we have the "code" for a random variable y iff we have unitaries P and Y s.t.
    $$
    synthesizer|0> = \sum_{w \in W} \sqrt{p(w)} |w> |garbage_{w}>
    $$
    and
    $$
    encoder|w>|0^b> = |w>|y(w)>
    $$
    where b is the number of bits required to encode the real range of random variable y.

    References:
        https://arxiv.org/abs/2208.07544, Definition 2.2 for synthesizer (P) and
        Definition 2.10 for encoder (Y).
    """

    synthesizer: sp.PrepareOracle
    encoder: sp.SelectOracle


@frozen
class MeanEstimationOperator(cirq_infra.GateWithRegisters):
    r"""Mean estimation operator $U=REFL_{p} ROT_{y}$ as per Sec 3.1 of arxiv.org:2208.07544."""

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
            self.code.synthesizer, control_val=None if self.cv == () else self.cv[0]
        )

    @cached_property
    def select(self) -> complex_phase_oracle.ComplexPhaseOracle:
        return complex_phase_oracle.ComplexPhaseOracle(self.code.encoder, self.arctan_bitsize)

    @cached_property
    def control_registers(self) -> cirq_infra.Registers:
        return self.code.encoder.control_registers

    @cached_property
    def selection_registers(self) -> cirq_infra.SelectionRegisters:
        return self.code.encoder.selection_registers

    @cached_property
    def target_registers(self) -> cirq_infra.Registers:
        return self.code.encoder.target_registers

    @cached_property
    def registers(self) -> cirq_infra.Registers:
        return cirq_infra.Registers(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )

    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        print(qubit_regs)
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
    ) -> 'MeanEstimationOperator':
        if num_controls is None:
            num_controls = 1
        if control_values is None:
            control_values = [1] * num_controls
        if len(control_values) == 1 and self._control_val is None:
            return MeanEstimationOperator(
                CodeForRandomVariable(
                    Y=self.code.encoder.controlled(control_values=control_values),
                    synthesizer=self.code.synthesizer,
                ),
                self.arctan_bitsize,
                cv=control_values,
                power=self.power,
            )
        raise NotImplementedError(f'Cannot create a controlled version of {self}')

    def with_power(self, new_power: int) -> 'MeanEstimationOperator':
        return MeanEstimationOperator(self.code, cv=self.cv, power=new_power)

    def __pow__(self, power: int):
        return self.with_power(self.power * power)
