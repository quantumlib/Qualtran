#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from functools import cached_property
from typing import Collection, Optional, Sequence, Tuple, Union

import attrs
import cirq
from numpy.typing import NDArray

from qualtran import GateWithRegisters, Register, Signature
from qualtran._infra.gate_with_registers import total_bits
from qualtran.bloqs.mean_estimation.complex_phase_oracle import ComplexPhaseOracle
from qualtran.bloqs.reflection_using_prepare import ReflectionUsingPrepare
from qualtran.bloqs.select_and_prepare import PrepareOracle, SelectOracle


@attrs.frozen
class CodeForRandomVariable:
    r"""A collection of `encoder` and `synthesizer` for a random variable y.

    We say we have "the code" for a random variable $y$ defined on a probability space
    $(W, p)$ if we have both, a synthesizer and an encoder defined as follows:

    The synthesizer is responsible to "prepare" the state
    $\sum_{w \in W} \sqrt{p(w)} |w> |garbage_{w}>$ on the "selection register" $w$ and potentially
    using a "junk register" corresponding to $|garbage_{w}>$. Thus, for convenience, the synthesizer
    follows the LCU PREPARE Oracle API.
    $$
    synthesizer|0> = \sum_{w \in W} \sqrt{p(w)} |w> |garbage_{w}>
    $$


    The encoder is responsible to encode the value of random variable $y(w)$ in a "target register"
    when the corresponding "selection register" stores integer $w$. Thus, for convenience, the
    encoder follows the LCU SELECT Oracle API.
    $$
    encoder|w>|0^b> = |w>|y(w)>
    $$
    where b is the number of bits required to encode the real range of random variable y.

    References:
        https://arxiv.org/abs/2208.07544, Definition 2.2 for synthesizer (P) and
        Definition 2.10 for encoder (Y).
    """

    synthesizer: PrepareOracle
    encoder: SelectOracle

    def __attrs_post_init__(self):
        assert self.synthesizer.selection_registers == self.encoder.selection_registers


@attrs.frozen
class MeanEstimationOperator(GateWithRegisters):
    r"""Mean estimation operator $U=REFL_{p} ROT_{y}$ as per Sec 3.1 of arxiv.org:2208.07544.

    The MeanEstimationOperator (aka KO Operator) expects `CodeForRandomVariable` to specify the
    synthesizer and encoder, that follows LCU SELECT/PREPARE API for convenience. It is composed
    of two unitaries:

        - REFL_{p}: Reflection around the state prepared by synthesizer $P$. It applies the unitary
            $P(2|0><0| - I)P^{\dagger}$.
        - ROT_{y}: Applies a complex phase $\exp(i * -2\arctan{y_{w}})$ when the selection register
            stores $w$. This is achieved by using the encoder to encode $y(w)$ in a temporary target
            register.

    Note that both $REFL_{p}$ and $ROT_{y}$ only act upon a selection register, thus mean estimation
    operator expects only a selection register (and a control register, for a controlled version for
    phase estimation).
    """

    code: CodeForRandomVariable
    cv: Tuple[int, ...] = attrs.field(
        converter=lambda v: (v,) if isinstance(v, int) else tuple(v), default=()
    )
    power: int = 1
    arctan_bitsize: int = 32

    @cv.validator
    def _validate_cv(self, attribute, value):
        assert value in [(), (0,), (1,)]

    @cached_property
    def reflect(self) -> ReflectionUsingPrepare:
        return ReflectionUsingPrepare(
            self.code.synthesizer,
            control_val=None if self.cv == () else self.cv[0],
            global_phase=-1,
        )

    @cached_property
    def select(self) -> ComplexPhaseOracle:
        return ComplexPhaseOracle(self.code.encoder, self.arctan_bitsize)

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return self.code.encoder.control_registers

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return self.code.encoder.selection_registers

    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.control_registers, *self.selection_registers])

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        select_reg = {reg.name: quregs[reg.name] for reg in self.select.signature}
        reflect_reg = {reg.name: quregs[reg.name] for reg in self.reflect.signature}
        select_op = self.select.on_registers(**select_reg)
        reflect_op = self.reflect.on_registers(**reflect_reg)
        for _ in range(self.power):
            yield select_op
            yield reflect_op

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = [] if self.cv == () else [["@(0)", "@"][self.cv[0]]]
        wire_symbols += ['U_ko'] * (total_bits(self.signature) - total_bits(self.control_registers))
        if self.power != 1:
            wire_symbols[-1] = f'U_ko^{self.power}'
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def controlled(
        self,
        num_controls: Optional[int] = None,
        control_values: Optional[
            Union[cirq.ops.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> 'MeanEstimationOperator':
        if num_controls is None:
            num_controls = 1
        if control_values is None:
            control_values = [1] * num_controls
        if (
            isinstance(control_values, Sequence)
            and len(control_values) == 1
            and isinstance(control_values[0], int)
            and not self.cv
        ):
            c_select = self.code.encoder.controlled(control_values=control_values)
            assert isinstance(c_select, SelectOracle)
            return MeanEstimationOperator(
                CodeForRandomVariable(encoder=c_select, synthesizer=self.code.synthesizer),
                cv=self.cv + (control_values[0],),
                power=self.power,
                arctan_bitsize=self.arctan_bitsize,
            )
        raise NotImplementedError(
            f'Cannot create a controlled version of {self} with control_values={control_values}.'
        )

    def with_power(self, new_power: int) -> 'MeanEstimationOperator':
        return MeanEstimationOperator(
            self.code, cv=self.cv, power=new_power, arctan_bitsize=self.arctan_bitsize
        )

    def __pow__(self, power: int):
        return self.with_power(self.power * power)
