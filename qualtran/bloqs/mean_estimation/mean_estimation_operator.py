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
from typing import Iterator, Optional, Tuple

import attrs
import cirq
from numpy.typing import NDArray

from qualtran import CtrlSpec, Register, Signature
from qualtran._infra.gate_with_registers import GateWithRegisters, total_bits
from qualtran._infra.single_qubit_controlled import SpecializedSingleQubitControlledExtension
from qualtran.bloqs.mean_estimation.complex_phase_oracle import ComplexPhaseOracle
from qualtran.bloqs.multiplexers.select_base import SelectOracle
from qualtran.bloqs.reflections.reflection_using_prepare import ReflectionUsingPrepare
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle


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
class MeanEstimationOperator(GateWithRegisters, SpecializedSingleQubitControlledExtension):  # type: ignore[misc]
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
    control_val: Optional[int] = None
    arctan_bitsize: int = 32

    @cached_property
    def reflect(self) -> ReflectionUsingPrepare:
        return ReflectionUsingPrepare(
            self.code.synthesizer, global_phase=-1, control_val=self.control_val
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
    ) -> Iterator[cirq.OP_TREE]:
        select_reg = {reg.name: quregs[reg.name] for reg in self.select.signature}
        reflect_reg = {reg.name: quregs[reg.name] for reg in self.reflect.signature}
        yield self.select.on_registers(**select_reg)
        yield self.reflect.on_registers(**reflect_reg)

    def get_single_qubit_controlled_bloq(self, control_val: int) -> 'MeanEstimationOperator':
        c_encoder = self.code.encoder.controlled(ctrl_spec=CtrlSpec(cvs=control_val))
        assert isinstance(c_encoder, SelectOracle)
        c_code = attrs.evolve(self.code, encoder=c_encoder)
        return attrs.evolve(self, code=c_code, control_val=control_val)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        if self.control_val is not None:
            wire_symbols.append("@" if self.control_val == 1 else "(0)")
        wire_symbols += ['U_ko'] * (total_bits(self.signature) - total_bits(self.control_registers))
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)
