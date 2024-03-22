#  Copyright 2024 Google LLC
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
from typing import Tuple

import attrs
import cirq

from qualtran import Bloq, GateWithRegisters, QFxp, Register, Signature
from qualtran.bloqs.phase_estimation.lp_resource_state import LPResourceState
from qualtran.bloqs.qft.qft_text_book import QFTTextBook
from qualtran.bloqs.qubitization_walk_operator import QubitizationWalkOperator


@attrs.frozen
class QubitizationQPE(GateWithRegisters):
    """Heisenberg limited phase estimation circuit for learning eigenphase of `walk`.

    The Bloq yields an OPTREE to construct Heisenberg limited phase estimation circuit
    for learning eigenphases of the `walk` operator with `m` bits of accuracy. The
    circuit is implemented as given in Fig.2 of Ref-1.

    Args:
        walk: Qubitization walk operator.
        m: Number of bits of accuracy for phase estimation.

    Ref:
        1) [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity]
            (https://arxiv.org/abs/1805.03662)
            Fig. 2
    """

    walk: QubitizationWalkOperator
    m_bits: int
    state_prep: Bloq = attrs.field()
    inverse_qft: Bloq = attrs.field()

    @state_prep.default
    def _default_state_prep(self):
        return LPResourceState(self.m_bits)

    @inverse_qft.default
    def _default_inverse_qft(self):
        return QFTTextBook(self.m_bits, with_reverse=True).adjoint()

    def __attrs_post_init__(self):
        assert self.state_prep.signature.n_qubits() == self.m_bits

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return tuple(self.walk.signature)

    @cached_property
    def phase_registers(self) -> Tuple[Register, ...]:
        return (Register('qpe_reg', QFxp(self.m_bits, self.m_bits)),)

    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.phase_registers, *self.target_registers])

    def pretty_name(self) -> str:
        return f'QubitizationQPE[{self.m}]'

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs
    ) -> cirq.OP_TREE:
        walk_regs = {reg.name: quregs[reg.name] for reg in self.walk.signature}
        reflect_regs = {reg.name: walk_regs[reg.name] for reg in self.walk.reflect.signature}

        reflect_controlled = self.walk.reflect.controlled(control_values=[0])
        walk_controlled = self.walk.controlled(control_values=[0])

        qpre_reg = quregs['qpe_reg']

        yield self.state_prep.on(*qpre_reg)
        yield walk_controlled.on_registers(**walk_regs, control=qpre_reg[-1])
        walk = self.walk
        for i in range(self.m_bits - 1, 0, -1):
            yield reflect_controlled.on_registers(control=qpre_reg[i], **reflect_regs)
            yield walk.on_registers(**walk_regs)
            walk = walk**2
            yield reflect_controlled.on_registers(control=qpre_reg[i], **reflect_regs)
        yield self.inverse_qft.on(*qpre_reg)
