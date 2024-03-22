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
import math
from functools import cached_property
from typing import Tuple

import attrs
import cirq

from qualtran import Bloq, GateWithRegisters, Register, Signature
from qualtran.bloqs.basic_gates import Hadamard, OnEach
from qualtran.bloqs.qft.qft_text_book import QFTTextBook


@attrs.frozen
class KitaevQPE(GateWithRegisters):
    r"""Kitaev's Phase Estimation algorithm

    Originally introduced by Kitaev in https://arxiv.org/abs/quant-ph/9511026.

    Args:
        m_bits: Bitsize of the phase register to be used during phase estimation
        unitary: A cirq Gate representing the unitary to run the phase estimation protocol on.
    """

    unitary: Bloq
    m_bits: int
    state_prep: Bloq = attrs.field()
    inverse_qft: Bloq = attrs.field()

    @state_prep.default
    def _default_state_prep(self):
        return OnEach(self.m_bits, Hadamard())

    @inverse_qft.default
    def _default_inverse_qft(self):
        return QFTTextBook(self.m_bits, with_reverse=True).adjoint()

    def __attrs_post_init__(self):
        assert self.state_prep.signature.n_qubits() == self.m_bits

    @classmethod
    def from_precision_and_eps(cls, unitary: Bloq, precision: int, eps: float):
        r"""Obtain accurate estimate of $\phi$ to $precision$ bits with $1-eps$ success probability.

        Uses Eq 5.35 from Neilson and Chuang to estimate the size of phase register s.t. we can
        estimate the phase $\phi$ to $precision$ bits of accuracy with probability at least
        $1 - eps$.

        $$
            t = n + ceil(\log(2 + \frac{1}{2\eps}))
        $$

        Args:
            unitary: Unitary operation to obtain phase estimate of.
            precision: Number of bits of precision
            eps: Probability of success.
        """
        m_bits = precision + math.ceil(math.log(2 + 1 / (2 * eps)))
        return KitaevQPE(m_bits=m_bits, unitary=unitary)

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return tuple(self.unitary.signature)

    @cached_property
    def phase_registers(self) -> Tuple[Register, ...]:
        return tuple(Signature.build(qpe_reg=self.m_bits))

    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.phase_registers, *self.target_registers])

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs
    ) -> cirq.OP_TREE:
        target_quregs = {reg.name: quregs[reg.name] for reg in self.target_registers}
        unitary_op = self.unitary.on_registers(**target_quregs)

        phase_qubits = quregs['qpe_reg']

        yield self.state_prep.on(*phase_qubits)
        for i, qbit in enumerate(phase_qubits[::-1]):
            yield cirq.pow(unitary_op.controlled_by(qbit), 2**i)
        yield self.inverse_qft.on(*phase_qubits)
