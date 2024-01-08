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
from typing import Dict, TYPE_CHECKING

import attrs
import cirq
import numpy as np
import pytest

from qualtran import GateWithRegisters, Signature
from qualtran.bloqs.arithmetic.multiplication import PlusEqualProduct
from qualtran.bloqs.qft.qft_phase_gradient import QFTPhaseGradient
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from qualtran.testing import assert_valid_bloq_decomposition

if TYPE_CHECKING:
    from qualtran import BloqBuilder, SoquetT


@attrs.frozen
class TestQFTWithPhaseGradient(GateWithRegisters):
    bitsize: int
    with_reverse: bool

    @property
    def signature(self) -> 'Signature':
        return Signature.build(q=self.bitsize)

    def build_composite_bloq(self, bb: 'BloqBuilder', *, q: 'SoquetT') -> Dict[str, 'SoquetT']:
        phase_grad = bb.add(PhaseGradientState(self.bitsize))
        q, phase_grad = bb.add(
            QFTPhaseGradient(self.bitsize, self.with_reverse), q=q, phase_grad=phase_grad
        )
        bb.add(PhaseGradientState(self.bitsize, adjoint=True), phase_grad=phase_grad)
        return {'q': q}


@pytest.mark.parametrize('n', [2, 3, 4, 5])
@pytest.mark.parametrize('without_reverse', [True, False])
def test_qft_with_phase_gradient(n: int, without_reverse: bool):
    qft_bloq = TestQFTWithPhaseGradient(n, not without_reverse)
    qft_cirq = cirq.QuantumFourierTransformGate(n, without_reverse=without_reverse)

    assert np.allclose(cirq.unitary(qft_bloq), cirq.unitary(qft_cirq))
    assert np.allclose(cirq.unitary(qft_bloq**-1), cirq.unitary(qft_cirq**-1))

    assert_valid_bloq_decomposition(qft_bloq)


@pytest.mark.parametrize('n', [10, 100, 500])
def test_qft_text_book_t_complexity(n: int):
    qft_bloq = QFTPhaseGradient(n)
    print(qft_bloq.t_complexity())

    def f(x):
        if x == 1:
            return 0
        return f(x // 2) + f(x - x // 2) + PlusEqualProduct(x // 2, x - x // 2, x).t_complexity().t

    qft_t_complexity = qft_bloq.t_complexity()
    assert qft_t_complexity.t == f(n) <= 8 * (n**2)
    assert qft_t_complexity.rotations == 0
