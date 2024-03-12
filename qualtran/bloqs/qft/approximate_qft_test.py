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
import math
from typing import Dict, TYPE_CHECKING

import attrs
import cirq
import numpy as np
import pytest

from qualtran import GateWithRegisters, Signature
from qualtran.bloqs.arithmetic.multiplication import PlusEqualProduct
from qualtran.bloqs.qft.approximate_qft import ApproximateQFT
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from qualtran.testing import assert_valid_bloq_decomposition

if TYPE_CHECKING:
    from qualtran import BloqBuilder, SoquetT


@attrs.frozen
class TestApproximateQFT(GateWithRegisters):
    bitsize: int
    with_reverse: bool

    @property
    def signature(self) -> 'Signature':
        return Signature.build(q=self.bitsize)

    def build_composite_bloq(self, bb: 'BloqBuilder', *, q: 'SoquetT') -> Dict[str, 'SoquetT']:
        def b(n):
            return n

        phase_grad = bb.add(PhaseGradientState(b(self.bitsize), exponent=-1))

        q, phase_grad = bb.add(
            ApproximateQFT(self.bitsize, b, self.with_reverse), q=q, phase_grad=phase_grad
        )
        bb.add(PhaseGradientState(self.bitsize).adjoint(), phase_grad=phase_grad)
        return {'q': q}


@pytest.mark.parametrize('n', [2, 3, 4, 5])
@pytest.mark.parametrize('without_reverse', [True, False])
def test_qft_with_phase_gradient(n: int, without_reverse: bool):
    qft_bloq = TestApproximateQFT(n, not without_reverse)
    qft_cirq = cirq.QuantumFourierTransformGate(n, without_reverse=without_reverse)
    np.testing.assert_allclose(cirq.unitary(qft_bloq), cirq.unitary(qft_cirq))
    np.testing.assert_allclose(cirq.unitary(qft_bloq**-1), cirq.unitary(qft_cirq**-1))

    assert_valid_bloq_decomposition(qft_bloq)


@pytest.mark.parametrize('n', [10, 123])
def test_qft_text_book_t_complexity(n: int):
    qft_bloq = ApproximateQFT(n, with_reverse=False)

    def f(n, b):
        t_complexity = 0
        for i in range(1, n):
            t_complexity += PlusEqualProduct(min(i, b - 1), 1, min(i, b - 1) + 1).t_complexity().t
        return t_complexity

    qft_t_complexity = qft_bloq.t_complexity()
    b = math.ceil(math.log2(n))

    assert qft_t_complexity.t == f(n, b) <= 8 * n * (math.log2(n) ** 2)
    assert qft_t_complexity.rotations == 0
