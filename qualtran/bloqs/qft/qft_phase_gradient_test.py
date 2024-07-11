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
import sympy

from qualtran import GateWithRegisters, Signature
from qualtran.bloqs.qft.qft_phase_gradient import _qft_phase_gradient_small, QFTPhaseGradient
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from qualtran.symbolics.math_funcs import smax
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
        bb.add(PhaseGradientState(self.bitsize).adjoint(), phase_grad=phase_grad)
        return {'q': q}


@pytest.mark.parametrize('n', [2, 3, 4, 5])
@pytest.mark.parametrize('without_reverse', [True, False])
def test_qft_with_phase_gradient(n: int, without_reverse: bool):
    qft_bloq = TestQFTWithPhaseGradient(n, not without_reverse)
    qft_cirq = cirq.QuantumFourierTransformGate(n, without_reverse=without_reverse)

    np.testing.assert_allclose(cirq.unitary(qft_bloq), cirq.unitary(qft_cirq))
    np.testing.assert_allclose(cirq.unitary(qft_bloq**-1), cirq.unitary(qft_cirq**-1))

    assert_valid_bloq_decomposition(qft_bloq)


@pytest.mark.parametrize('n', [10, 123])
def test_qft_phase_gradient_t_complexity(n: int):
    qft_bloq = QFTPhaseGradient(n)
    n_symb = sympy.symbols('n')
    symbolic_qft_bloq = QFTPhaseGradient(bitsize=n_symb)
    plus_equals_prod_cost = 8 * smax(n_symb // 2, n_symb - n_symb // 2) ** 2
    symbolic_qft_t_complexity = symbolic_qft_bloq.t_complexity().t
    assert symbolic_qft_t_complexity == plus_equals_prod_cost * sympy.log(n_symb, 2)
    qft_t_complexity = qft_bloq.t_complexity()
    assert qft_t_complexity.t <= 8 * (n**2)
    assert qft_t_complexity.rotations == 0


def test_qft_phase_gradient_small_auto(bloq_autotester):
    bloq_autotester(_qft_phase_gradient_small)
