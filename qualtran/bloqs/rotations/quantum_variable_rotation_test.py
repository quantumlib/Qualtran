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
import attrs
import cirq
import numpy as np
import pytest
from fxpmath import Fxp
from numpy._typing import NDArray

from qualtran import GateWithRegisters, QFxp, Register, Signature
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientUnitary
from qualtran.bloqs.rotations.quantum_variable_rotation import (
    _qvr_phase_gradient,
    _qvr_zpow,
    QvrPhaseGradient,
)


def test_qvr_zpow_auto(bloq_autotester):
    bloq_autotester(_qvr_zpow)


def test_qvr_phase_gradient_auto(bloq_autotester):
    bloq_autotester(_qvr_phase_gradient)


@attrs.frozen
class TestQvrPhaseGradient(GateWithRegisters):
    cost_reg: Register
    gamma: float
    eps: float

    @property
    def signature(self) -> Signature:
        return Signature([self.cost_reg])

    @property
    def qvr(self) -> QvrPhaseGradient:
        return QvrPhaseGradient(self.cost_reg, self.gamma, self.eps)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        x = quregs[self.cost_reg.name]
        phase_grad = context.qubit_manager.qalloc(self.qvr.b_grad)
        yield cirq.H.on_each(*phase_grad)
        yield PhaseGradientUnitary(self.qvr.b_grad, -1).on(*phase_grad)
        yield self.qvr.on_registers(x=x, phase_grad=phase_grad)
        yield PhaseGradientUnitary(self.qvr.b_grad, +1).on(*phase_grad)
        yield cirq.H.on_each(*phase_grad)


@pytest.mark.slow
@pytest.mark.parametrize('normalize', [True, False])
def test_qvr_phase_gradient_cost_reg_greater_than_b_grad(normalize: bool):
    n, gamma, eps = (13, (2**20 - 1) / 2**20, 1e-1)
    # Note that `gamma` is of the form `0.111111111` and thus has worst case complexity
    # in terms of adding errors
    assert Fxp(gamma, signed=False).bin() == '1' * 20
    cost_reg = Register('x', QFxp(n, n * int(normalize), signed=False))
    qvr_test = TestQvrPhaseGradient(cost_reg, gamma, eps)

    assert qvr_test.qvr.b_grad < n

    q = cirq.LineQubit.range(n)
    circuit = cirq.Circuit(cirq.H.on_each(*q), qvr_test.on(*q))
    final_state = cirq.Simulator().simulate(circuit).final_state_vector
    expected_state = np.array(
        [
            np.exp(1j * 2 * np.pi * gamma * x / (2 ** (int(normalize) * n))) / np.sqrt(2**n)
            for x in range(2**n)
        ]
    )
    np.testing.assert_allclose(final_state, expected_state, atol=eps)


@pytest.mark.parametrize(
    'gamma, expected_additions, normalize',
    [
        (1 / 2**4, 1, True),
        (1 / 2**4, 1, False),
        (
            (2**20 - 1) / 2**20,
            4,
            True,
        ),  # If you normalize, only O(log(1/eps)) bits are needed for gamma.
        (
            (2**20 - 1) / 2**20,
            11,
            False,
        ),  # If you don't normalize, all 20 bits are needed for gamma.
    ],
)
@pytest.mark.parametrize('n, eps', [(14, 1e-1)])
def test_qvr_phase_gradient_t_complexity(
    n: int, eps: float, gamma: float, expected_additions: int, normalize: bool
):
    # Note that `gamma` is of the form `0.111111111` and thus has worst case complexity
    # in terms of adding errors
    cost_reg = Register('x', QFxp(n, n * int(normalize), signed=False))
    qvr = QvrPhaseGradient(cost_reg, gamma, eps)
    assert qvr.b_grad < n
    assert qvr.t_complexity().t == 4 * expected_additions * (qvr.b_grad - 2)
