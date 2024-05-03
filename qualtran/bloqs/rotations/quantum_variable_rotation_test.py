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
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]  # type: ignore[type-var]
    ) -> cirq.OP_TREE:
        x = quregs[self.cost_reg.name]
        phase_grad = context.qubit_manager.qalloc(int(self.qvr.b_grad))
        yield cirq.H.on_each(*phase_grad)
        yield PhaseGradientUnitary(int(self.qvr.b_grad), -1).on(*phase_grad)
        yield self.qvr.on_registers(x=x, phase_grad=phase_grad)
        yield PhaseGradientUnitary(int(self.qvr.b_grad), +1).on(*phase_grad)
        yield cirq.H.on_each(*phase_grad)


def trace_distance(u, v):
    dist = 1 - np.abs(np.trace(u @ v.conj().T)) / 2
    if np.isclose(dist, 0):
        dist = 0
    return np.sqrt(dist)


@pytest.mark.parametrize('gamma', [1 / 2, 1 / 4, 1 / 8, 1 / 16, 15 / 16])
@pytest.mark.parametrize('eps', [1e-2, 1e-3, 1e-4, 1e-5])
def test_qvr_phase_gradient_unitary(gamma: float, eps: float):
    zpow = cirq.Z**gamma
    zpow_qvr = TestQvrPhaseGradient(Register('x', QFxp(1, 1, signed=False)), gamma, eps)
    assert trace_distance(cirq.unitary(zpow), cirq.unitary(zpow_qvr)) < eps


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


@pytest.mark.parametrize('n', range(2, 50))
def test_eps_set_by_cost_size(n: int):
    # We want `gamma` to have `n` set bits (`0.11111...11`) for the worst case comparison.
    num, den = 2**n - 1, 2**n
    gamma = num / den

    eps = 2 * np.pi / (2**n)
    qvr = QvrPhaseGradient.from_bitsize(n, gamma, eps=eps)
    assert qvr.b_phase == n
    assert qvr.gamma_dtype.bitsize == n
    expected_num_additions = (n + 2) // 2
    assert qvr.num_additions == expected_num_additions
    expected_b_grad = np.ceil(np.log2(expected_num_additions)) + n
    assert qvr.b_grad_via_formula == expected_b_grad
    # `b_grad_via_fxp_optimization` can be higher than `b_grad_via_formula` because we haven't
    # implemented the optimization where we represent `gamma` is represented as a sum of positive
    # and negative terms so that the total number additions is `(gamma_bitsize + 2) // 2` instead
    # of `gamma_bitsize`. Since `b_grad_via_fxp_optimization` is computed by actually doing the
    # additions into the target register, the higher number of additions result in a higher error
    # and thus need (at-most 1) more bit.
    assert qvr.b_grad_via_fxp_optimization <= qvr.b_grad_via_formula + 1


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
