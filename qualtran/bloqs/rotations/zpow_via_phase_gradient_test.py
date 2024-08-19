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
import pytest
from attrs import frozen

from qualtran import Bloq, BloqBuilder, Signature, Soquet, SoquetT
from qualtran.bloqs.basic_gates import ZPowGate
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from qualtran.bloqs.rotations.zpow_via_phase_gradient import (
    _zpow_const_via_phase_grad,
    _zpow_const_via_phase_grad_symb_angle,
    _zpow_const_via_phase_grad_symb_prec,
    ZPowConstViaPhaseGradient,
)
from qualtran.linalg.testing import assert_unitaries_equivalent_upto_global_phase
from qualtran.resource_counting import GateCounts, get_cost_value, QECGatesCost


@pytest.mark.parametrize(
    "bloq",
    [
        _zpow_const_via_phase_grad,
        _zpow_const_via_phase_grad_symb_prec,
        _zpow_const_via_phase_grad_symb_angle,
    ],
)
def test_examples(bloq_autotester, bloq):
    if bloq_autotester.check_name == 'serialize':
        pytest.skip()

    bloq_autotester(bloq)


def test_cost():
    bloq = _zpow_const_via_phase_grad()

    costs = get_cost_value(bloq, QECGatesCost())
    b_grad = bloq.phase_grad_bitsize
    assert costs == GateCounts(toffoli=b_grad - 2, clifford=4)


def test_cost_symbolic_angle():
    bloq = _zpow_const_via_phase_grad_symb_angle()

    costs = get_cost_value(bloq, QECGatesCost())
    b_grad = bloq.phase_grad_bitsize
    assert costs == GateCounts(toffoli=b_grad - 2, clifford=80)


def test_cost_symbolic_prec():
    bloq = _zpow_const_via_phase_grad_symb_prec()

    costs = get_cost_value(bloq, QECGatesCost())
    b_grad = bloq.phase_grad_bitsize
    assert costs == GateCounts(toffoli=b_grad - 2, clifford=2 * b_grad)


@frozen
class TestZPowUsingPhaseGradient(Bloq):
    exponent: float
    phase_grad_bitsize: int

    @property
    def signature(self) -> 'Signature':
        return Signature.build(q=1)

    def build_composite_bloq(self, bb: 'BloqBuilder', q: 'Soquet') -> dict[str, 'SoquetT']:
        phase_grad = bb.add(PhaseGradientState(self.phase_grad_bitsize))
        q, phase_grad = bb.add(
            ZPowConstViaPhaseGradient(self.exponent, self.phase_grad_bitsize),
            q=q,
            phase_grad=phase_grad,
        )
        bb.add(PhaseGradientState(self.phase_grad_bitsize).adjoint(), phase_grad=phase_grad)
        return {'q': q}


@pytest.mark.parametrize(
    ("exponent", "phase_grad_bitsize"),
    [(3 / 4, 3), (3 / 4, 4), pytest.param(0.175, 7, marks=pytest.mark.slow)],
)
def test_unitary(exponent: float, phase_grad_bitsize: int):
    bloq = TestZPowUsingPhaseGradient(exponent, phase_grad_bitsize)
    actual = bloq.tensor_contract()
    expected = ZPowGate(exponent).tensor_contract()
    assert_unitaries_equivalent_upto_global_phase(actual, expected, atol=2**-phase_grad_bitsize)
