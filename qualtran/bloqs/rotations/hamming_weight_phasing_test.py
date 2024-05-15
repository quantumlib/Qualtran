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

import qualtran.testing as qlt_testing
from qualtran import GateWithRegisters, Signature
from qualtran.bloqs.rotations.hamming_weight_phasing import (
    HammingWeightPhasing,
    HammingWeightPhasingViaPhaseGradient,
)
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from qualtran.cirq_interop.testing import GateHelper
from qualtran.resource_counting.generalizers import (
    cirq_to_bloqs,
    generalize_rotation_angle,
    ignore_split_join,
)
from qualtran.symbolics import SymbolicInt

if TYPE_CHECKING:
    from qualtran import BloqBuilder, SoquetT


@pytest.mark.parametrize('n', [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize('theta', [1 / 10, 1 / 5, 1 / 7, np.pi / 2])
def test_hamming_weight_phasing(n: int, theta: float):
    gate = HammingWeightPhasing(n, theta)
    qlt_testing.assert_valid_bloq_decomposition(gate)
    qlt_testing.assert_equivalent_bloq_counts(
        gate, [ignore_split_join, cirq_to_bloqs, generalize_rotation_angle]
    )

    assert gate.t_complexity().rotations == n.bit_length()
    assert gate.t_complexity().t == 4 * (n - n.bit_count())

    gh = GateHelper(gate)
    sim = cirq.Simulator(dtype=np.complex128)
    initial_state = cirq.testing.random_superposition(dim=2**n, random_state=12345)
    state_prep = cirq.Circuit(cirq.StatePreparationChannel(initial_state).on(*gh.quregs['x']))
    brute_force_phasing = cirq.Circuit(state_prep, (cirq.Z**theta).on_each(*gh.quregs['x']))
    expected_final_state = sim.simulate(brute_force_phasing).final_state_vector

    hw_phasing = cirq.Circuit(state_prep, HammingWeightPhasing(n, theta).on(*gh.quregs['x']))
    hw_final_state = sim.simulate(hw_phasing).final_state_vector
    assert np.allclose(expected_final_state, hw_final_state, atol=1e-7)


@pytest.mark.parametrize('n', [10, 32, 64, 100, 1024])
def test_hamming_weight_phasing_large(n: int):
    gate = HammingWeightPhasing(n, 1 / 10)
    qlt_testing.assert_valid_bloq_decomposition(gate)
    qlt_testing.assert_equivalent_bloq_counts(gate, [ignore_split_join, generalize_rotation_angle])

    assert gate.t_complexity().rotations == n.bit_length()
    assert gate.t_complexity().t == 4 * (n - n.bit_count())


@attrs.frozen
class TestHammingWeightPhasingViaPhaseGradient(GateWithRegisters):
    bitsize: int
    exponent: float
    eps: float

    @property
    def signature(self) -> 'Signature':
        return Signature.build(x=self.bitsize)

    @property
    def b_grad(self) -> SymbolicInt:
        return HammingWeightPhasingViaPhaseGradient(self.bitsize, self.exponent, self.eps).b_grad

    def build_composite_bloq(self, bb: 'BloqBuilder', *, x: 'SoquetT') -> Dict[str, 'SoquetT']:
        b_grad = self.b_grad
        phase_grad = bb.add(PhaseGradientState(b_grad))
        x, phase_grad = bb.add(
            HammingWeightPhasingViaPhaseGradient(self.bitsize, self.exponent, self.eps),
            x=x,
            phase_grad=phase_grad,
        )
        bb.add(PhaseGradientState(b_grad).adjoint(), phase_grad=phase_grad)
        return {'x': x}


@pytest.mark.slow
@pytest.mark.parametrize('n', [2, 3])
@pytest.mark.parametrize(
    'theta, eps', [(1, 1e-1), (0.5, 1e-2), (1 / 10, 1e-3), (1.20345, 1e-3), (-1.1934341, 1e-3)]
)
def test_hamming_weight_phasing_via_phase_gradient(n: int, theta: float, eps: float):
    gate = TestHammingWeightPhasingViaPhaseGradient(n, theta, eps)
    qlt_testing.assert_valid_bloq_decomposition(gate)

    gh = GateHelper(gate)
    sim = cirq.Simulator(dtype=np.complex128)
    initial_state = cirq.testing.random_superposition(dim=2**n, random_state=12345)
    state_prep = cirq.Circuit(cirq.StatePreparationChannel(initial_state).on(*gh.quregs['x']))
    brute_force_phasing = cirq.Circuit(state_prep, (cirq.Z**theta).on_each(*gh.quregs['x']))
    expected_final_state = sim.simulate(brute_force_phasing).final_state_vector

    hw_phasing = cirq.Circuit(state_prep, gh.operation)
    hw_final_state = sim.simulate(hw_phasing).final_state_vector
    np.testing.assert_allclose(expected_final_state, hw_final_state, atol=eps)


@pytest.mark.parametrize('n, theta, eps', [(5_000, 1 / 100, 1e-1)])
def test_hamming_weight_phasing_via_phase_gradient_t_complexity(n: int, theta: float, eps: float):
    hwp_t_complexity = HammingWeightPhasingViaPhaseGradient(n, theta, eps).t_complexity()
    naive_hwp_t_complexity = HammingWeightPhasing(n, theta, eps).t_complexity()

    total_t = hwp_t_complexity.t_incl_rotations(eps=eps / n.bit_length())
    naive_total_t = naive_hwp_t_complexity.t_incl_rotations(eps=eps / n.bit_length())

    assert total_t < naive_total_t
