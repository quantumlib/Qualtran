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
import cirq
import numpy as np
import pytest

from qualtran.bloqs.rotations.hamming_weight_phasing import HammingWeightPhasing
from qualtran.cirq_interop.testing import GateHelper
from qualtran.testing import assert_valid_bloq_decomposition


@pytest.mark.parametrize('n', [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize('theta', [1 / 10, 1 / 5, 1 / 7, np.pi / 2])
def test_hamming_weight_phasing(n: int, theta: float):
    gate = HammingWeightPhasing(n, theta)
    assert_valid_bloq_decomposition(gate)

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
