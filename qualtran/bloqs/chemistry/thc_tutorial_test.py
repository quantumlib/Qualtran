# Copyright 2023 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cirq
import cirq_ft
import numpy as np
import pytest

from qualtran.bloqs.chemistry.thc_tutorial import SignedStatePreparationAliasSampling


@pytest.mark.parametrize("num_states, epsilon", [[2, 3e-3], [3, 3.0e-3], [4, 5.0e-3], [7, 8.0e-3]])
def test_state_preparation_via_coherent_alias_sampling(num_states, epsilon):
    lcu_coefficients = np.random.randint(1, 10, num_states)
    signs = np.random.randint(0, 2, num_states)
    probs = lcu_coefficients / np.sum(lcu_coefficients)
    gate = SignedStatePreparationAliasSampling.from_lcu_probs(
        lcu_probabilities=(-1) ** signs * probs, probability_epsilon=epsilon
    )
    g = cirq_ft.testing.GateHelper(gate)
    qubit_order = g.operation.qubits
    # assertion to ensure that simulating the `decomposed_circuit` doesn't run out of memory.
    assert len(g.circuit.all_qubits()) < 22
    result = cirq.Simulator(dtype=np.complex128).simulate(g.circuit, qubit_order=qubit_order)
    state_vector = result.final_state_vector
    # State vector is of the form |l>|temp_{l}>. We trace out the |temp_{l}> part to
    # get the coefficients corresponding to |l>.
    L, logL = len(lcu_coefficients), len(g.quregs['selection'])
    state_vector = state_vector.reshape(2**logL, len(state_vector) // 2**logL)
    num_non_zero = (abs(state_vector) > 1e-6).sum(axis=1)
    prepared_state = state_vector.sum(axis=1)
    assert all(num_non_zero[:L] > 0) and all(num_non_zero[L:] == 0)
    assert all(np.abs(prepared_state[:L]) > 1e-6) and all(np.abs(prepared_state[L:]) <= 1e-6)
    prepared_state = prepared_state[:L] / np.sqrt(num_non_zero[:L])
    # Assert that the absolute square of prepared state (probabilities instead of amplitudes) is
    # same as `lcu_coefficients` upto `epsilon`.
    np.testing.assert_allclose(probs, abs(prepared_state) ** 2, atol=epsilon)
