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
import cirq
import numpy as np
import pytest

from qualtran.bloqs.basic_gates import Hadamard, OnEach
from qualtran.bloqs.for_testing.qubitization_walk_test import get_uniform_pauli_qubitized_walk
from qualtran.bloqs.phase_estimation.kitaev_qpe_text_book_test import simulate_theta_estimate
from qualtran.bloqs.phase_estimation.lp_resource_state import LPResourceState
from qualtran.bloqs.phase_estimation.qubitization_qpe import QubitizationQPE
from qualtran.cirq_interop.testing import GateHelper


@pytest.mark.parametrize('num_terms', [2, 3, 4])
@pytest.mark.parametrize('use_resource_state', [True, False])
def test_kitaev_phase_estimation_qubitized_walk(num_terms: int, use_resource_state: bool):
    precision, eps = 5, 0.05
    ham, walk = get_uniform_pauli_qubitized_walk(num_terms)

    ham_coeff = [abs(ps.coefficient.real) for ps in ham]
    qubitization_lambda = np.sum(ham_coeff)
    g = GateHelper(walk)
    L_state = np.zeros(2 ** len(g.quregs['selection']))
    L_state[: len(ham_coeff)] = np.sqrt(ham_coeff / qubitization_lambda)

    eigen_values, eigen_vectors = np.linalg.eigh(ham.matrix())

    # 1. Construct QPE bloq

    state_prep = LPResourceState(precision) if use_resource_state else OnEach(precision, Hadamard())
    gh = GateHelper(QubitizationQPE(walk, precision, state_prep=state_prep))
    qpe_reg, selection, target = (gh.quregs['qpe_reg'], gh.quregs['selection'], gh.quregs['target'])
    for eig_idx, eig_val in enumerate(eigen_values):
        # Apply QPE to determine eigenvalue for walk operator W on initial state |L>|k>
        # 2. State preparation for initial eigenstate.
        L_K = np.kron(L_state, eigen_vectors[:, eig_idx].flatten())
        L_K /= abs(np.linalg.norm(L_K))
        prep_L_K = cirq.Circuit(cirq.StatePreparationChannel(L_K).on(*selection, *target))

        # 3. QPE circuit with state prep
        qpe_with_init = prep_L_K + gh.circuit
        assert len(qpe_with_init.all_qubits()) < 23

        # 4. Estimate theta
        theta = simulate_theta_estimate(qpe_with_init, qpe_reg)
        assert 0 <= theta <= 1

        # 5. Verify that the estimated phase is correct.
        phase = theta * np.pi
        is_close = [
            np.allclose(np.abs(eig_val / qubitization_lambda), np.abs(np.cos(phase)), atol=eps),
            np.allclose(np.abs(eig_val / qubitization_lambda), np.abs(np.sin(phase)), atol=eps),
        ]
        assert np.any(is_close)
