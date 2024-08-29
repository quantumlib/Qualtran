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

from qualtran.bloqs.for_testing.qubitization_walk_test import get_uniform_pauli_qubitized_walk
from qualtran.bloqs.phase_estimation.lp_resource_state import LPResourceState
from qualtran.bloqs.phase_estimation.qpe_window_state import RectangularWindowState
from qualtran.bloqs.phase_estimation.qubitization_qpe import (
    _qubitization_qpe_chem_thc,
    _qubitization_qpe_hubbard_model_small,
    _qubitization_qpe_sparse_chem,
    QubitizationQPE,
)
from qualtran.bloqs.phase_estimation.text_book_qpe_test import simulate_theta_estimate
from qualtran.cirq_interop.testing import GateHelper
from qualtran.testing import execute_notebook


@pytest.mark.slow
def test_qubitization_qpe_bloq_autotester(bloq_autotester):
    bloq_autotester(_qubitization_qpe_hubbard_model_small)


@pytest.mark.slow
def test_qubitization_qpe_chem_thc_bloq_autotester(bloq_autotester):
    bloq_autotester(_qubitization_qpe_chem_thc)


@pytest.mark.slow
def test_qubitization_qpe_sparse_chem_bloq_autotester(bloq_autotester):
    bloq_autotester(_qubitization_qpe_sparse_chem)


@pytest.mark.slow
@pytest.mark.parametrize('num_terms', [2, 3, 4])
@pytest.mark.parametrize('use_resource_state', [True, False])
def test_qubitization_phase_estimation_of_walk(num_terms: int, use_resource_state: bool):
    precision, eps = 5, 0.05
    ham, walk = get_uniform_pauli_qubitized_walk(num_terms)

    ham_coeff = np.array([abs(ps.coefficient.real) for ps in ham])
    qubitization_lambda = np.sum(ham_coeff)
    g = GateHelper(walk)
    # matrix = cirq.unitary(walk)
    L_state = np.zeros(2 ** len(g.quregs['selection']))
    L_state[: len(ham_coeff)] = np.sqrt(ham_coeff / qubitization_lambda)

    eigen_values, eigen_vectors = np.linalg.eigh(ham.matrix())

    # 1. Construct QPE bloq

    state_prep = (
        LPResourceState(precision) if use_resource_state else RectangularWindowState(precision)
    )
    gh = GateHelper(QubitizationQPE(walk, ctrl_state_prep=state_prep))
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
        # Since we apply U^\dagger for 0-control; the phase difference is twice
        # and therefore phase is pi * theta instead of 2 * pi * theta
        phase = np.pi * theta

        # We can measure either `2 * arccos(Ek/λ)` or `2 * arccos(Ek/λ) + pi`;
        # therefore Ek/λ = cos(theta/2) OR cos((theta - pi) / 2) = sin(theta/2)
        # TODO: Fig-2 of https://arxiv.org/abs/1805.03662 says the value of the first bit can be used
        #       to differentiate between the two cases, but I couldn't reproduce the argument here.
        #       Hence, we don't have a deterministic check to figure out correct value of
        #       `eig_val / qubitization_lambda` by the estimated `phase`.
        is_close = [
            np.allclose(np.abs(eig_val / qubitization_lambda), np.abs(np.cos(phase)), atol=eps),
            np.allclose(np.abs(eig_val / qubitization_lambda), np.abs(np.sin(phase)), atol=eps),
        ]
        assert np.any(is_close)


@pytest.mark.notebook
def test_phase_estimation_of_qubitized_hubbard_model():
    execute_notebook('phase_estimation_of_quantum_walk')
