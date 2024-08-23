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

from qualtran.bloqs.basic_gates import ZPowGate
from qualtran.bloqs.for_testing.qubitization_walk_test import get_uniform_pauli_qubitized_walk
from qualtran.bloqs.phase_estimation.lp_resource_state import LPResourceState
from qualtran.bloqs.phase_estimation.qpe_window_state import RectangularWindowState
from qualtran.bloqs.phase_estimation.text_book_qpe import TextbookQPE
from qualtran.cirq_interop.testing import GateHelper


def simulate_theta_estimate(circuit, measurement_register) -> float:
    sim = cirq.Simulator()
    qubit_order = cirq.QubitOrder.explicit(measurement_register, fallback=cirq.QubitOrder.DEFAULT)
    final_state = sim.simulate(circuit, qubit_order=qubit_order).final_state_vector
    m_bits = len(measurement_register)
    samples = cirq.sample_state_vector(final_state, indices=[*range(m_bits)], repetitions=1000)
    counts = np.bincount(samples.dot(1 << np.arange(samples.shape[-1] - 1, -1, -1)))
    assert len(counts) <= 2**m_bits
    return np.argmax(counts) / 2**m_bits


@pytest.mark.parametrize('theta', [0.234, 0.78, 0.54])
def test_textbook_phase_estimation_zpow_theta(theta):
    precision, error_bound = 3, 0.1
    gh = GateHelper(TextbookQPE(ZPowGate(exponent=2 * theta), RectangularWindowState(precision)))
    circuit = cirq.Circuit(cirq.X(*gh.quregs['q']), cirq.decompose_once(gh.operation))
    precision_register = gh.quregs['qpe_reg']
    assert abs(simulate_theta_estimate(circuit, precision_register) - theta) < error_bound


@pytest.mark.slow
@pytest.mark.parametrize('num_terms', [2, 3, 4])
@pytest.mark.parametrize('use_resource_state', [True, False])
def test_textbook_phase_estimation_qubitized_walk(num_terms: int, use_resource_state: bool):
    precision, eps = 5, 0.05
    ham, walk = get_uniform_pauli_qubitized_walk(num_terms)

    ham_coeff = [abs(ps.coefficient.real) for ps in ham]
    qubitization_lambda = np.sum(ham_coeff)
    g = GateHelper(walk)
    L_state = np.zeros(2 ** len(g.quregs['selection']))
    L_state[: len(ham_coeff)] = np.sqrt(ham_coeff / qubitization_lambda)

    eigen_values, eigen_vectors = np.linalg.eigh(ham.matrix())

    state_prep = (
        LPResourceState(precision) if use_resource_state else RectangularWindowState(precision)
    )
    gh = GateHelper(TextbookQPE(walk, ctrl_state_prep=state_prep))
    # 1. Construct QPE bloq
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
        phase = theta * 2 * np.pi
        np.testing.assert_allclose(eig_val / qubitization_lambda, np.cos(phase), atol=eps)
