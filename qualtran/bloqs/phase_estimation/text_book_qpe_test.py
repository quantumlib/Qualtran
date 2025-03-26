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

import qualtran.testing as qlt_testing
from qualtran import Signature
from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran.bloqs.basic_gates import ZPowGate
from qualtran.bloqs.chemistry.hubbard_model.qubitization import get_walk_operator_for_hubbard_model
from qualtran.bloqs.for_testing.qubitization_walk_test import get_uniform_pauli_qubitized_walk
from qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp import (
    HamiltonianSimulationByGQSP,
)
from qualtran.bloqs.phase_estimation import LPResourceState, RectangularWindowState, TextbookQPE
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


@pytest.mark.parametrize(
    'num_terms', [pytest.param(n, marks=() if n <= 2 else pytest.mark.slow) for n in [2, 3, 4]]
)
@pytest.mark.parametrize('use_resource_state', [True, False])
def test_textbook_phase_estimation_qubitized_walk(num_terms: int, use_resource_state: bool):
    precision, eps = 5, 0.05
    ham, walk = get_uniform_pauli_qubitized_walk(num_terms)

    ham_coeff = [abs(ps.coefficient.real) for ps in ham]
    qubitization_lambda = np.sum(ham_coeff)
    n_select_bits = Signature(walk.selection_registers).n_qubits()
    L_state = np.zeros(2**n_select_bits)
    L_state[: len(ham_coeff)] = np.sqrt(ham_coeff / qubitization_lambda)

    eigen_values, eigen_vectors = np.linalg.eigh(ham.matrix())

    state_prep = (
        LPResourceState(precision) if use_resource_state else RectangularWindowState(precision)
    )

    # 1. Construct QPE bloq
    qpe_bloq = TextbookQPE(walk, ctrl_state_prep=state_prep)

    # TODO cirq simulation seems to fail for controlled `QubitizationWalkOperator`.
    #      the following code decomposes a few levels till it gets only simulable bloqs.
    #      https://github.com/quantumlib/Qualtran/issues/1495
    def should_decompose(binst):
        from qualtran import Adjoint, Controlled
        from qualtran.bloqs.basic_gates import Power
        from qualtran.bloqs.qubitization import QubitizationWalkOperator

        bloqs_to_decompose = (TextbookQPE, QubitizationWalkOperator, Power)

        if binst.bloq_is(bloqs_to_decompose):
            return True

        if binst.bloq_is(Controlled) or binst.bloq_is(Adjoint):
            return isinstance(binst.bloq.subbloq, bloqs_to_decompose)

        return False

    cbloq = qpe_bloq.as_composite_bloq().flatten(pred=should_decompose)
    quregs = get_named_qubits(cbloq.signature.lefts())
    qpe_circuit, quregs = cbloq.to_cirq_circuit_and_quregs(None, **quregs)
    for eig_idx, eig_val in enumerate(eigen_values):
        # Apply QPE to determine eigenvalue for walk operator W on initial state |L>|k>
        # 2. State preparation for initial eigenstate.
        L_K = np.kron(L_state, eigen_vectors[:, eig_idx].flatten())
        L_K /= abs(np.linalg.norm(L_K))
        prep_L_K = cirq.Circuit(
            cirq.StatePreparationChannel(L_K).on(*quregs['selection'], *quregs['target'])
        )

        # 3. QPE circuit with state prep
        qpe_with_init = prep_L_K + qpe_circuit
        assert len(qpe_with_init.all_qubits()) < 23

        # 4. Estimate theta
        theta = simulate_theta_estimate(qpe_with_init, quregs['qpe_reg'])
        assert 0 <= theta <= 1

        # 5. Verify that the estimated phase is correct.
        phase = theta * 2 * np.pi
        np.testing.assert_allclose(eig_val / qubitization_lambda, np.cos(phase), atol=eps)


def test_qpe_of_gqsp():
    # This triggered a bug in the cirq interop.
    # https://github.com/quantumlib/Qualtran/issues/1570

    walk_op = get_walk_operator_for_hubbard_model(2, 2, 1, 1)
    hubbard_time_evolution_by_gqsp = HamiltonianSimulationByGQSP(walk_op, t=5, precision=1e-7)
    textbook_qpe_w_gqsp = TextbookQPE(hubbard_time_evolution_by_gqsp, RectangularWindowState(3))
    qlt_testing.assert_valid_bloq_decomposition(textbook_qpe_w_gqsp)
