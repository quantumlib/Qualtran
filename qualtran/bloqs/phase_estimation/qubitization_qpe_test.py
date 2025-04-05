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
import time

import cirq
import numpy as np
import pytest

from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran.bloqs.for_testing.qubitization_walk_test import get_uniform_pauli_qubitized_walk
from qualtran.bloqs.phase_estimation.lp_resource_state import LPResourceState
from qualtran.bloqs.phase_estimation.qpe_window_state import RectangularWindowState
from qualtran.bloqs.phase_estimation.qubitization_qpe import (
    _qubitization_qpe_chem_thc,
    _qubitization_qpe_hubbard_model_large,
    _qubitization_qpe_hubbard_model_small,
    _qubitization_qpe_ising,
    _qubitization_qpe_sparse_chem,
    QubitizationQPE,
)
from qualtran.bloqs.phase_estimation.text_book_qpe_test import simulate_theta_estimate
from qualtran.cirq_interop.testing import GateHelper
from qualtran.serialization.bloq import bloqs_to_proto
from qualtran.testing import execute_notebook


def test_ising_example(bloq_autotester):
    bloq_autotester(_qubitization_qpe_ising)


def test_qubitization_qpe_hubbard_model_small_autotester(bloq_autotester):
    bloq_autotester(_qubitization_qpe_hubbard_model_small)


def test_serialization_speed():
    start = time.perf_counter()
    bloqs_to_proto(_qubitization_qpe_hubbard_model_small.make())
    end = time.perf_counter()
    # Should take substantially less time than this
    if (end - start) > 2.0:
        assert False, 'Serialization should only check one level; and should be quick.'


@pytest.mark.slow
def test_qubitization_qpe_hubbard_model_large_autotester(bloq_autotester):
    bloq_autotester(_qubitization_qpe_hubbard_model_large)


@pytest.mark.slow
def test_qubitization_qpe_chem_thc_bloq_autotester(bloq_autotester):
    bloq_autotester(_qubitization_qpe_chem_thc)


@pytest.mark.slow
def test_qubitization_qpe_sparse_chem_bloq_autotester(bloq_autotester):
    bloq_autotester(_qubitization_qpe_sparse_chem)


@pytest.mark.parametrize(
    'num_terms', [pytest.param(n, marks=() if n <= 2 else pytest.mark.slow) for n in [2, 3, 4]]
)
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
    qpe_bloq = QubitizationQPE(walk, state_prep)

    # TODO cirq simulation seems to fail for controlled `QubitizationWalkOperator`.
    #      the following code decomposes a few levels till it gets only simulable bloqs.
    #      https://github.com/quantumlib/Qualtran/issues/1495
    def should_decompose(binst):
        from qualtran import Adjoint, Controlled
        from qualtran.bloqs.basic_gates import Power
        from qualtran.bloqs.qubitization import QubitizationWalkOperator

        bloqs_to_decompose = (QubitizationQPE, QubitizationWalkOperator, Power)

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


@pytest.mark.notebook
def test_notebook():
    execute_notebook('qubitization_qpe')
