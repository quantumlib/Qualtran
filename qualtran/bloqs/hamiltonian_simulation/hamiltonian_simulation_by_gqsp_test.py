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
import scipy
from numpy.typing import NDArray

from qualtran.bloqs.for_testing.random_gate import RandomGate
from qualtran.bloqs.for_testing.random_select_and_prepare import random_qubitization_walk_operator
from qualtran.bloqs.qsp.generalized_qsp_test import (
    assert_matrices_almost_equal,
    check_polynomial_pair_on_random_points_on_unit_circle,
    verify_generalized_qsp,
)
from qualtran.bloqs.qubitization_walk_operator import QubitizationWalkOperator

from .hamiltonian_simulation_by_gqsp import (
    _hubbard_time_evolution_by_gqsp,
    HamiltonianSimulationByGQSP,
)


def test_examples(bloq_autotester):
    bloq_autotester(_hubbard_time_evolution_by_gqsp)


@pytest.mark.slow
@pytest.mark.parametrize("bitsize", [1, 2])
@pytest.mark.parametrize("t", [2, 3, 5])
@pytest.mark.parametrize("alpha", [1, 2, 3])
@pytest.mark.parametrize("precision", [1e-7, 1e-10])
def test_generalized_qsp_with_exp_cos_approx_on_random_unitaries(
    bitsize: int, t: float, alpha: float, precision: float
):
    random_state = np.random.RandomState(42)

    for _ in range(5):
        U = RandomGate.create(bitsize, random_state=random_state)
        gqsp = HamiltonianSimulationByGQSP(None, t=t, alpha=alpha, precision=precision).gqsp
        P, Q = gqsp.P, gqsp.Q

        check_polynomial_pair_on_random_points_on_unit_circle(
            P, Q, random_state=random_state, rtol=2 * precision
        )
        verify_generalized_qsp(U, P, Q, negative_power=len(P) // 2)


def verify_hamiltonian_simulation_by_gqsp(
    W: QubitizationWalkOperator, H: NDArray[np.complex_], *, t: float, alpha: float = 1
):
    N = H.shape[0]

    W_e_iHt = HamiltonianSimulationByGQSP(W, t=t, alpha=alpha, precision=1e-7)
    result_unitary = cirq.unitary(W_e_iHt)

    expected_top_left = scipy.linalg.expm(-1j * H * t)
    actual_top_left = result_unitary[:N, :N]
    assert_matrices_almost_equal(expected_top_left, actual_top_left)


@pytest.mark.slow
@pytest.mark.parametrize("select_bitsize", [1])
@pytest.mark.parametrize("target_bitsize", [1])
def test_hamiltonian_simulation_by_gqsp(select_bitsize: int, target_bitsize: int):
    random_state = np.random.RandomState(42)

    for _ in range(5):
        W, H = random_qubitization_walk_operator(
            select_bitsize, target_bitsize, random_state=random_state
        )
        verify_hamiltonian_simulation_by_gqsp(W, H.matrix(), t=5, alpha=1)
