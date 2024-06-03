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
import numpy as np
import pytest

import qualtran.testing as qlt_testing
from qualtran.bloqs.phase_estimation.lp_resource_state import (
    _lp_resource_state_small,
    _lprs_interim_prep,
    LPResourceState,
    LPRSInterimPrep,
)
from qualtran.cirq_interop.t_complexity_protocol import t_complexity, TComplexity
from qualtran.cirq_interop.testing import GateHelper
from qualtran.resource_counting.generalizers import (
    generalize_rotation_angle,
    ignore_alloc_free,
    ignore_split_join,
)


def test_lprs_interim_auto(bloq_autotester):
    bloq_autotester(_lprs_interim_prep)


def test_lp_resource_state_auto(bloq_autotester):
    bloq_autotester(_lp_resource_state_small)


def get_interim_resource_state(m: int) -> np.ndarray:
    N = 2**m
    state_vector = np.zeros(2 * N, dtype=np.complex128)
    state_vector[:N] = np.cos(np.pi * (1 + np.arange(N)) / (1 + N))
    state_vector[N:] = 1j * np.sin(np.pi * (1 + np.arange(N)) / (1 + N))
    return np.sqrt(1 / N) * state_vector


def get_resource_state(m: int) -> np.ndarray:
    N = 2**m
    return np.sqrt(2 / (1 + N)) * np.sin(np.pi * (1 + np.arange(N)) / (1 + N))


@pytest.mark.parametrize('n', [*range(1, 14, 2)])
def test_intermediate_resource_state(n):
    bloq = LPRSInterimPrep(n)
    state = GateHelper(bloq).circuit.final_state_vector()
    np.testing.assert_allclose(state, get_interim_resource_state(n))


@pytest.mark.parametrize('n', [*range(1, 14, 2)])
def test_prepares_resource_state(n):
    bloq = LPResourceState(n)
    state = GateHelper(bloq).circuit.final_state_vector()
    np.testing.assert_allclose(state, get_resource_state(n))


@pytest.mark.parametrize('n', [*range(1, 14, 2)])
def test_t_complexity(n):
    bloq = LPResourceState(n)
    qlt_testing.assert_equivalent_bloq_counts(
        bloq, [ignore_split_join, ignore_alloc_free, generalize_rotation_angle]
    )
    lprs_interim_count = 3 * TComplexity(rotations=2 * n + 1, clifford=2 + 3 * n)
    multi_control_pauli_count = TComplexity(t=4 * n, clifford=17 * n + 5)
    misc_count = TComplexity(rotations=3, clifford=5)

    assert bloq.t_complexity() == (lprs_interim_count + multi_control_pauli_count + misc_count)


@pytest.mark.parametrize('bitsize', [*range(1, 14, 2)])
def test_interim_lp2s_interim_prep_t_complexity(bitsize: int):
    assert t_complexity(LPRSInterimPrep(bitsize)) == TComplexity(
        rotations=2 * bitsize + 1, clifford=2 + 3 * bitsize
    )
