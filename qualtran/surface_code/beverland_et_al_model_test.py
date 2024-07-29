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

import pytest
from attrs import frozen

import qualtran.testing as qlt_testing
from qualtran.resource_counting import GateCounts
from qualtran.surface_code import AlgorithmSummary, beverland_et_al_model, QECScheme
from qualtran.surface_code.rotation_cost_model import BeverlandEtAlRotationCost


@frozen
class Test:
    alg: AlgorithmSummary

    error_budget: float
    c_min: float

    time_steps: float
    code_distance: float

    t_states: float


_TESTS = [
    Test(
        alg=AlgorithmSummary(
            n_algo_qubits=100,
            n_logical_gates=GateCounts(rotation=30_100, measurement=int(1.4e6)),
            n_rotation_layers=501,
        ),
        error_budget=1e-3,
        c_min=1.5e6,
        time_steps=1.5e5,
        code_distance=9,
        t_states=602000,
    ),
    Test(
        alg=AlgorithmSummary(
            n_algo_qubits=1318,
            n_logical_gates=GateCounts(
                t=int(5.53e7), rotation=int(2.06e8), toffoli=int(1.35e11), measurement=int(1.37e9)
            ),
            n_rotation_layers=int(2.05e8),
        ),
        error_budget=1e-2,
        c_min=4.1e11,
        time_steps=4.1e11,
        code_distance=17,
        t_states=5.44e11,
    ),
    Test(
        alg=AlgorithmSummary(
            n_algo_qubits=12581,
            n_logical_gates=GateCounts(
                t=12, rotation=12, toffoli=int(3.73e9), measurement=int(1.08e9)
            ),
            n_rotation_layers=12,
        ),
        error_budget=1 / 3,
        c_min=1.23e10,
        time_steps=1.23e10,
        code_distance=13,
        t_states=1.49e10,
    ),
]


@pytest.mark.parametrize('test', _TESTS)
def test_minimum_time_step(test: Test):
    got = beverland_et_al_model.minimum_time_steps(
        error_budget=test.error_budget, alg=test.alg, rotation_model=BeverlandEtAlRotationCost
    )
    assert got == pytest.approx(test.c_min, rel=0.1)


@pytest.mark.parametrize('test', _TESTS)
def test_code_distance(test: Test):
    got = beverland_et_al_model.code_distance(
        error_budget=test.error_budget,
        time_steps=test.time_steps,
        alg=test.alg,
        qec_scheme=QECScheme.make_beverland_et_al(),
        physical_error=1e-4,
    )
    assert got == test.code_distance


@pytest.mark.parametrize('test', _TESTS)
def test_t_states(test: Test):
    got = beverland_et_al_model.t_states(
        error_budget=test.error_budget, alg=test.alg, rotation_model=BeverlandEtAlRotationCost
    )
    assert got == pytest.approx(test.t_states, rel=0.1)


def test_notebook():
    qlt_testing.execute_notebook('beverland_et_al_model')
