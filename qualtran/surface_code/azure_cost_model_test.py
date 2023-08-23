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
from attrs import frozen

from qualtran.surface_code import azure_cost_model
from qualtran.surface_code.algorithm_specs import AlgorithmSpecs
from qualtran.surface_code.physical_parameters import BeverlandEtAl as physical_parameters
from qualtran.surface_code.quantum_error_correction_scheme import GateBasedSurfaceCode
from qualtran.surface_code.rotation_cost_model import BeverlandEtAl as rotation_cost_model
from qualtran.surface_code.t_factory import TFactory


@frozen
class Test:
    alg: AlgorithmSpecs
    q_alg: float

    error_budget: float
    c_min: float

    time_steps: float
    code_distance: float

    t_states: float


_TESTS = [
    Test(
        alg=AlgorithmSpecs(
            algorithm_qubits=100,
            rotation_gates=30100,
            measurements=14 * 10**5,
            rotation_circuit_depth=501,
        ),
        q_alg=230,
        error_budget=1e-3,
        c_min=1.5e6,
        time_steps=1.5e5,
        code_distance=9,
        t_states=6e5,
    ),
    Test(
        alg=AlgorithmSpecs(
            algorithm_qubits=1318,
            rotation_gates=2.06e8,
            measurements=1.37e9,
            rotation_circuit_depth=2.05e8,
            toffoli_gates=1.35e11,
            t_gates=5.53e7,
        ),
        q_alg=2740,
        error_budget=1e-2,
        c_min=4.1e11,
        time_steps=4.1e11,
        code_distance=17,
        t_states=5.44e11,
    ),
    Test(
        alg=AlgorithmSpecs(
            algorithm_qubits=12581,
            rotation_gates=12,
            measurements=1.08e9,
            rotation_circuit_depth=12,
            toffoli_gates=3.73e10,
            t_gates=12,
        ),
        q_alg=25481,
        error_budget=1 / 3,
        c_min=1.23e11,
        time_steps=1.23e10,
        code_distance=13,
        t_states=1.49e11,
    ),
]


@pytest.mark.parametrize('test', _TESTS)
def test_logical_qubits(test: Test):
    assert azure_cost_model.logical_qubits(test.alg) == test.q_alg


@pytest.mark.parametrize('test', _TESTS)
def test_minimum_time_step(test: Test):
    got = azure_cost_model.minimum_time_steps(
        test.error_budget, test.alg, rotation_cost=rotation_cost_model
    )
    want = test.c_min
    rel = abs(got - want) / want
    assert rel < 0.1


@pytest.mark.parametrize('test', _TESTS)
def test_code_distance(test: Test):
    got = azure_cost_model.code_distance(
        test.error_budget,
        test.time_steps,
        test.alg,
        qec_scheme=GateBasedSurfaceCode(error_rate_scaler=0.03, error_rate_threshold=0.01),
        physical_parameters=physical_parameters,
    )
    want = test.code_distance
    assert got == want


def test_code_distance_list_input():
    alg = AlgorithmSpecs(
        algorithm_qubits=12581,
        rotation_gates=12,
        measurements=1.08e9,
        rotation_circuit_depth=12,
        toffoli_gates=3.73e10,
        t_gates=12,
    )
    error_budgets = np.array([1 / 3, 1 / 4])
    time_steps = 1.23e10
    got = azure_cost_model.code_distance(
        error_budgets,
        time_steps,
        alg,
        qec_scheme=GateBasedSurfaceCode(error_rate_scaler=0.03, error_rate_threshold=0.01),
        physical_parameters=physical_parameters,
    )
    want = [13, 15]
    np.testing.assert_equal(got, want)


@pytest.mark.parametrize('test', _TESTS)
def test_t_states(test: Test):
    got = azure_cost_model.t_states(test.error_budget, test.alg, rotation_cost=rotation_cost_model)
    want = test.t_states
    rel = abs(got - want) / want
    assert rel < 0.1, f'{got=} {want=} {rel=}'


@pytest.mark.parametrize(
    'num_t_states, runtime, factory, num_factories',
    [
        (
            2.4e6,
            6.78,
            TFactory(error_rate=5.6e-11, num_qubits=3240, duration=46.8e-6, t_states_rate=1),
            17,
        ),
        (
            5.44e11,
            30.5 * 24 * 3600,  # A month + half a day.
            TFactory(error_rate=2.13e-15, num_qubits=16000, duration=83.2e-6, t_states_rate=1),
            18,
        ),
        (
            1.49e10,
            17 * 3600 + 43 * 60,  # 17h43m
            TFactory(error_rate=5.51e-13, num_qubits=5760, duration=72.8e-6, t_states_rate=1),
            18,
        ),
    ],
)
def test_number_t_factories(num_t_states, runtime, factory, num_factories):
    got = azure_cost_model.number_t_factories(
        num_t_states=num_t_states, algorithm_runtime=runtime, t_factory=factory
    )
    np.testing.assert_allclose(got, num_factories)
