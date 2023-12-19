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

from qualtran.surface_code import azure_cost_model
from qualtran.surface_code.algorithm_summary import AlgorithmSummary
from qualtran.surface_code.quantum_error_correction_scheme_summary import (
    BeverlandSuperconductingQubits,
)
from qualtran.surface_code.rotation_cost_model import BeverlandEtAlRotationCost


@frozen
class Test:
    alg: AlgorithmSummary
    q_alg: float

    error_budget: float
    c_min: float

    time_steps: float
    code_distance: float

    t_states: float


_TESTS = [
    Test(
        alg=AlgorithmSummary(
            algorithm_qubits=100,
            rotation_gates=30100,
            measurements=1.4 * 10**6,
            rotation_circuit_depth=501,
        ),
        q_alg=230,
        error_budget=1e-3,
        c_min=1.5e6,
        time_steps=1.5e5,
        code_distance=9,
        t_states=602000,
    ),
    Test(
        alg=AlgorithmSummary(
            algorithm_qubits=1318,
            t_gates=5.53e7,
            rotation_circuit_depth=2.05e8,
            rotation_gates=2.06e8,
            toffoli_gates=1.35e11,
            measurements=1.37e9,
        ),
        q_alg=2740,
        error_budget=1e-2,
        c_min=4.1e11,
        time_steps=4.1e11,
        code_distance=17,
        t_states=5.44e11,
    ),
    Test(
        alg=AlgorithmSummary(
            algorithm_qubits=12581,
            t_gates=12,
            rotation_circuit_depth=12,
            rotation_gates=12,
            toffoli_gates=3.73e9,
            measurements=1.08e9,
        ),
        q_alg=25481,
        error_budget=1 / 3,
        c_min=1.23e10,
        time_steps=1.23e10,
        code_distance=13,
        t_states=1.49e10,
    ),
]


@pytest.mark.parametrize('test', _TESTS)
def test_logical_qubits(test: Test):
    assert azure_cost_model.logical_qubits(test.alg) == test.q_alg


@pytest.mark.parametrize('test', _TESTS)
def test_minimum_time_step(test: Test):
    got = azure_cost_model.minimum_time_steps(
        test.error_budget, test.alg, rotation_model=BeverlandEtAlRotationCost
    )
    assert got == pytest.approx(test.c_min, rel=0.1)


@pytest.mark.parametrize('test', _TESTS)
def test_code_distance(test: Test):
    got = azure_cost_model.code_distance(
        test.error_budget,
        test.time_steps,
        test.alg,
        qec=BeverlandSuperconductingQubits,
        physical_error_rate=1e-4,
    )
    assert got == test.code_distance


@pytest.mark.parametrize('test', _TESTS)
def test_t_states(test: Test):
    got = azure_cost_model.t_states(
        test.error_budget, test.alg, rotation_model=BeverlandEtAlRotationCost
    )
    assert got == pytest.approx(test.t_states, rel=0.1)
