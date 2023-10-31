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

from qualtran.surface_code.algorithm_summary import AlgorithmSummary


def test_mul():
    assert AlgorithmSummary(t_gates=9) == 3 * AlgorithmSummary(t_gates=3)

    with pytest.raises(TypeError):
        _ = complex(1, 0) * AlgorithmSummary(rotation_gates=1)


def test_addition():
    with pytest.raises(TypeError):
        _ = AlgorithmSummary() + 5

    a = AlgorithmSummary(
        algorithm_qubits=7,
        measurements=8,
        t_gates=8,
        toffoli_gates=9,
        rotation_gates=8,
        rotation_circuit_depth=3,
    )
    b = AlgorithmSummary(
        algorithm_qubits=4,
        measurements=1,
        t_gates=1,
        toffoli_gates=4,
        rotation_gates=2,
        rotation_circuit_depth=1,
    )
    assert a + b == AlgorithmSummary(
        algorithm_qubits=11,
        measurements=9,
        t_gates=9,
        toffoli_gates=13,
        rotation_gates=10,
        rotation_circuit_depth=4,
    )


def test_subtraction():
    with pytest.raises(TypeError):
        _ = AlgorithmSummary() - 5

    a = AlgorithmSummary(
        algorithm_qubits=7,
        measurements=8,
        t_gates=8,
        toffoli_gates=9,
        rotation_gates=8,
        rotation_circuit_depth=3,
    )
    b = AlgorithmSummary(
        algorithm_qubits=4,
        measurements=1,
        t_gates=1,
        toffoli_gates=4,
        rotation_gates=2,
        rotation_circuit_depth=1,
    )
    assert a - b == AlgorithmSummary(
        algorithm_qubits=3,
        measurements=7,
        t_gates=7,
        toffoli_gates=5,
        rotation_gates=6,
        rotation_circuit_depth=2,
    )
