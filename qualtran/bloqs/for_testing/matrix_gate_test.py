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

from .matrix_gate import MatrixGate


@pytest.mark.parametrize("bitsize", [1, 2, 3])
def test_create_and_unitary(bitsize: int):
    random_state = np.random.RandomState(42)
    for _ in range(5):
        gate = MatrixGate.random(bitsize, random_state=random_state)
        np.testing.assert_allclose(cirq.unitary(gate), gate.matrix)
        np.testing.assert_allclose(cirq.unitary(gate**-1), np.conj(gate.matrix).T)
