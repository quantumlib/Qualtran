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
import cirq
import numpy as np

from .su2_rotation import _hadamard, _su2_rotation_gate, _t_gate, SU2RotationGate


def test_cirq_decompose_SU2_to_single_qubit_pauli_gates():
    random_state = np.random.default_rng(42)

    for _ in range(20):
        theta, phi, lambd = random_state.random(size=3) * 2 * np.pi
        gate = SU2RotationGate(theta, phi, lambd)

        np.testing.assert_allclose(cirq.unitary(gate), gate.rotation_matrix)


def test_tensors():
    random_state = np.random.default_rng(42)

    for _ in range(20):
        theta, phi, lambd = random_state.random(size=3) * 2 * np.pi
        gate = SU2RotationGate(theta, phi, lambd)

        np.testing.assert_allclose(gate.tensor_contract(), gate.rotation_matrix)


def test_su2_rotation_gates(bloq_autotester):
    bloq_autotester(_su2_rotation_gate)
    bloq_autotester(_t_gate)
    bloq_autotester(_hadamard)
