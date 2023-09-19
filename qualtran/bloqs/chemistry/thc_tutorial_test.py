# Copyright 2023 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cirq
import cirq_ft
import numpy as np
import pytest

from qualtran.bloqs.chemistry.thc_tutorial import (
    ContiguousRegister,
    PrepareMatrix,
    PrepareUpperTriangular,
    SignedStatePreparationAliasSampling,
    SignedStatePreparationAliasSamplingLowerCost,
    UniformPrepareUpperTriangular,
)


@pytest.mark.parametrize("num_states, epsilon", [[2, 3e-3], [3, 3.0e-3], [4, 5.0e-3], [7, 8.0e-3]])
def test_signed_state_preparation(num_states, epsilon):
    np.random.seed(11)
    lcu_coefficients = np.random.randint(1, 10, num_states)
    signs = np.random.randint(0, 2, num_states)
    probs = lcu_coefficients / np.sum(lcu_coefficients)
    gate = SignedStatePreparationAliasSampling.from_lcu_probs(
        lcu_probabilities=(-1) ** signs * probs, probability_epsilon=epsilon
    )
    g = cirq_ft.testing.GateHelper(gate)
    qubit_order = g.operation.qubits
    zs = cirq.Circuit([cirq.Moment(cirq.Z(g.quregs['theta'][0]))])
    # Add a layer of Zs to pull out the sign
    sp_with_zs = cirq.Circuit(cirq.decompose_once(g.operation)) + zs
    # assertion to ensure that simulating the `decomposed_circuit` doesn't run out of memory.
    assert len(g.circuit.all_qubits()) < 22
    result = cirq.Simulator(dtype=np.complex128).simulate(sp_with_zs, qubit_order=qubit_order)
    state_vector = result.final_state_vector
    # State vector is of the form |l>|temp_{l}>. We trace out the |temp_{l}> part to
    # get the coefficients corresponding to |l>.
    L, logL = len(lcu_coefficients), len(g.quregs['selection'])
    state_vector = state_vector.reshape(2**logL, len(state_vector) // 2**logL)
    num_non_zero = (abs(state_vector) > 1e-6).sum(axis=1)
    prepared_state = state_vector.sum(axis=1)
    assert all(num_non_zero[:L] > 0) and all(num_non_zero[L:] == 0)
    assert all(np.abs(prepared_state[:L]) > 1e-6) and all(np.abs(prepared_state[L:]) <= 1e-6)
    prepared_state = prepared_state[:L] / np.sqrt(num_non_zero[:L])
    # Assert that the absolute square of prepared state (probabilities instead
    # of amplitudes) is same as `lcu_coefficients` upto `epsilon`.
    np.testing.assert_allclose(probs, abs(prepared_state) ** 2, atol=epsilon)
    state_signs = np.real(np.sign(prepared_state))
    np.testing.assert_equal(state_signs, (-1) ** signs)


@pytest.mark.parametrize("num_states, epsilon", [[2, 3e-3], [3, 3.0e-3], [4, 5.0e-3], [7, 8.0e-3]])
def test_signed_state_preparation_lower_non_clifford(num_states, epsilon):
    np.random.seed(11)
    lcu_coefficients = np.random.randint(1, 10, num_states)
    signs = np.random.randint(0, 2, num_states)
    probs = lcu_coefficients / np.sum(lcu_coefficients)
    gate = SignedStatePreparationAliasSamplingLowerCost.from_lcu_probs(
        lcu_probabilities=(-1) ** signs * probs, probability_epsilon=epsilon
    )
    g = cirq_ft.testing.GateHelper(gate)
    qubit_order = g.operation.qubits
    zs = cirq.Circuit([cirq.Moment(cirq.Z(g.quregs['theta'][0]))])
    # Add a layer of Zs to pull out the sign
    sp_with_zs = cirq.Circuit(cirq.decompose_once(g.operation)) + zs
    # assertion to ensure that simulating the `decomposed_circuit` doesn't run out of memory.
    assert len(g.circuit.all_qubits()) < 22
    result = cirq.Simulator(dtype=np.complex128).simulate(sp_with_zs, qubit_order=qubit_order)
    state_vector = result.final_state_vector
    # State vector is of the form |l>|temp_{l}>. We trace out the |temp_{l}> part to
    # get the coefficients corresponding to |l>.
    L, logL = len(lcu_coefficients), len(g.quregs['selection'])
    state_vector = state_vector.reshape(2**logL, len(state_vector) // 2**logL)
    num_non_zero = (abs(state_vector) > 1e-6).sum(axis=1)
    prepared_state = state_vector.sum(axis=1)
    assert all(num_non_zero[:L] > 0) and all(num_non_zero[L:] == 0)
    assert all(np.abs(prepared_state[:L]) > 1e-6) and all(np.abs(prepared_state[L:]) <= 1e-6)
    prepared_state = prepared_state[:L] / np.sqrt(num_non_zero[:L])
    # Assert that the absolute square of prepared state (probabilities instead
    # of amplitudes) is same as `lcu_coefficients` upto `epsilon`.
    np.testing.assert_allclose(probs, abs(prepared_state) ** 2, atol=epsilon)
    state_signs = np.real(np.sign(prepared_state))
    np.testing.assert_equal(state_signs, (-1) ** signs)


@pytest.mark.parametrize("num_rows_cols", [2, 4, 8])
def test_uniform_state_prep_up_triang(num_rows_cols):
    gate = UniformPrepareUpperTriangular(num_rows_cols)
    g = cirq_ft.testing.GateHelper(gate)
    assert len(g.circuit.all_qubits()) < 22
    qubit_order = g.operation.qubits
    result = cirq.Simulator(dtype=np.complex128).simulate(
        cirq.Circuit(cirq.decompose_once(g.operation))
    )
    ntot = num_rows_cols**2
    nupt = num_rows_cols * (num_rows_cols + 1) // 2
    assert len(np.where(np.abs(result.final_state_vector) > 1e-8)[0]) == nupt


# @pytest.mark.parametrize("num_rows_cols", [2, 4, 8])
# def test_state_prep_up_triang(num_rows_cols):
#     np.random.seed(3748)
#     zeta = np.random.randint(1, 10, size=(num_rows_cols, num_rows_cols))
#     gate = PrepareUpperTriangular.build()
#     g = cirq_ft.testing.GateHelper(gate)
#     assert len(g.circuit.all_qubits()) < 22
#     qubit_order = g.operation.qubits
#     # print(cirq.Circuit(cirq.decompose_once(g.operation)))
#     result = cirq.Simulator(dtype=np.complex128).simulate(
#         cirq.Circuit(cirq.decompose_once(g.operation))
#     )
#     ntot = num_rows_cols**2
#     nupt = num_rows_cols * (num_rows_cols + 1) // 2
#     assert len(np.where(np.abs(result.final_state_vector) > 1e-8)[0]) == nupt


@pytest.mark.parametrize("num_rows_cols", [2, 4])
def test_state_prep_matrix(num_rows_cols):
    np.random.seed(3748)
    zeta = np.random.randint(1, 10, size=(num_rows_cols, num_rows_cols))
    gate = PrepareMatrix.build(mat=zeta, epsilon=0.005)
    g = cirq_ft.testing.GateHelper(gate)
    qubit_order = g.operation.qubits
    assert len(g.circuit.all_qubits()) < 22
    qubit_order = g.operation.qubits
    result = cirq.Simulator(dtype=np.complex128).simulate(g.circuit, qubit_order=qubit_order)
    state_vector = result.final_state_vector
    L, logL = len(zeta.ravel()), (int(np.prod(zeta.shape) - 1).bit_length())
    state_vector = state_vector.reshape(2**logL, len(state_vector) // 2**logL)
    num_non_zero = (abs(state_vector) > 1e-6).sum(axis=1)
    prepared_state = state_vector.sum(axis=1)
    assert all(num_non_zero[:L] > 0) and all(num_non_zero[L:] == 0)
    assert all(np.abs(prepared_state[:L]) > 1e-6) and all(np.abs(prepared_state[L:]) <= 1e-6)
    prepared_state = prepared_state[:L] / np.sqrt(num_non_zero[:L])
    probs = zeta / np.sum(zeta)
    np.testing.assert_allclose(probs.ravel(), abs(prepared_state) ** 2, atol=0.004)


def test_contiguous_register_gate():
    gate = ContiguousRegister(3, 6, 8)
    circuit = cirq.Circuit(gate.on(*cirq.LineQubit.range(2 * 3 + 6)))
    basis_map = {}
    for p in range(2**3):
        for q in range(2**3):
            inp = f'0b_{p:03b}_{q:03b}_{0:06b}'
            out = f'0b_{p:03b}_{q:03b}_{p*8 + q:06b}'
            basis_map[int(inp, 2)] = int(out, 2)

    cirq.testing.assert_equivalent_computational_basis_map(basis_map, circuit)
