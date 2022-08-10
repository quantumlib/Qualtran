import pytest
import cirq
import numpy as np
import cirq_qubitization
from cirq_qubitization.generic_select_test import OneDimensionalIsingModel


def get_lcu_coefficients(num_sites):
    # PBC Ising in 1-D has `num_sites` ZZ operations and `num_sites` X operations.
    # Thus 2 * `num_sites` Pauli ops
    selection_register_size = int(np.ceil(np.log(2 * num_sites)))
    # Get paulistring terms
    # right now we only handle positive interaction term values
    target = cirq.LineQubit.range(num_sites)
    ising_inst = OneDimensionalIsingModel(num_sites, np.pi / 3, np.pi / 7)
    pauli_string_hamiltonian = [*ising_inst.get_pauli_sum(target)]
    dense_pauli_string_hamiltonian = [tt.dense(target) for tt in pauli_string_hamiltonian]
    qubitization_lambda = sum(xx.coefficient.real for xx in dense_pauli_string_hamiltonian)
    lcu_coeffs = (
        np.array([xx.coefficient.real for xx in dense_pauli_string_hamiltonian])
        / qubitization_lambda
    )
    return lcu_coeffs


@pytest.mark.parametrize("num_sites, epsilon", [[2, 1.0e-2], [3, 1.0e-2], [4, 1.0e-2], [5, 1.0e-2]])
def test_generic_subprepare(num_sites, epsilon):
    lcu_coefficients = get_lcu_coefficients(num_sites)
    subprepare_gate = cirq_qubitization.GenericSubPrepare(
        lcu_probabilities=lcu_coefficients, probability_epsilon=epsilon
    )
    q = cirq.LineQubit.range(cirq.num_qubits(subprepare_gate))
    selection = q[: subprepare_gate.selection_register]
    temp = q[
        subprepare_gate.selection_register : subprepare_gate.selection_register
        + subprepare_gate.temp_register
    ]
    ancilla = q[-subprepare_gate.ancilla_register :]
    result = cirq.Simulator(dtype=np.complex128).simulate(
        cirq.Circuit(
            subprepare_gate.on_registers(
                selection_register=selection, temp_register=temp, selection_ancilla=ancilla
            )
        )
    )
    state_vector = result.final_state_vector
    # State vector is of the form |l>|temp_{l}>. We trace out the |temp_{l}> part to
    # get the coefficients corresponding to |l>.
    L, logL = len(lcu_coefficients), subprepare_gate.selection_register
    state_vector = state_vector.reshape(2**logL, len(state_vector) // 2**logL)
    num_non_zero = (state_vector > 1e-6).sum(axis=1)
    prepared_state = state_vector.sum(axis=1)
    assert all(num_non_zero[:L] > 0) and all(num_non_zero[L:] == 0)
    assert all(prepared_state[:L] > 1e-6) and all(prepared_state[L:] <= 1e-6)
    prepared_state = prepared_state[:L] / np.sqrt(num_non_zero[:L])
    # Assert that the absolute square of prepared state (probabilities instead of amplitudes) is
    # same as `lcu_coefficients` upto `epsilon`.
    np.testing.assert_allclose(lcu_coefficients, abs(prepared_state) ** 2, atol=epsilon * 10)
