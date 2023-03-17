import cirq
import numpy as np
import pytest

import cirq_qubitization
from cirq_qubitization.generic_select_test import get_1d_ising_lcu_coeffs


@pytest.mark.parametrize("num_sites, epsilon", [[2, 1.0e-2], [3, 1.0e-2], [4, 1.0e-2], [5, 1.0e-2]])
def test_generic_subprepare(num_sites, epsilon):
    lcu_coefficients = get_1d_ising_lcu_coeffs(num_sites)
    subprepare_gate = cirq_qubitization.GenericSubPrepare(
        lcu_probabilities=lcu_coefficients, probability_epsilon=epsilon
    )
    q = cirq.LineQubit.range(cirq.num_qubits(subprepare_gate))
    selection = q[: subprepare_gate.selection_bitsize]
    temp = q[
        subprepare_gate.selection_bitsize : subprepare_gate.selection_bitsize
        + subprepare_gate.temp_bitsize
    ]
    op = subprepare_gate.on_registers(selection=selection, temp=temp)
    circuit = cirq.Circuit(cirq.I.on_each(*q), cirq.decompose(op))
    all_qubits = q + sorted(circuit.all_qubits() - frozenset(q))
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=all_qubits)
    state_vector = result.final_state_vector
    # State vector is of the form |l>|temp_{l}>. We trace out the |temp_{l}> part to
    # get the coefficients corresponding to |l>.
    L, logL = len(lcu_coefficients), subprepare_gate.selection_bitsize
    state_vector = state_vector.reshape(2**logL, len(state_vector) // 2**logL)
    num_non_zero = (state_vector > 1e-6).sum(axis=1)
    prepared_state = state_vector.sum(axis=1)
    assert all(num_non_zero[:L] > 0) and all(num_non_zero[L:] == 0)
    assert all(prepared_state[:L] > 1e-6) and all(prepared_state[L:] <= 1e-6)
    prepared_state = prepared_state[:L] / np.sqrt(num_non_zero[:L])
    # Assert that the absolute square of prepared state (probabilities instead of amplitudes) is
    # same as `lcu_coefficients` upto `epsilon`.
    np.testing.assert_allclose(lcu_coefficients, abs(prepared_state) ** 2, atol=epsilon * 10)
