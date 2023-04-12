import cirq
import numpy as np
import pytest

import cirq_qubitization as cq
import cirq_qubitization.cirq_infra.testing as cq_testing
from cirq_qubitization.generic_select_test import get_1d_ising_lcu_coeffs


def construct_gate_helper_and_qubit_order(data, eps):
    gate = cq.StatePreparationAliasSampling(lcu_probabilities=data, probability_epsilon=eps)
    with cq.cirq_infra.memory_management_context():
        g = cq_testing.GateHelper(gate)
        _ = g.decomposed_circuit
    ordered_input = sum(g.quregs.values(), start=[])
    qubit_order = cirq.QubitOrder.explicit(ordered_input, fallback=cirq.QubitOrder.DEFAULT)
    return g, qubit_order


@pytest.mark.parametrize("num_sites, epsilon", [[2, 1e-2], [3, 1.0e-2], [4, 2.0e-1], [7, 1.0e-1]])
def test_state_preparation_via_coherent_alias_sampling(num_sites, epsilon):
    lcu_coefficients = get_1d_ising_lcu_coeffs(num_sites)
    g, qubit_order = construct_gate_helper_and_qubit_order(lcu_coefficients, epsilon)
    result = cirq.Simulator(dtype=np.complex128).simulate(
        g.decomposed_circuit, qubit_order=qubit_order
    )
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
    # Assert that the absolute square of prepared state (probabilities instead of amplitudes) is
    # same as `lcu_coefficients` upto `epsilon`.
    np.testing.assert_allclose(lcu_coefficients, abs(prepared_state) ** 2, atol=epsilon)


def test_state_preparation_via_coherent_alias_sampling_diagram():
    data = np.asarray(range(1, 5)) / np.sum(range(1, 5))
    g, qubit_order = construct_gate_helper_and_qubit_order(data, 0.05)
    circuit = cirq.Circuit(cirq.decompose_once(g.operation))
    cirq.testing.assert_has_diagram(
        circuit,
        '''
selection0: ────────PrepareUniformSuperposition───In───────────────────×(y)───
                    │                             │                    │
selection1: ────────target────────────────────────In───────────────────×(y)───
                                                  │                    │
sigma_mu0: ─────────H─────────────────────────────┼────────In(y)───────┼──────
                                                  │        │           │
sigma_mu1: ─────────H─────────────────────────────┼────────In(y)───────┼──────
                                                  │        │           │
sigma_mu2: ─────────H─────────────────────────────┼────────In(y)───────┼──────
                                                  │        │           │
alt0: ────────────────────────────────────────────QROM_0───┼───────────×(x)───
                                                  │        │           │
alt1: ────────────────────────────────────────────QROM_0───┼───────────×(x)───
                                                  │        │           │
keep0: ───────────────────────────────────────────QROM_1───In(x)───────┼──────
                                                  │        │           │
keep1: ───────────────────────────────────────────QROM_1───In(x)───────┼──────
                                                  │        │           │
keep2: ───────────────────────────────────────────QROM_1───In(x)───────┼──────
                                                           │           │
less_than_equal: ──────────────────────────────────────────+(x <= y)───@──────
''',
        qubit_order=qubit_order,
    )
