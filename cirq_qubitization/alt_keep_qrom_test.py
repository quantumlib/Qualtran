import numpy as np

from cirq_qubitization import construct_alt_keep_qrom
from cirq_qubitization.bit_tools import iter_bits
from cirq_qubitization.cirq_infra import testing as cq_testing
from cirq_qubitization.generic_select_test import get_1d_ising_lcu_coeffs


def test_alt_keep_qrom():
    num_sites = 4
    lcu_coeffs = get_1d_ising_lcu_coeffs(num_sites)
    epsilon = 1.0e-2  # precision value is kept low so we can simulate the output
    qrom = construct_alt_keep_qrom(lcu_coefficients=lcu_coeffs, probability_epsilon=epsilon)

    alternates, keep_numers = qrom.data
    mu = max([int(xx).bit_length() for xx in keep_numers])

    # for this test mu should be equal to 4
    assert mu == 4

    # now test for correct properties of out_distribution
    n = len(lcu_coeffs)
    keep_denom = 2**mu
    assert len(alternates) == n
    assert len(keep_numers) == n
    assert all(0 <= e < keep_denom for e in keep_numers)

    out_distribution = np.array(keep_numers) / (n * keep_denom)
    for i in range(n):
        switch_probability = 1 - keep_numers[i] / keep_denom
        out_distribution[alternates[i]] += 1 / n * switch_probability

    assert np.allclose(out_distribution, lcu_coeffs, atol=epsilon)

    # now prepare qubits and iterate through selection register to confirm
    # QROM data output
    g = cq_testing.GateHelper(qrom)

    for selection_integer in range(n):
        qubit_vals = {x: 0 for x in g.all_qubits}
        qubit_vals |= zip(
            g.quregs['selection'], iter_bits(selection_integer, g.r['selection'].bitsize)
        )
        initial_state = [qubit_vals[x] for x in g.all_qubits]
        for ti, d in enumerate(qrom.data):
            target = g.quregs[f"target{ti}"]
            qubit_vals |= zip(target, iter_bits(int(d[selection_integer]), len(target)))
        final_state = [qubit_vals[x] for x in g.all_qubits]

        cq_testing.assert_circuit_inp_out_cirqsim(
            g.decomposed_circuit, g.all_qubits, initial_state, final_state
        )
