import cirq
import numpy as np

from cirq_qubitization import construct_alt_keep_qrom
from cirq_qubitization.cirq_infra import testing as cq_testing
from cirq_qubitization.bit_tools import iter_bits
from cirq_qubitization.generic_select_test import get_1d_ising_lcu_coeffs


def test_alt_keep_qrom():
    num_sites = 4
    lcu_coeffs = get_1d_ising_lcu_coeffs(num_sites)
    epsilon = 1.0e-2  # precision value is kept low so we can simulate the output
    qrom = construct_alt_keep_qrom(lcu_coefficients=lcu_coeffs, probability_epsilon=epsilon)

    alternates, keep_numers = qrom._data
    mu = max([xx.bit_length() for xx in keep_numers])

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
    qubit_regs = qrom.registers.get_named_qubits()
    all_qubits = qrom.registers.merge_qubits(**qubit_regs)
    selection = qubit_regs["selection"]
    targets = [qubit_regs[f"target{i}"] for i in range(len(qrom._data))]
    circuit = cirq.Circuit(qrom.on_registers(**qubit_regs))

    for selection_integer in range(qrom.iteration_length):
        svals = list(iter_bits(selection_integer, len(selection)))
        qubit_vals = {x: 0 for x in all_qubits}
        qubit_vals.update({s: sval for s, sval in zip(selection, svals)})

        initial_state = [qubit_vals[x] for x in all_qubits]
        for target, d in zip(targets, qrom._data):
            for q, b in zip(target, iter_bits(d[selection_integer], len(target))):
                qubit_vals[q] = b
        final_state = [qubit_vals[x] for x in all_qubits]

        cq_testing.assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)
