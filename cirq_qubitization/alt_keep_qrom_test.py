import cirq
import numpy as np
from cirq_qubitization import construct_alt_keep_qrom
from cirq_qubitization.generic_select_test import OneDimensionalIsingModel


def test_alt_keep_qrom():
    num_sites = 4
    target = cirq.LineQubit.range(num_sites)  # This is just for getting Hamiltonian coefficients
    ising_inst = OneDimensionalIsingModel(
        num_sites, j_zz_interaction=np.pi / 3, gamma_x_interaction=np.pi / 7
    )
    pauli_sum_hamiltonian = ising_inst.get_pauli_sum(target)
    pauli_string_hamiltonian = [*pauli_sum_hamiltonian]
    dense_pauli_string_hamiltonian = [tt.dense(target) for tt in pauli_string_hamiltonian]
    qubitization_lambda = sum(xx.coefficient.real for xx in dense_pauli_string_hamiltonian)
    lcu_coeffs = (
        np.array([xx.coefficient.real for xx in dense_pauli_string_hamiltonian])
        / qubitization_lambda
    )
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

    sim = cirq.Simulator()
    for selection_integer in range(qrom.iteration_length):
        svals = [int(x) for x in format(selection_integer, f"0{len(selection)}b")]
        qubit_vals = {x: 0 for x in all_qubits}
        qubit_vals.update({s: sval for s, sval in zip(selection, svals)})

        initial_state = [qubit_vals[x] for x in all_qubits]
        result = sim.simulate(circuit, initial_state=initial_state, qubit_order=all_qubits)

        for target, d in zip(targets, qrom._data):
            for q, b in zip(target, f"{d[selection_integer]:0{len(target)}b}"):
                qubit_vals[q] = int(b)
        final_state = [qubit_vals[x] for x in all_qubits]
        expected_output = "".join(str(x) for x in final_state)
        assert result.dirac_notation()[1:-1] == expected_output
