import itertools
import cirq
import numpy as np
from cirq_qubitization import construct_alt_keep_qrom
from cirq_qubitization.generic_select_test import OneDimensionalIsingModel


def test_alt_keep_qrom():
    sim = cirq.Simulator(dtype=np.complex128)
    num_sites = 4
    target = cirq.LineQubit.range(num_sites)  # This is just for getting Hamiltonian coefficients
    ising_inst = OneDimensionalIsingModel(num_sites, j_zz_interaction=np.pi / 3, gamma_x_interaction=np.pi / 7)
    pauli_sum_hamiltonian = ising_inst.get_pauli_sum(target)
    pauli_string_hamiltonian = [*pauli_sum_hamiltonian]
    dense_pauli_string_hamiltonian = [
        tt.dense(target) for tt in pauli_string_hamiltonian
    ]
    qubitization_lambda = sum(
        xx.coefficient.real for xx in dense_pauli_string_hamiltonian
    )
    lcu_coeffs = (
        np.array([xx.coefficient.real for xx in dense_pauli_string_hamiltonian]) 
        / qubitization_lambda
    )
    epsilon = 1.0E-2  # precision value is kept low so we can simulate the output
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
    all_qubits = cirq.LineQubit.range(qrom.num_qubits())
    control, selection, ancilla, flat_target = (
        all_qubits[0],
        all_qubits[1 : 2 * qrom.selection_register : 2],
        all_qubits[2 : 2 * qrom.selection_register + 1 : 2],
        all_qubits[2 * qrom.selection_register + 1 :],
    )
    target_lengths = [max(d).bit_length() for d in qrom._data]
    target = [
        flat_target[y - x : y]
        for x, y in zip(target_lengths, itertools.accumulate(target_lengths))
    ]
    circuit = cirq.Circuit(
        qrom.on(
            control_register=control,
            selection_register=selection,
            selection_ancilla=ancilla,
            target_register=target if len(target) > 1 else flat_target,
        )
    )

    sim = cirq.Simulator()
    for selection_integer in range(qrom.iteration_length):
        svals = [
            int(x) for x in format(selection_integer, f"0{qrom.selection_register}b")
        ]
        qubit_vals = {x: int(x == control) for x in all_qubits}
        qubit_vals.update({s: sval for s, sval in zip(selection, svals)})

        initial_state = [qubit_vals[x] for x in all_qubits]
        result = sim.simulate(circuit, initial_state=initial_state)

        start = 2 * qrom.selection_register + 1
        for d, d_bits in zip(qrom._data, target_lengths):
            end = start + d_bits
            initial_state[start:end] = [
                int(x) for x in format(d[selection_integer], f"0{end - start}b")
            ]
            start = end
        expected_output = "".join(str(x) for x in initial_state)
        assert result.dirac_notation()[1:-1] == expected_output