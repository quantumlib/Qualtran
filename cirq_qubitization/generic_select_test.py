from typing import List
import numpy as np
import cirq

from cirq_qubitization.generic_select import Apply_ABSTRACTSELECT


class OneDimensionalIsingModel:

    def __init__(self, num_sites, j_zz_interaction=-1, gamma_x_interaction=-1) -> None:
        """
        H = -J\sum_{k=0}^{L-1}sigma_{k}^{Z}sigma_{(k+1)%L}^{Z} + -Gamma \sum_{k=0}^{L-1}sigma_{k}^{X}
        """
        self.num_sites = num_sites
        self.j = j_zz_interaction
        self.gamma = gamma_x_interaction

        self.qop_hamiltonian = None
        self.fermion_ham = None

    def get_cirq_operator(self, qubits: List[cirq.Qid]) -> cirq.PauliSum:
        """
        Construct the Hamiltonian as a PauliSum object

        :param qubits: list of qubits         
        :return: cirq.PauliSum representing the Hamiltonian
        """
        n_qubits = len(qubits)
        cirq_pauli_terms = []
        for k in range(self.num_sites):
            cirq_pauli_terms.append(cirq.PauliString(
                {qubits[k]: cirq.Z, qubits[(k + 1) % self.num_sites]: cirq.Z}, coefficient=self.j))
            cirq_pauli_terms.append(cirq.PauliString(
                {qubits[k]: cirq.X}, coefficient=self.gamma))
        return cirq.PauliSum().from_pauli_strings(cirq_pauli_terms)


def test_ising_zero_bitflip_select():
    sim = cirq.Simulator(dtype=np.complex128)
    num_sites = 4
    target_register_size = num_sites
    num_select_unitaries = 2 * num_sites
    # PBC Ising in 1-D has num_sites ZZ operations and num_sites X operations.
    # Thus 2 * num_sites Pauli ops
    selection_register_size = int(np.ceil(np.log(num_select_unitaries)))
    control_register_size = 1
    all_qubits = cirq.LineQubit.range(
        2 * selection_register_size + target_register_size + 1)
    control, selection, ancilla, target = (
        all_qubits[0],
        all_qubits[1: 2 * selection_register_size: 2],
        all_qubits[2: 2 * selection_register_size + 1: 2],
        all_qubits[2 * selection_register_size + 1:],
    )

    # Get paulistring terms
    # right now we only handle positive interaction term values
    ising_inst = OneDimensionalIsingModel(num_sites, 1, 1)
    paulistring_hamiltonian = ising_inst.get_cirq_operator(target)
    individual_hamiltonian_paulistrings = [
        tt for tt in paulistring_hamiltonian]
    assert all(
        [type(xx) == cirq.PauliString for xx in individual_hamiltonian_paulistrings])
    qubitization_lambda = sum(
        xx.coefficient.real for xx in individual_hamiltonian_paulistrings)

    # built select with unary iteration gate
    op = Apply_ABSTRACTSELECT(selection_register_length=selection_register_size,
                              target_register_length=target_register_size,
                              select_unitaries=individual_hamiltonian_paulistrings).on(control, *selection, *ancilla, *target)
    circuit = cirq.Circuit(cirq.decompose_once(op))

    # now we need to have a superposition w.r.t all operators to act on target.
    # Normally this would be generated by a PREPARE circuit but we will
    # build it directly here.
    for selection_integer in range(num_select_unitaries):
        svals = [int(x) for x in format(
            selection_integer, f"0{selection_register_size}b")]
        # turn on control bit to activate circuit
        qubit_vals = {x: int(x == control) for x in all_qubits}
        # Initialize selection bits appropriately
        qubit_vals.update({s: sval for s, sval in zip(selection, svals)})

        initial_state = [qubit_vals[x] for x in all_qubits]
        result = sim.simulate(circuit, initial_state=initial_state)
        for qid_key, pauli_val in individual_hamiltonian_paulistrings[selection_integer]._qubit_pauli_map.items():
            if pauli_val == cirq.X:
                # Hamiltonian already defined on correct qubits so just take qid
                initial_state[qid_key._x] = 1
        expected_output = "".join(str(x) for x in initial_state)
        assert result.dirac_notation()[1:-1] == expected_output


def test_ising_one_bitflip_select():
    sim = cirq.Simulator(dtype=np.complex128)
    num_sites = 4
    target_register_size = num_sites
    num_select_unitaries = 2 * num_sites
    # PBC Ising in 1-D has num_sites ZZ operations and num_sites X operations.
    # Thus 2 * num_sites Pauli ops
    selection_register_size = int(np.ceil(np.log(num_select_unitaries)))
    control_register_size = 1
    all_qubits = cirq.LineQubit.range(
        2 * selection_register_size + target_register_size + 1)
    control, selection, ancilla, target = (
        all_qubits[0],
        all_qubits[1: 2 * selection_register_size: 2],
        all_qubits[2: 2 * selection_register_size + 1: 2],
        all_qubits[2 * selection_register_size + 1:],
    )

    # Get paulistring terms
    # right now we only handle positive interaction term values
    ising_inst = OneDimensionalIsingModel(num_sites, 1, 1)
    paulistring_hamiltonian = ising_inst.get_cirq_operator(target)
    individual_hamiltonian_paulistrings = [
        tt for tt in paulistring_hamiltonian]
    assert all(
        [type(xx) == cirq.PauliString for xx in individual_hamiltonian_paulistrings])
    qubitization_lambda = sum(
        xx.coefficient.real for xx in individual_hamiltonian_paulistrings)

    # built select with unary iteration gate
    op = Apply_ABSTRACTSELECT(selection_register_length=selection_register_size,
                              target_register_length=target_register_size,
                              select_unitaries=individual_hamiltonian_paulistrings).on(control, *selection, *ancilla, *target)
    circuit = cirq.Circuit(cirq.decompose_once(op))

    # now we need to have a superposition w.r.t all operators to act on target.
    # Normally this would be generated by a PREPARE circuit but we will
    # build it directly here.
    for selection_integer in range(num_select_unitaries):
        svals = [int(x) for x in format(
            selection_integer, f"0{selection_register_size}b")]
        # turn on control bit to activate circuit
        qubit_vals = {x: int(x == control) for x in all_qubits}
        # Initialize selection bits appropriately
        qubit_vals.update({s: sval for s, sval in zip(selection, svals)})
        # flip target register to all 1
        qubit_vals.update({t: 1 for t in target})
        initial_state = [qubit_vals[x] for x in all_qubits]
        result = sim.simulate(circuit, initial_state=initial_state)
        for qid_key, pauli_val in individual_hamiltonian_paulistrings[selection_integer]._qubit_pauli_map.items():
            if pauli_val == cirq.X:
                # Hamiltonian already defined on correct qubits so just take qid
                initial_state[qid_key._x] = 0
        expected_output = "".join(str(x) for x in initial_state)
        assert result.dirac_notation()[1:-1] == expected_outputgg