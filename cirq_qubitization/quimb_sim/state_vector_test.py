import random

import cirq
import numpy as np
import quimb.tensor as qtn

from cirq_qubitization.quimb_sim import circuit_to_tensors


def test_tensor_state_vector():
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.testing.random_circuit(qubits=qubits, n_moments=100, op_density=0.8)
    psi1 = cirq.final_state_vector(circuit, dtype=np.complex128)

    tensors, qubit_frontier, _ = circuit_to_tensors(
        circuit=circuit, initial_state={q: 0 for q in qubits}, final_state={}
    )
    tn = qtn.TensorNetwork(tensors)
    f_inds = tuple(f'i{qubit_frontier[q]}_{q}' for q in qubits)
    psi2 = tn.contract(inplace=True).to_dense(f_inds)

    np.testing.assert_allclose(psi1, psi2, atol=1e-8)


def test_initial_final():
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.testing.random_circuit(qubits=qubits, n_moments=100, op_density=0.8)
    circuit = circuit + cirq.inverse(circuit)

    init_final = {q: random.randint(0, 1) for q in qubits}
    tensors, qubit_frontier, _ = circuit_to_tensors(
        circuit=circuit, initial_state=init_final, final_state=init_final
    )
    tn = qtn.TensorNetwork(tensors)
    amp = tn.contract()

    np.testing.assert_allclose(amp, 1.0, atol=1e-8)
