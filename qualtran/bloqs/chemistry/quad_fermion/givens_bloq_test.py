import cirq
import numpy as np
import openfermion as of
from openfermion.circuits.gates import Ryxxy
from scipy.linalg import expm

def test_circuit_decomposition_givens():
    """
    confirm Figure 9 of [Quantum 4, 296 (2020)](https://quantum-journal.org/papers/q-2020-07-16-296/pdf/)
    corresponds to Givens rotation in OpenFermion
    """
    np.set_printoptions(linewidth=500)
    def circuit_construction(eta):
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit()
        circuit.append(cirq.X.on(qubits[0]))
        circuit.append(cirq.S.on(qubits[1]))

        circuit.append(cirq.CNOT(qubits[1], qubits[0]))
        circuit.append(cirq.H.on(qubits[1]))
        circuit.append(cirq.inverse(cirq.S.on(qubits[0])))

        circuit.append(cirq.CNOT(qubits[1], qubits[0]))
        circuit.append(cirq.rz(eta).on(qubits[0]))
        circuit.append(cirq.rz(eta).on(qubits[1]))
        circuit.append(cirq.CNOT(qubits[1], qubits[0]))

        circuit.append(cirq.X.on(qubits[0]))
        circuit.append(cirq.CNOT(qubits[0], qubits[1]))
        circuit.append(cirq.H.on(qubits[1]))
        circuit.append(cirq.CNOT(qubits[1], qubits[0]))
        circuit.append(cirq.inverse(cirq.S.on(qubits[0])))
        return circuit

    for _ in range(10):
        theta = 2 * np.pi / np.random.randn()
        ryxxy = cirq.unitary(Ryxxy(theta))
        print(theta, theta / np.pi)
        print(ryxxy.real)
        i, j = 0, 1
        theta_fop = theta * (of.FermionOperator(((i, 1), (j, 0))) - of.FermionOperator(((j, 1), (i, 0))))
        fUtheta = expm(of.get_sparse_operator(of.jordan_wigner(theta_fop), n_qubits=2).todense())
        print(fUtheta.real)
        assert np.allclose(fUtheta, ryxxy)
        circuit = circuit_construction(theta)
        test_unitary = cirq.unitary(circuit)
        print((1j * test_unitary).real)
        print(abs(np.trace(test_unitary.conj().T @ fUtheta)))