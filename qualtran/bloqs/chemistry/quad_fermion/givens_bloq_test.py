import cirq
import numpy as np
import openfermion as of
from openfermion.circuits.gates import Ryxxy
from scipy.linalg import expm

from qualtran.bloqs.basic_gates import CNOT, Hadamard, SGate, XGate
from qualtran.bloqs.chemistry.quad_fermion.givens_bloq import (
    ComplexGivensRotationByPhaseGradient,
    RealGivensRotationByPhaseGradient,
)
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad


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
        i, j = 0, 1
        theta_fop = theta * (
            of.FermionOperator(((i, 1), (j, 0))) - of.FermionOperator(((j, 1), (i, 0)))
        )
        fUtheta = expm(of.get_sparse_operator(of.jordan_wigner(theta_fop), n_qubits=2).todense())
        assert np.allclose(fUtheta, ryxxy)
        circuit = circuit_construction(theta)
        test_unitary = cirq.unitary(circuit)
        assert np.isclose(4, abs(np.trace(test_unitary.conj().T @ fUtheta)))


def test_count_t_cliffords():
    add_into_phasegrad_gate = AddIntoPhaseGrad(
        x_bitsize=4, phase_bitsize=4, right_shift=0, sign=1, controlled=1
    )
    res = add_into_phasegrad_gate._t_complexity_()
    assert res.t == 16

    gate = RealGivensRotationByPhaseGradient(phasegrad_bitsize=4)
    gate_counts = gate.bloq_counts()

    assert gate_counts[CNOT()] == 5
    assert gate_counts[Hadamard()] == 2
    assert gate_counts[SGate(is_adjoint=False)] == 2
    assert gate_counts[SGate(is_adjoint=True)] == 1
    assert gate_counts[XGate()] == 2
    assert gate_counts[add_into_phasegrad_gate] == 2

    costs = gate.t_complexity()
    assert costs.t == 32
    assert costs.clifford == 12


def test_complex_givens_costs():
    gate = ComplexGivensRotationByPhaseGradient(phasegrad_bitsize=4)
    costs = gate.t_complexity()
    assert costs.t == 48
    assert costs.clifford == 12
