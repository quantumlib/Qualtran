#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import cirq
import numpy as np
import openfermion as of
import pytest
from openfermion.circuits.gates import Ryxxy
from scipy.linalg import expm

from qualtran.bloqs.basic_gates import CNOT, Hadamard, SGate, XGate
from qualtran.bloqs.chemistry.quad_fermion.givens_bloq import (
    ComplexGivensRotationByPhaseGradient,
    RealGivensRotationByPhaseGradient,
)
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
from qualtran.resource_counting.t_counts_from_sigma import t_counts_from_sigma


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


@pytest.mark.parametrize("x_bitsize", [4, 5, 6, 7])
def test_count_t_cliffords(x_bitsize: int):
    add_into_phasegrad_gate = AddIntoPhaseGrad(
        x_bitsize=x_bitsize, phase_bitsize=x_bitsize, right_shift=0, sign=1, controlled=1
    )
    _, sigma = add_into_phasegrad_gate.call_graph()
    add_t_counts = t_counts_from_sigma(sigma)
    assert add_t_counts == ((x_bitsize - 2) * 2) * 4

    gate = RealGivensRotationByPhaseGradient(phasegrad_bitsize=x_bitsize)
    gate_counts = gate.bloq_counts()
    assert gate_counts[CNOT()] == 5
    assert gate_counts[Hadamard()] == 2
    assert gate_counts[SGate(is_adjoint=False)] == 2
    assert gate_counts[SGate(is_adjoint=True)] == 1
    assert gate_counts[XGate()] == 2
    assert gate_counts[add_into_phasegrad_gate] == 2

    costs = gate.t_complexity()
    assert costs.t == 2 * add_t_counts
    assert costs.clifford == 12


@pytest.mark.parametrize("x_bitsize", [4, 5, 6, 7])
def test_complex_givens_costs(x_bitsize: int):
    gate = ComplexGivensRotationByPhaseGradient(phasegrad_bitsize=x_bitsize)
    add_into_phasegrad_t_cost = ((x_bitsize - 2) * 2) * 4
    costs = gate.t_complexity()
    assert costs.t == add_into_phasegrad_t_cost * 3
    assert costs.clifford == 12
