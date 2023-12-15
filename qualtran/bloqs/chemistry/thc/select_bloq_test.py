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
import pytest
import scipy.linalg

from qualtran.bloqs.chemistry.thc.select_bloq import _thc_rotations, _thc_sel, find_givens_angles


def test_thc_rotations(bloq_autotester):
    bloq_autotester(_thc_rotations)


def test_thc_select(bloq_autotester):
    bloq_autotester(_thc_sel)


@pytest.mark.parametrize("theta", 2 * np.pi * np.random.random(10))
def test_interleaved_cliffords(theta):
    a, b = cirq.LineQubit.range(2)
    XY = cirq.Circuit([cirq.X(a), cirq.Y(b)])
    UXY = cirq.unitary(XY)
    RXY_ref = scipy.linalg.expm(-1j * theta * UXY / 2)
    C0 = [cirq.H(a), cirq.S(b) ** -1, cirq.H(b), cirq.CNOT(a, b)]
    C1 = [cirq.S(a) ** -1, cirq.H(a), cirq.H(b), cirq.CNOT(a, b)]
    RXY = cirq.unitary(cirq.Circuit([C0, cirq.Rz(rads=theta)(b), cirq.inverse(C0)]))
    assert np.allclose(RXY, RXY_ref)
    YX = cirq.Circuit([cirq.Y(a), cirq.X(b)])
    UYX = cirq.unitary(YX)
    RYX_ref = scipy.linalg.expm(1j * theta * UYX / 2)
    RYX = cirq.unitary(cirq.Circuit([C1, cirq.Rz(rads=-theta)(b), cirq.inverse(C1)]))
    assert np.allclose(RYX, RYX_ref)


def test_givens_angles_original():
    def givens_matrix(theta, p, q, phi, norb):
        mat = np.eye(norb, dtype=np.complex128)
        mat[p, p] = np.cos(theta)
        mat[p, q] = -np.exp(1j * phi) * np.sin(theta)
        mat[q, p] = np.sin(theta)
        mat[q, q] = np.exp(1j * phi) * np.cos(theta)
        return mat

    norb = 6
    u = np.random.random((norb, norb))
    u = u + u.T
    u, _ = np.linalg.qr(u)
    from openfermion.linalg.givens_rotations import givens_decomposition_square

    decomp, diagonal = givens_decomposition_square(u)
    D = np.diag(diagonal)
    U = np.eye(norb)
    for x in decomp:
        prod_g = np.eye(norb)
        for parallel_ops in reversed(x):
            p, q, theta, phi = parallel_ops
            g = givens_matrix(theta, p, q, phi, norb)
            prod_g = prod_g @ g
        U = prod_g @ U
    U = D.dot(U)


# def test_givens_unitary():
#     num_orb = 4
#     mat = np.random.random((num_orb, num_orb))
#     mat = 0.5 * (mat + mat.T)
#     unitary, _ = np.linalg.qr(mat)
#     assert np.allclose(unitary.T @ unitary, np.eye(num_orb))
#     thetas = find_givens_angles(unitary)
#     qubits = cirq.LineQubit.range(num_orb)
#     from openfermion.ops import FermionOperator
#     from openfermion.transforms import jordan_wigner, qubit_operator_to_pauli_sum

#     def majoranas_as_paulis(p, qubits=None):
#         a_p = FermionOperator(f'{p}')
#         a_p_dag = FermionOperator(f'{p}^')
#         maj_0 = a_p + a_p_dag
#         maj_1 = -1j * (a_p - a_p_dag)
#         return (
#             qubit_operator_to_pauli_sum(jordan_wigner(maj_0), qubits=qubits),
#             qubit_operator_to_pauli_sum(jordan_wigner(maj_1), qubits=qubits),
#         )

#     g0, g1 = majoranas_as_paulis(2, qubits=qubits)
#     # assert g0.matrix().shape == (2**num_orb, 2**num_orb)

#     # def build_vop(u, p, theta, qubits):
#     #     id_before = cirq.IdentityGate(qubits[p])
#     #     id_after = cirq.IdentityGate(qubits[p + 1])
#     #     RXY = scipy.linalg.expm(1j * theta * UXY.matrix())
#     #     return RXY

#     # U = np.eye(2**num_orb)
#     # for p in range(num_orb - 1):
#     #     Vp = build_vop(0, p, thetas[0, p], qubits)
#     #     U = np.dot(U, Vp)

#     # Z = cirq.unitary(cirq.Circuit(cirq.Z(qubits[0])  + [cirq.IdentityGate(q) for q in qubits[1:]]))
#     # maj_0, maj_1 = zip(*[build_majoranas(p, qubits) for p in range(num_orb)])
#     # trans = U.conj().T @ Z @ U
