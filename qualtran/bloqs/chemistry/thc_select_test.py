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

import qualtran.testing as qlt_testing
from qualtran.bloqs.chemistry.thc_select import find_givens_angles


def _make_select():
    from qualtran.bloqs.chemistry.thc import SelectTHC

    num_spat = 4
    num_mu = 8
    return SelectTHC(num_mu=num_mu, num_spat=num_spat)


def test_select_thc():
    num_mu = 10
    num_spin_orb = 2 * 4
    select = SelectTHC(num_mu=num_mu, num_spin_orb=num_spin_orb)
    qlt_testing.assert_valid_bloq_decomposition(select)


@pytest.mark.parametrize("theta", 2 * np.pi * np.random.random(10))
def test_interleaved_cliffords(theta):
    a, b = cirq.LineQubit.range(2)
    XY = cirq.Circuit([cirq.X(a), cirq.Y(b)])
    UXY = cirq.unitary(XY)
    RXY_ref = scipy.linalg.expm(-1j * theta * UXY / 2)
    C0 = [cirq.H(a), cirq.S(b) ** -1, cirq.H(b), cirq.CNOT(a, b)]
    C1 = [cirq.S(a), cirq.H(a), cirq.H(b), cirq.CNOT(a, b)]
    RXY = cirq.unitary(cirq.Circuit([C0, cirq.Rz(rads=theta)(b), cirq.inverse(C0)]))
    assert np.allclose(RXY, RXY_ref)
    YX = cirq.Circuit([cirq.Y(a), cirq.X(b)])
    UYX = cirq.unitary(YX)
    RYX_ref = scipy.linalg.expm(1j * theta * UYX / 2)
    RYX = cirq.unitary(cirq.Circuit([C1, cirq.Rz(rads=-theta)(b), cirq.inverse(C1)]))
    assert np.allclose(RYX.T, RYX_ref)


def test_givens_unitary():
    num_orb = 10
    mat = np.random.random((num_orb, num_orb))
    mat = 0.5 * (mat + mat.T)
    unitary, _ = np.linalg.qr(mat)
    assert np.allclose(unitary.T @ unitary, np.eye(num_orb))
    thetas = find_givens_angles(unitary)
    qubits = cirq.LineQubit.range(num_orb)
    from openfermion.linalg import get_sparse_operator
    from openfermion.ops import MajoranaOperator

    gamma_0 = MajoranaOperator(((1), 1))
    print(get_sparse_operator(gamma_0))

    # def build_vop(u, p, theta, qubits):
    #     id_before = cirq.IdentityGate(qubits[p])
    #     # for i in range(p):
    #     #     #print(after)
    #     #     id_before *= cirq.IdentityGate(i)
    #     id_after = cirq.IdentityGate(qubits[p + 1])
    #     # print(id_after)
    #     # for i in range(p + 1, len(qubits)):
    #     #     id_after *= cirq.IdentityGate(i)
    #     print()
    #     print(id_before)
    #     print(id_after)
    #     XY = id_before * cirq.X(qubits[p]) * cirq.Y(qubits[p + 1]) * id_after
    #     RXY = scipy.linalg.expm(1j * theta * UXY.matrix())
    #     return RXY

    # U = np.eye(2**num_orb)
    # for p in range(num_orb - 1):
    #     Vp = build_vop(0, p, thetas[0, p], qubits)
    #     U = np.dot(U, Vp)

    # Z = cirq.unitary(cirq.Circuit(cirq.Z(qubits[0]) + [cirq.IdentityGate(q) for q in qubits[1:]]))
    # maj_0, maj_1 = zip(*[build_majoranas(p, qubits) for p in range(num_orb)])
    # trans = U.conj().T @ Z @ U
