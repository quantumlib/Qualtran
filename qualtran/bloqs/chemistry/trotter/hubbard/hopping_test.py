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
import scipy.linalg
from openfermion import (
    FermionOperator,
    get_sparse_operator,
    jordan_wigner,
    QuadraticHamiltonian,
    QubitOperator,
)

from qualtran import Bloq
from qualtran.bloqs.basic_gates import Rz, TGate, ZPowGate
from qualtran.bloqs.chemistry.trotter.hubbard.hopping import (
    _hopping_tile,
    _hopping_tile_hwp,
    _plaquette,
    BasisChange,
    HoppingPlaquette,
    Rxx,
    Ryy,
)
from qualtran.bloqs.qft.two_bit_ffft import TwoBitFFFT
from qualtran.bloqs.util_bloqs import ArbitraryClifford
from qualtran.resource_counting.generalizers import PHI


def test_hopping_tile(bloq_autotester):
    bloq_autotester(_hopping_tile)


def test_hopping_tile_hwp(bloq_autotester):
    bloq_autotester(_hopping_tile_hwp)


def test_hopping_plaquette(bloq_autotester):
    bloq_autotester(_plaquette)


def catch_rotations(bloq) -> Bloq:
    if isinstance(bloq, (Rz, ZPowGate)):
        if isinstance(bloq, ZPowGate):
            return Rz(angle=PHI)
        elif abs(bloq.angle) < 1e-12:
            return ArbitraryClifford(1)
        else:
            return Rz(angle=PHI)
    return bloq


def test_plaquette_decomposition_tensor_contract():
    q1, q2 = cirq.LineQubit.range(2)
    alpha = 0.123
    xx = cirq.PauliString({q1: cirq.X, q2: cirq.X}, coefficient=1)
    yy = cirq.PauliString({q1: cirq.Y, q2: cirq.Y}, coefficient=1)
    expxx = scipy.linalg.expm(1j * alpha * cirq.PauliSum.from_pauli_strings(xx).matrix())
    expyy = scipy.linalg.expm(1j * alpha * cirq.PauliSum.from_pauli_strings(yy).matrix())
    givens_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(2 * alpha), 1j * np.sin(2 * alpha), 0],
            [0, 1j * np.sin(2 * alpha), np.cos(2 * alpha), 0],
            [0, 0, 0, 1],
        ]
    )
    assert np.allclose(expxx @ expyy, givens_matrix)
    fmat = TwoBitFFFT(0, 1).tensor_contract()
    # Test E13
    # minus signs are flipped here relative to the expression in the paper
    exp_n_0 = scipy.linalg.expm(
        -1j * 2 * alpha * get_sparse_operator(FermionOperator('0^ 0'), n_qubits=2).toarray()
    )
    exp_n_1 = scipy.linalg.expm(
        1j * 2 * alpha * get_sparse_operator(FermionOperator('1^ 1'), n_qubits=2).toarray()
    )
    xx_yy = fmat @ exp_n_0 @ exp_n_1 @ fmat
    assert np.allclose(xx_yy, givens_matrix)
    # Test E10
    rxx = Rxx(2 * alpha).tensor_contract()
    ryy = Ryy(2 * alpha).tensor_contract()
    assert np.allclose(rxx, expxx)
    assert np.allclose(ryy, expyy)
    assert np.allclose(rxx @ ryy, xx_yy)
    plaq = HoppingPlaquette(alpha).tensor_contract()
    # build E4
    # doesn't work yet.
    V = BasisChange().tensor_contract()
    print(jordan_wigner(FermionOperator('0')))
    print(jordan_wigner(FermionOperator('1')))
    a_0 = get_sparse_operator(FermionOperator('0'), n_qubits=2).toarray()
    a_1 = get_sparse_operator(FermionOperator('1'), n_qubits=2).toarray()
    print(a_0)
    print(a_1)
    print(fmat)
    x = fmat @ a_0 @ fmat.conj().T
    print(x)
    print((a_1 - a_0) / np.sqrt(2))
    assert np.allclose(x, (a_1 - a_0) / np.sqrt(2))
    x = fmat @ a_1 @ fmat.conj().T
    assert np.allclose(x, (a_1 + a_0) / np.sqrt(2))
    # b = get_sparse_operator(
    #     FermionOperator('0') + FermionOperator('1') + FermionOperator('2') + FermionOperator('3'),
    #     n_qubits=4,
    # ).toarray()
    # print(V @ a_2 @ V.conj().T)
    # assert np.allclose(V @ a_2 @ V.conj().T, 0.5 * b)
    # op = 0
    # for i in range(4):
    #     for j in range(4):
    #         if (j + 1) % 4 == i or (i + 1) % 4 == j:
    #             op += FermionOperator(f'{i}^ {j}')
    # k_mat = scipy.linalg.expm(1j * alpha * get_sparse_operator(op).toarray())
    # x, y = np.where(np.abs(k_mat) > 1e-12)
    # for i, j in zip(x, y):
    #     print(i, j, k_mat[i, j], plaq[i, j])
    # print(plaq[np.where(np.abs(k_mat) > 1e-12)])
    # print(np.where(abs(plaq - k_mat) > 1e-12))
    # assert np.allclose(plaq, k_mat)


def test_hopping_tile_t_counts():
    bloq = _hopping_tile()
    _, counts = bloq.call_graph(generalizer=catch_rotations)
    assert counts[TGate()] == 8 * bloq.length**2 // 2
    assert counts[Rz(PHI)] == 2 * bloq.length**2 // 2


def test_hopping_tile_hwp_t_counts():
    bloq = _hopping_tile_hwp()
    _, counts = bloq.call_graph(generalizer=catch_rotations)
    n_rot_par = bloq.length**2 // 2
    print(counts, 2 * 4 * (n_rot_par - n_rot_par.bit_count()) - counts[TGate()])
    assert counts[Rz(PHI)] == 2 * n_rot_par.bit_length()
    assert counts[TGate()] == 8 * bloq.length**2 // 2 + 2 * 4 * (n_rot_par - n_rot_par.bit_count())
