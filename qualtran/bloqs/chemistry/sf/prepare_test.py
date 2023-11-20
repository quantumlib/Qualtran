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

from openfermion.resource_estimates.utils import power_two, QI, QI2, QR, QR2

from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.chemistry.sf.prepare import (
    _prep_inner,
    _prep_outer,
    InnerPrepareSingleFactorization,
    OuterPrepareSingleFactorization,
)


def test_prep_inner(bloq_autotester):
    bloq_autotester(_prep_inner)


def test_prep_outer(bloq_autotester):
    bloq_autotester(_prep_outer)


def test_outerprep_t_counts():
    # Reiher et al hamiltonian parameters (from openfermion unit tests,
    # resource_estimates/sf/compute_cost_sf_test.py)
    num_spin_orb = 108
    num_aux = 200
    num_bits_state_prep = 10
    num_bits_rot_aa = 7
    outer_prep = OuterPrepareSingleFactorization(
        num_aux, num_bits_state_prep=num_bits_state_prep, num_bits_rot_aa=num_bits_rot_aa
    )
    _, counts = outer_prep.call_graph()
    toff = counts[TGate()] // 4
    outer_prep = OuterPrepareSingleFactorization(
        num_aux,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa=num_bits_rot_aa,
        adjoint=True,
    )
    eta = power_two(num_aux + 1)
    _, counts = outer_prep.call_graph()
    toff += counts[TGate()] // 4
    # Number of qubits for the first register
    nL = num_aux.bit_length()
    # Number of qubits for p and q registers
    cost1a = 2 * (3 * nL - 3 * eta + 2 * num_bits_rot_aa - 9)
    bL = nL + num_bits_state_prep + 2
    cost1b = QR(num_aux + 1, bL)[-1] + QI(num_aux + 1)[-1]
    cost1cd = 2 * (num_bits_state_prep + nL + 1)
    assert toff == (cost1a + cost1b + cost1cd)


def test_inner_prepare_t_counts():
    num_spin_orb = 108
    num_aux = 200
    num_bits_state_prep = 10
    num_bits_rot_aa = 7
    in_prep = InnerPrepareSingleFactorization(
        num_aux=num_aux,
        num_spin_orb=num_spin_orb,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa=num_bits_rot_aa,
        adjoint=False,
        kp1=2**1,
        kp2=2**6,
    )
    _, counts = in_prep.call_graph()
    toff = counts[TGate()] // 4
    in_prep = InnerPrepareSingleFactorization(
        num_aux=num_aux,
        num_spin_orb=num_spin_orb,
        num_bits_rot_aa=num_bits_rot_aa,
        num_bits_state_prep=num_bits_state_prep,
        adjoint=True,
        kp1=2**1,
        kp2=2**8,
    )
    _, counts = in_prep.call_graph()
    # factor of two from squaring
    toff += counts[TGate()] // 4
    toff *= 2
    # Number of qubits for p and q registers
    # copied from compute_cost_sf.py
    nN = (num_spin_orb // 2 - 1).bit_length()
    cost2a = 4 * (6 * nN + 2 * num_bits_rot_aa - 7)
    # Cost of computing contiguous register in step 2 (b).
    cost2b = 4 * (nN**2 + nN - 1)
    # Number of coefficients for first state preparation on p & q.
    # correct the data size here: https://github.com/quantumlib/OpenFermion/issues/838
    nprime = int(num_spin_orb**2 // 8 + num_spin_orb // 2)
    nprime_err = int(num_spin_orb**2 // 8 + num_spin_orb // 4)
    bp = int(2 * nN + num_bits_state_prep + 2)
    cost2c = (
        QR2(num_aux + 1, nprime_err, bp)[-1]
        + QI((num_aux + 1) * nprime_err)[-1]
        + QR2(num_aux, nprime_err, bp)[-1]
        + QI(num_aux * nprime_err)[-1]
    )
    our_qrom_cost = (
        QR2(num_aux + 1, nprime, bp)[-1]
        + QI2(num_aux + 1, nprime)[-1]
        + QR2(num_aux + 1, nprime, bp)[-1]
        + QI2(num_aux + 1, nprime)[-1]
    )
    # See https://github.com/quantumlib/Qualtran/issues/526
    delta_qrom = our_qrom_cost - cost2c
    cost2de = 4 * (num_bits_state_prep + 2 * nN)
    of_cost = cost2a + cost2b + cost2c + cost2de + delta_qrom
    assert toff == of_cost
