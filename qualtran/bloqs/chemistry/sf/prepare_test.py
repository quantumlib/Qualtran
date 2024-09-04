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

from qualtran.bloqs.chemistry.sf.prepare import (
    _prep_inner,
    _prep_outer,
    InnerPrepareSingleFactorization,
    OuterPrepareSingleFactorization,
)
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
    PrepareUniformSuperposition,
)
from qualtran.resource_counting import get_cost_value, QECGatesCost


def test_prep_inner(bloq_autotester):
    bloq_autotester(_prep_inner)


def test_prep_outer(bloq_autotester):
    bloq_autotester(_prep_outer)


def test_outerprep_t_counts():
    # Reiher et al hamiltonian parameters (from openfermion unit tests,
    # resource_estimates/sf/compute_cost_sf_test.py)
    num_aux = 200
    num_bits_state_prep = 10
    num_bits_rot_aa = 7
    outer_prep = OuterPrepareSingleFactorization(
        num_aux, num_bits_state_prep=num_bits_state_prep, num_bits_rot_aa=num_bits_rot_aa
    )
    # Note: this contains rotations, but we just divide the total T count by 4 to approximate
    # the toffoli count: https://github.com/quantumlib/Qualtran/issues/390
    toff = get_cost_value(outer_prep, QECGatesCost()).total_t_count(ts_per_cswap=4)
    toff -= 4 * (num_bits_state_prep - 1)
    nb_l = num_aux.bit_length()
    outer_prep = OuterPrepareSingleFactorization(
        num_aux, num_bits_state_prep=num_bits_state_prep, num_bits_rot_aa=num_bits_rot_aa
    ).adjoint()
    toff += get_cost_value(outer_prep, QECGatesCost()).total_t_count(ts_per_cswap=4)
    # See https://github.com/quantumlib/Qualtran/issues/390
    # inequality difference
    toff -= 4 * (num_bits_state_prep - 1)
    toff //= 4
    # Number of qubits for the first register
    # Number of qubits for p and q registers
    eta = power_two(num_aux + 1)
    cost1a = 2 * (3 * nb_l - 3 * eta + 2 * num_bits_rot_aa - 9)
    # correct the expected cost by using a different uniform superposition algorithm
    # see: https://github.com/quantumlib/Qualtran/issues/611
    prep = PrepareUniformSuperposition(num_aux + 1)
    cost1a_mod = get_cost_value(prep, QECGatesCost()).total_t_count(ts_per_cswap=4) // 4
    cost1a_mod += get_cost_value(prep.adjoint(), QECGatesCost()).total_t_count(ts_per_cswap=4) // 4
    assert cost1a != cost1a_mod
    bL = nb_l + num_bits_state_prep + 2
    cost1b = QR(num_aux + 1, bL)[-1] + QI(num_aux + 1)[-1]
    cost1cd = 2 * (num_bits_state_prep + nb_l + 1)
    of_cost = cost1a_mod + cost1b + cost1cd
    toff -= 1  # TODO: https://github.com/quantumlib/Qualtran/issues/1391
    assert toff == of_cost


def test_inner_prepare_t_counts():
    num_spin_orb = 108
    num_aux = 200
    num_bits_state_prep = 10
    num_bits_rot_aa = 7
    nN = (num_spin_orb // 2 - 1).bit_length()
    in_prep = InnerPrepareSingleFactorization(
        num_aux=num_aux,
        num_spin_orb=num_spin_orb,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa=num_bits_rot_aa,
        kp1=2**2,
        kp2=2**5,
    )
    # Note: this contains rotations, but we just divide the total T count by 4 to approximate
    # the toffoli count: https://github.com/quantumlib/Qualtran/issues/390
    toff = get_cost_value(in_prep, QECGatesCost()).total_t_count(ts_per_cswap=4)
    # inequality difference
    toff -= 4 * num_bits_state_prep - 4
    in_prep = InnerPrepareSingleFactorization(
        num_aux=num_aux,
        num_spin_orb=num_spin_orb,
        num_bits_rot_aa=num_bits_rot_aa,
        num_bits_state_prep=num_bits_state_prep,
        kp1=2**1,
        kp2=2**8,
    ).adjoint()
    # factor of two from squaring
    toff += get_cost_value(in_prep, QECGatesCost()).total_t_count(ts_per_cswap=4)
    # See https://github.com/quantumlib/Qualtran/issues/390
    # inequality difference
    toff -= 4 * num_bits_state_prep - 4
    toff *= 2
    toff //= 4
    # Number of qubits for p and q registers
    # copied from compute_cost_sf.py
    cost2a = 4 * (6 * nN + 2 * num_bits_rot_aa - 7)
    # Cost of computing contiguous register in step 2 (b).
    cost2b = 4 * (nN**2 + nN - 1)
    # Number of coefficients for first state preparation on p & q.
    nprime = int(num_spin_orb**2 // 8 + num_spin_orb // 4)
    bp = int(2 * nN + num_bits_state_prep + 2)
    cost2c = (
        QR2(num_aux + 1, nprime, bp)[-1]
        + QI((num_aux + 1) * nprime)[-1]
        + QR2(num_aux, nprime, bp)[-1]
        + QI(num_aux * nprime)[-1]
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
