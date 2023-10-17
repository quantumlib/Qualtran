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

from openfermion.resource_estimates.sf.compute_cost_sf import compute_cost
from openfermion.resource_estimates.utils import QI, QI2, QR2

from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.chemistry.single_factorization import SingleFactorization
from qualtran.resource_counting import get_bloq_counts_graph
from qualtran.testing import assert_valid_bloq_decomposition, execute_notebook


def _make_single_factorization():
    from qualtran.bloqs.chemistry.single_factorization import SingleFactorization

    return SingleFactorization(10, 20, 8)


def test_single_factorization():
    sf = SingleFactorization(10, 12, 8)
    assert_valid_bloq_decomposition(sf)


def test_compare_cost_to_openfermion():
    num_spin_orb = 10
    num_aux = 50
    num_bits_state_prep = 12
    num_bits_rot_aa_outer = 1  # captured from OF.
    num_bits_rot_aa_inner = 6
    unused_lambda = 10
    unused_de = 1e-3
    unused_stps = 100
    of_cost, _, _ = compute_cost(
        num_spin_orb, unused_lambda, unused_de, num_aux, num_bits_state_prep, unused_stps
    )
    # See https://github.com/quantumlib/OpenFermion/issues/838
    bloq = SingleFactorization(
        num_spin_orb,
        num_aux,
        num_bits_state_prep,
        num_bits_rot_aa_outer,
        num_bits_rot_aa_inner,
        kl=2**1,
        kl_inv=2**3,
        kp1=4,
        kp1_inv=2**3,
        kp2=2,
        kp2_inv=2**2,
    )
    _, counts = get_bloq_counts_graph(bloq)
    nl = num_aux.bit_length()
    nn = (num_spin_orb // 2 - 1).bit_length()
    cost_refl = nl + 2 * nn + 2 * num_bits_state_prep + 2
    nprime_of = num_spin_orb**2 // 8 + num_spin_orb // 4
    nprime = num_spin_orb**2 // 8 + num_spin_orb // 2
    bp = 2 * nn + num_bits_state_prep + 2
    # There is a discrepency between what is in OF and what is in the paper.
    # Here we correct the OF costs using the correct QROM costs and data sizes.
    qr_a_of = QR2(num_aux + 1, nprime_of, bp)[-1]
    qr_b_of = QR2(num_aux, nprime_of, bp)[-1]
    delta_a = QR2(num_aux + 1, nprime, bp)[-1] - qr_a_of
    delta_b = QR2(num_aux, nprime, bp)[-1] - qr_b_of
    qi_a_of = QI((num_aux + 1) * nprime_of)[-1]
    qi_b_of = QI((num_aux) * nprime_of)[-1]
    delta_a_inv = QI2(num_aux + 1, nprime)[-1] - qi_a_of
    delta_b_inv = QI2(num_aux, nprime)[-1] - qi_b_of
    # Add in phase estimation + walk reflection costs which are absent in qualtran but present in OF.
    cost_qualtran = counts[TGate()] // 4 + cost_refl + 2
    of_cost += delta_a + delta_b
    of_cost += delta_a_inv + delta_b_inv
    # + 1 for controlled select, missing in qualtran
    assert cost_qualtran + 1 == of_cost


def test_notebook():
    execute_notebook("single_factorization")
