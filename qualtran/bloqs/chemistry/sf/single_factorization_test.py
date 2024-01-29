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
from openfermion.resource_estimates.utils import power_two, QI, QI2, QR2

from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.chemistry.sf.single_factorization import (
    _sf_block_encoding,
    _sf_one_body,
    SingleFactorizationBlockEncoding,
    SingleFactorizationOneBody,
)
from qualtran.bloqs.prepare_uniform_superposition import PrepareUniformSuperposition
from qualtran.testing import execute_notebook


def test_sf_block_encoding(bloq_autotester):
    bloq_autotester(_sf_block_encoding)


def test_one_body_block_encoding(bloq_autotester):
    bloq_autotester(_sf_one_body)


def test_compare_cost_one_body_decomp():
    num_spin_orb = 10
    num_aux = 50
    num_bits_state_prep = 12
    num_bits_rot_aa = 7
    bloq = SingleFactorizationOneBody(
        num_aux=num_aux,
        num_spin_orb=num_spin_orb,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa=num_bits_rot_aa,
    )
    costs = bloq.call_graph()[1]
    cbloq_costs = bloq.decompose_bloq().call_graph()[1]
    assert costs[TGate()] == cbloq_costs[TGate()]


def test_compare_cost_to_openfermion():
    num_spin_orb = 108
    num_aux = 200
    num_bits_state_prep = 10
    num_bits_rot_aa_outer = 5  # captured from OF.
    num_bits_rot_aa_inner = 7  # OF
    unused_lambda = 10
    unused_de = 1e-3
    unused_stps = 100
    # See https://github.com/quantumlib/OpenFermion/issues/838
    bloq = SingleFactorizationBlockEncoding(
        num_spin_orb,
        num_aux,
        num_bits_state_prep,
        num_bits_rot_aa_outer,
        num_bits_rot_aa_inner,
        kp1=2**2,
        kp2=2**5,
        kp1_inv=2**1,
        kp2_inv=2**8,
    )
    of_cost, _, _ = compute_cost(
        num_spin_orb, unused_lambda, unused_de, num_aux, num_bits_state_prep, unused_stps
    )
    _, counts = bloq.call_graph()
    nl = num_aux.bit_length()
    nn = (num_spin_orb // 2 - 1).bit_length()
    cost_refl = nl + 2 * nn + 2 * num_bits_state_prep + 2
    cost_qualtran = counts[TGate()]
    delta_refl = num_bits_state_prep + 1  # missing bits of reflection for state preparation.
    cost_qualtran -= (
        # there are 4 swaps costed as Toffolis in openfermion, qualtran uses 7 T gates.
        4 * (7 - 4) * nn
        +
        # inner prepare differences (swaps + inequality test)
        4 * ((7 - 4) * (2 * nn) + 4 * num_bits_state_prep - 4)
        +
        # outer prepare differences (swaps + inequality test)
        2 * ((7 - 4) * (nl + 1) + 4 * num_bits_state_prep - 4)
    )
    cost_qualtran //= 4
    # correct the expected cost by using a different uniform superposition algorithm
    # https://github.com/quantumlib/Qualtran/issues/611
    eta = power_two(num_aux + 1)
    cost1a = 2 * (3 * nl - 3 * eta + 2 * num_bits_rot_aa_outer - 9)
    prep = PrepareUniformSuperposition(num_aux + 1)
    cost1a_mod = prep.call_graph()[1][TGate()] // 4
    cost1a_mod += prep.adjoint().call_graph()[1][TGate()] // 4
    delta_uni_prep = cost1a_mod - cost1a
    cost_qualtran -= delta_uni_prep
    cost_qualtran += delta_refl
    cost_qualtran += cost_refl
    cost_qualtran += 1  # extra toffoli currently missing for additional control
    cost_qualtran += 2  # controlling phase estimation
    nprime = int(num_spin_orb**2 // 8 + num_spin_orb // 4)
    bp = int(2 * nn + num_bits_state_prep + 2)
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
    of_cost += our_qrom_cost - cost2c
    assert cost_qualtran == of_cost


def test_notebook():
    execute_notebook("single_factorization")
