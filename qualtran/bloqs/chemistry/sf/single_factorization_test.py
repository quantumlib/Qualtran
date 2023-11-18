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

from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.chemistry.sf.single_factorization import (
    _sf_block_encoding,
    _sf_one_body,
    SingleFactorizationBlockEncoding,
    SingleFactorizationOneBody,
)
from qualtran.testing import execute_notebook


def test_sf_block_encoding(bloq_autotester):
    bloq_autotester(_sf_block_encoding)


def test_one_body_block_encoding(bloq_autotester):
    bloq_autotester(_sf_one_body)


def test_compare_cost_one_body_decomp():
    num_spin_orb = 10
    num_aux = 50
    num_bits_state_prep = 12
    num_bits_rot = 7
    bloq = SingleFactorizationOneBody(
        num_aux=num_aux,
        num_spin_orb=num_spin_orb,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa=7,
    )
    costs = bloq.call_graph()[1]
    cbloq_costs = bloq.decompose_bloq().call_graph()[1]
    assert costs[TGate()] == cbloq_costs[TGate()]


def test_compare_cost_to_openfermion():
    num_spin_orb = 108
    num_aux = 200
    num_bits_state_prep = 10
    num_bits_rot_aa_outer = 7  # captured from OF.
    num_bits_rot_aa_inner = 7  # OF
    unused_lambda = 10
    unused_de = 1e-3
    unused_stps = 100
    of_cost, _, _ = compute_cost(
        num_spin_orb, unused_lambda, unused_de, num_aux, num_bits_state_prep, unused_stps
    )
    # See https://github.com/quantumlib/OpenFermion/issues/838
    bloq = SingleFactorizationBlockEncoding(
        num_spin_orb,
        num_aux,
        num_bits_state_prep,
        num_bits_rot_aa_outer,
        num_bits_rot_aa_inner,
        kp1=4,
        kp2=2**5,
        kp1_inv=2**2,
        kp2_inv=2**5,
    )
    _, counts = bloq.call_graph()
    nl = num_aux.bit_length()
    nn = (num_spin_orb // 2 - 1).bit_length()
    cost_refl = nl + 2 * nn + 2 * num_bits_state_prep + 2
    cost_qualtran = counts[TGate()] // 4  # + cost_refl + 2
    # there are 4 swaps costed as Toffolis in openfermion, qualtran uses 7 T gates.
    delta_swaps = 4 * (7 - 4) * 3 // 4
    delta_refl = num_bits_state_prep + 1
    cost_qualtran -= delta_swaps  #
    cost_qualtran += delta_refl
    cost_qualtran += cost_refl
    cost_qualtran += 1  # extra toffoli currently missing for additional control
    cost_qualtran += 2  # controlling phase estimation
    cost_qualtran -= (
        61  # https://github.com/quantumlib/OpenFermion/issues/838 cost for fixing OF differences.
    )
    assert cost_qualtran == of_cost


def test_notebook():
    execute_notebook("single_factorization")
