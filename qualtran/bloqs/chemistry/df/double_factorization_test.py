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
from openfermion.resource_estimates.df.compute_cost_df import compute_cost
from openfermion.resource_estimates.utils import power_two

from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.chemistry.df.double_factorization import (
    _df_block_encoding,
    _df_one_body,
    DoubleFactorizationBlockEncoding,
    DoubleFactorizationOneBody,
)
from qualtran.bloqs.prepare_uniform_superposition import PrepareUniformSuperposition
from qualtran.testing import execute_notebook


def test_df_block_encoding(bloq_autotester):
    bloq_autotester(_df_block_encoding)


def test_one_body_block_encoding(bloq_autotester):
    bloq_autotester(_df_one_body)


def test_compare_cost_one_body_decomp():
    num_spin_orb = 10
    num_aux = 50
    num_eig = num_aux * num_spin_orb // 2
    num_bits_state_prep = 12
    num_bits_rot = 7
    bloq = DoubleFactorizationOneBody(
        num_spin_orb=num_spin_orb,
        num_aux=num_aux,
        num_eig=num_eig,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa=7,
        num_bits_rot=num_bits_rot,
    )
    costs = bloq.call_graph()[1]
    cbloq_costs = bloq.decompose_bloq().call_graph()[1]
    assert costs[TGate()] == cbloq_costs[TGate()]


def test_compare_cost_to_openfermion():
    num_spin_orb = 108
    num_aux = 360
    num_bits_rot = 16
    num_eig = 13031
    num_bits_state_prep = 10
    lambd = 294.8
    bloq = DoubleFactorizationBlockEncoding(
        num_spin_orb=num_spin_orb,
        num_aux=num_aux,
        num_eig=num_eig,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa_outer=7,
        num_bits_rot_aa_inner=7,
        num_bits_rot=num_bits_rot,
    )
    _, counts = bloq.call_graph()
    # https://github.com/quantumlib/OpenFermion/issues/839
    of_cost = compute_cost(
        num_spin_orb, lambd, 1e-3, num_aux, num_eig, num_bits_state_prep, num_bits_rot, 10_000
    )[0]
    # included in OF estimates
    nxi = (num_spin_orb // 2 - 1).bit_length()
    nl = num_aux.bit_length()
    inner_prep_qrom_diff = 8
    prog_rot_qrom_diff = 60
    missing_toffoli = 4  # need one more toffoli for second application of CZ
    swap_cost = 4 * (7 - 4) * num_spin_orb // 2
    qual_cost = (
        counts[TGate()] - inner_prep_qrom_diff - prog_rot_qrom_diff + missing_toffoli - swap_cost
    )
    # correct the expected cost by using a different uniform superposition algorithm
    # see: https://github.com/quantumlib/Qualtran/issues/611
    eta = power_two(num_aux + 1)
    cost1a = 4 * 2 * (3 * nl - 3 * eta + 2 * 7 - 9)
    prep = PrepareUniformSuperposition(num_aux + 1)
    cost1a_mod = prep.call_graph()[1][TGate()]
    cost1a_mod += prep.adjoint().call_graph()[1][TGate()]
    delta_uni_prep = cost1a_mod - cost1a
    qual_cost -= delta_uni_prep
    inner_refl = num_bits_state_prep + 1
    walk_refl = nl + nxi + num_bits_state_prep + 1
    qpe_toff = 2
    of_cost = of_cost - inner_refl - walk_refl - qpe_toff
    of_cost *= 4
    assert of_cost == qual_cost


def test_notebook():
    execute_notebook("double_factorization")
