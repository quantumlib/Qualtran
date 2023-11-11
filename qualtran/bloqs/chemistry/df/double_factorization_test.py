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

from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.chemistry.df.double_factorization import (
    _df_block_encoding,
    _df_one_body,
    DoubleFactorizationBlockEncoding,
    DoubleFactorizationOneBody,
)
from qualtran.testing import execute_notebook


def test_prep_inner(bloq_autotester):
    bloq_autotester(_df_block_encoding)


def test_prep_outer(bloq_autotester):
    bloq_autotester(_df_one_body)


def test_compare_cost_one_body():
    num_spin_orb = 10
    num_aux = 50
    num_eig = num_spin_orb // 2
    num_bits_state_prep = 12
    num_bits_rot = 7
    bloq = DoubleFactorizationOneBody(
        num_spin_orb=num_spin_orb,
        num_aux=num_aux,
        num_xi=num_eig,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa=7,
        num_bits_rot=num_bits_rot,
    )
    _, costs = bloq.call_graph()
    qual_costs = 2 * costs[TGate()]
    qual_costs += 4
    # inner prepare
    of_cost = 4 * (497 + 1)
    # rotations
    of_cost += 4 * (615 + 3)
    of_cost += 7 * 4 * (num_spin_orb // 2)  # Note swaps cost 7 Ts if using basic controlled swaps.
    of_cost += 3 * 4
    assert qual_costs == of_cost


def test_compare_cost_to_openfermion():
    num_spin_orb = 10
    num_aux = 50
    num_eig = num_spin_orb // 2
    num_bits_state_prep = 12
    num_bits_rot = 7
    unused_lambda = 10
    unused_de = 1e-3
    unused_stps = 100
    bloq = DoubleFactorizationBlockEncoding(
        num_spin_orb=num_spin_orb,
        num_aux=num_aux,
        num_xi=num_eig,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa_outer=1,
        num_bits_rot_aa_inner=7,
        num_bits_rot=num_bits_rot,
    )
    _, counts = bloq.call_graph()
    # https://github.com/quantumlib/OpenFermion/issues/839
    of_cost = compute_cost(
        num_spin_orb,
        unused_lambda,
        unused_de,
        num_aux,
        num_eig * num_aux,
        num_bits_state_prep,
        num_bits_rot,
        unused_stps,
    )[0]
    # included in OF estimates
    refl_cost = 22
    walk_cost = 2
    # discrepencies with implementation
    in_prep_diff = 1  # shouldn't output one-body second time around
    rot_diff = 3  # Difference from not loading one-body
    in_data_diff = 6  # Hardcoded problem in OF
    diff_ctrl_z = 1  # 1 additional Toffoli from controlling l_ne_zero
    # reflection on ancilla for state prepation (alias sampling) + 1 (they cost it
    # as n_bits not n_bits - 1)
    inner_refl_diff = num_bits_state_prep + 1
    diff = in_prep_diff + rot_diff + in_data_diff - diff_ctrl_z - inner_refl_diff
    qual_cost = counts[TGate()] + 4 * (refl_cost + walk_cost - diff)
    of_cost *= 4
    of_cost += 60
    assert of_cost == qual_cost


def test_notebook():
    execute_notebook("double_factorization")
