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
import pytest
from openfermion.resource_estimates.df.compute_cost_df import compute_cost
from openfermion.resource_estimates.utils import QI, QR

from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.chemistry.double_factorization import (
    DoubleFactorization,
    get_num_bits_lxi,
    get_qroam_cost,
    InnerPrepare,
    OuterPrepare,
    OutputIndexedData,
    ProgRotGateArray,
)
from qualtran.resource_counting import get_bloq_counts_graph
from qualtran.testing import assert_valid_bloq_decomposition, execute_notebook


def test_compare_cost_to_openfermion():
    num_spin_orb = 10
    num_aux = 50
    num_eig = num_spin_orb // 2
    num_bits_state_prep = 12
    num_bits_rot = 7
    unused_lambda = 10
    unused_de = 1e-3
    unused_stps = 100
    bloq = DoubleFactorization(
        num_spin_orb=num_spin_orb,
        num_aux=num_aux,
        num_xi=num_eig,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa_outer=1,
        num_bits_rot_aa_inner=7,
        num_bits_rot=num_bits_rot,
    )
    _, counts = get_bloq_counts_graph(bloq)
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
    refl_cost = 22
    walk_cost = 2
    in_prep_diff = 1
    rot_diff = 3
    in_data_diff = 6
    diff_ctrl_z = 1
    diff = in_prep_diff + rot_diff + in_data_diff - diff_ctrl_z
    assert of_cost == counts[TGate()] // 4 + refl_cost + walk_cost - diff


def test_notebook():
    execute_notebook("double_factorization")
