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
from openfermion.resource_estimates.sparse.costing_sparse import cost_sparse
from openfermion.resource_estimates.utils import power_two

from qualtran import Bloq
from qualtran.bloqs.arithmetic.comparison import LessThanEqual
from qualtran.bloqs.chemistry.sparse import PrepareSparse, SelectSparse
from qualtran.bloqs.chemistry.sparse.prepare_test import build_random_test_integrals
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
    PrepareUniformSuperposition,
)
from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.resource_counting.generalizers import generalize_cswap_approx
from qualtran.symbolics import SymbolicInt
from qualtran.testing import execute_notebook


def make_prep_sparse(num_spin_orb, num_bits_state_prep, num_bits_rot_aa):
    tpq, eris = build_random_test_integrals(num_spin_orb // 2)
    prep_sparse = PrepareSparse.from_hamiltonian_coeffs(
        num_spin_orb,
        tpq,
        eris,
        log_block_size=5,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa=num_bits_rot_aa,
    )
    num_nnz = len(prep_sparse.alt_pqrs[0])
    return prep_sparse, num_nnz


def get_toffoli_count(bloq: Bloq) -> SymbolicInt:
    """Get the toffoli count of a bloq assuming no raw Ts."""
    counts = get_cost_value(bloq, QECGatesCost(), generalizer=generalize_cswap_approx)
    cost_dict = counts.total_t_and_ccz_count(ts_per_rotation=0)
    assert cost_dict['n_t'] == 0
    return cost_dict['n_ccz']


@pytest.mark.parametrize("num_spin_orb, num_bits_rot_aa", ((8, 3), (12, 4), (16, 3)))
def test_sparse_costs_against_openfermion(num_spin_orb, num_bits_rot_aa):
    num_bits_state_prep = 12
    sel_sparse = SelectSparse(num_spin_orb)
    cost = get_toffoli_count(sel_sparse)
    prep_sparse, num_non_zero = make_prep_sparse(num_spin_orb, num_bits_state_prep, num_bits_rot_aa)
    cost += get_toffoli_count(prep_sparse)
    prep_sparse_adj = prep_sparse.adjoint()
    cost += get_toffoli_count(prep_sparse_adj)
    unused_lambda = 10
    unused_de = 1e-3
    unused_stps = 10
    logd = (num_non_zero - 1).bit_length()
    refl_cost = num_bits_state_prep + logd + 4  # page 40 listing 4
    # correct the expected cost by using a different uniform superposition algorithm
    # see: https://github.com/quantumlib/Qualtran/issues/611
    eta = power_two(prep_sparse.num_non_zero)
    cost_uni_prep = 2 * (
        3 * (prep_sparse.num_non_zero - 1).bit_length() - 3 * eta + 2 * num_bits_rot_aa - 9
    )
    prep_uni = PrepareUniformSuperposition(prep_sparse.num_non_zero)
    delta_uni_prep = (
        get_toffoli_count(prep_uni) + get_toffoli_count(prep_uni.adjoint()) - cost_uni_prep
    )
    # The -2 comes from a more accurate calculation of the QROAM costs in
    # Qualtran (constants are not ignored).  The difference arises from
    # uncontrolled unary iteration used by QROM, which QROAMClean delegates to.
    delta_qrom = -2
    # inequality test difference
    # https://github.com/quantumlib/Qualtran/issues/235
    lte = LessThanEqual(prep_sparse.num_bits_state_prep, prep_sparse.num_bits_state_prep)
    lte_cost = get_toffoli_count(lte) + get_toffoli_count(lte.adjoint())
    lte_cost_paper = prep_sparse.num_bits_state_prep  # inverted at zero cost
    delta_ineq = lte_cost - lte_cost_paper
    swap_cost = 8 * (num_spin_orb // 2 - 1).bit_length() + 1  # inverted at zero cost
    adjusted_cost_qualtran = cost - delta_qrom - delta_uni_prep - delta_ineq - swap_cost
    cost_of = cost_sparse(
        num_spin_orb, unused_lambda, num_non_zero, unused_de, num_bits_state_prep, unused_stps
    )[0]
    assert adjusted_cost_qualtran == cost_of - refl_cost


@pytest.mark.notebook
def test_notebook():
    execute_notebook("sparse")
