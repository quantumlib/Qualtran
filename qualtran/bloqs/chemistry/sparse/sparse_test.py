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
from typing import Optional, Union

import attrs
import numpy as np
import pytest
from openfermion.resource_estimates.sparse.costing_sparse import cost_sparse
from openfermion.resource_estimates.utils import power_two, QI

from qualtran import Adjoint, Bloq
from qualtran.bloqs.arithmetic.comparison import LessThanEqual
from qualtran.bloqs.chemistry.sparse import PrepareSparse, SelectSparse
from qualtran.bloqs.chemistry.sparse.prepare_test import build_random_test_integrals
from qualtran.bloqs.data_loading.select_swap_qrom import find_optimal_log_block_size, SelectSwapQROM
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
        qroam_block_size=32,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa=num_bits_rot_aa,
    )
    num_nnz = len(prep_sparse.alt_pqrs[0])
    return prep_sparse, num_nnz


def qrom_cost(prep: PrepareSparse) -> int:
    """Get the paper qrom cost."""
    n_n = (prep.num_spin_orb // 2 - 1).bit_length()
    if prep.qroam_block_size is None:
        target_bitsizes = (
            (n_n,) * 4 + (1,) * 2 + (n_n,) * 4 + (1,) * 2 + (prep.num_bits_state_prep,)
        )
        block_size = 2 ** find_optimal_log_block_size(prep.num_non_zero, sum(target_bitsizes))
    else:
        block_size = prep.qroam_block_size
    if prep.is_adjoint:
        num_toff_qrom = int(np.ceil(prep.num_non_zero / block_size)) + block_size  # A15
    else:
        output_size = prep.num_bits_state_prep + 8 * n_n + 4
        num_toff_qrom = int(np.ceil(prep.num_non_zero / block_size)) + output_size * (
            block_size - 1
        )  # A14
    return num_toff_qrom


def get_toffoli_count(bloq: Bloq) -> SymbolicInt:
    """Get the toffoli count of a bloq assuming no raw Ts."""
    counts = get_cost_value(bloq, QECGatesCost(), generalizer=generalize_cswap_approx)
    cost_dict = counts.total_t_and_ccz_count(ts_per_rotation=0)
    assert cost_dict['n_t'] == 0
    return cost_dict['n_ccz']


def get_sel_swap_qrom_toff_count(prep: PrepareSparse) -> SymbolicInt:
    """Utility function to pick out the SelectSwapQROM cost from the prepare call graph."""

    def keep_qrom(bloq):
        if isinstance(bloq, SelectSwapQROM):
            return True
        return False

    _, sigma = prep.call_graph(keep=keep_qrom)
    qrom_bloq: Optional[Union[SelectSwapQROM, Adjoint]] = None
    for k in sigma.keys():
        if isinstance(k, SelectSwapQROM):
            qrom_bloq = k
            break
        if isinstance(k, Adjoint) and isinstance(k.subbloq, SelectSwapQROM):
            qrom_bloq = k
            break
    if qrom_bloq is None:
        return 0
    return get_toffoli_count(qrom_bloq)


@pytest.mark.parametrize("num_spin_orb, num_bits_rot_aa", ((8, 3), (12, 4), (16, 3)))
def test_sparse_costs_against_openfermion(num_spin_orb, num_bits_rot_aa):
    num_bits_state_prep = 12
    sel_sparse = SelectSparse(num_spin_orb)
    cost = get_toffoli_count(sel_sparse)
    prep_sparse, num_non_zero = make_prep_sparse(num_spin_orb, num_bits_state_prep, num_bits_rot_aa)
    cost += get_toffoli_count(prep_sparse)
    prep_sparse_adj = attrs.evolve(
        prep_sparse, is_adjoint=True, qroam_block_size=2 ** QI(num_non_zero)[0]
    )
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
    # correct for SelectSwapQROM vs QROAM
    # https://github.com/quantumlib/Qualtran/issues/574
    paper_qrom = qrom_cost(prep_sparse)
    paper_qrom += qrom_cost(prep_sparse_adj)
    qual_qrom_cost = get_sel_swap_qrom_toff_count(prep_sparse)
    qual_qrom_cost += get_sel_swap_qrom_toff_count(prep_sparse_adj)
    delta_qrom = qual_qrom_cost - paper_qrom
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
