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
import attrs
import numpy as np
import pytest
from openfermion.resource_estimates.sparse.costing_sparse import cost_sparse
from openfermion.resource_estimates.utils import power_two, QI

from qualtran.bloqs.arithmetic.comparison import LessThanEqual
from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.chemistry.sparse import PrepareSparse, SelectSparse
from qualtran.bloqs.chemistry.sparse.prepare_test import build_random_test_integrals
from qualtran.bloqs.data_loading.select_swap_qrom import find_optimal_log_block_size, SelectSwapQROM
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
    PrepareUniformSuperposition,
)
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


def get_sel_swap_qrom_t_count(prep: PrepareSparse) -> int:
    """Utility function to pick out the SelectSwapQROM cost from the prepare call graph."""

    def keep_qrom(bloq):
        if isinstance(bloq, SelectSwapQROM):
            return True
        return False

    _, sigma = prep.call_graph(keep=keep_qrom)
    qrom_bloq = None
    for k in sigma.keys():
        if isinstance(k, SelectSwapQROM):
            qrom_bloq = k
            break
    if qrom_bloq is None:
        return 0
    return int(qrom_bloq.call_graph()[1].get(TGate(), 0))


@pytest.mark.slow
@pytest.mark.parametrize("num_spin_orb, num_bits_rot_aa", ((8, 3), (12, 4), (16, 3)))
def test_sparse_costs_against_openfermion(num_spin_orb, num_bits_rot_aa):
    num_bits_state_prep = 12
    cost = 0
    bloq = SelectSparse(num_spin_orb)
    _, sigma = bloq.call_graph()
    cost += sigma[TGate()]
    prep_sparse, num_non_zero = make_prep_sparse(num_spin_orb, num_bits_state_prep, num_bits_rot_aa)
    _, sigma = prep_sparse.call_graph()
    cost += sigma[TGate()]
    prep_sparse_adj = attrs.evolve(
        prep_sparse, is_adjoint=True, qroam_block_size=2 ** QI(num_non_zero)[0]
    )
    _, sigma = prep_sparse_adj.call_graph()
    cost += sigma[TGate()]
    unused_lambda = 10
    unused_de = 1e-3
    unused_stps = 10
    logd = (num_non_zero - 1).bit_length()
    refl_cost = 4 * (num_bits_state_prep + logd + 4)  # page 40 listing 4
    # Correct the swap cost:
    # 1. For prepare swaps are costed as Toffolis which we convert to 4 T gates, but a swap costs 7 T gates.
    # 2. The reference inverts the swaps at zero cost for Prep^, so we need to add this cost back.
    delta_swap = (
        8 * (7 - 4) * (num_spin_orb // 2 - 1).bit_length()
        + (7 - 4)
        + 8 * 7 * (num_spin_orb // 2 - 1).bit_length()
        + 7
    )
    cost_of = cost_sparse(
        num_spin_orb, unused_lambda, num_non_zero, unused_de, num_bits_state_prep, unused_stps
    )[0]
    # correct the expected cost by using a different uniform superposition algorithm
    # see: https://github.com/quantumlib/Qualtran/issues/611
    eta = power_two(prep_sparse.num_non_zero)
    cost_uni_prep = (
        4
        * 2
        * (3 * (prep_sparse.num_non_zero - 1).bit_length() - 3 * eta + 2 * num_bits_rot_aa - 9)
    )
    prep = PrepareUniformSuperposition(prep_sparse.num_non_zero)
    cost1a_mod = prep.call_graph()[1][TGate()]
    cost1a_mod += prep.adjoint().call_graph()[1][TGate()]
    # correct for SelectSwapQROM vs QROAM
    # https://github.com/quantumlib/Qualtran/issues/574
    our_qrom = get_sel_swap_qrom_t_count(prep_sparse)
    our_qrom += get_sel_swap_qrom_t_count(prep_sparse_adj)
    paper_qrom = qrom_cost(prep_sparse)
    paper_qrom += qrom_cost(prep_sparse_adj)
    delta_qrom = our_qrom - paper_qrom * 4
    # inequality test difference
    # https://github.com/quantumlib/Qualtran/issues/235
    lte = LessThanEqual(prep_sparse.num_bits_state_prep, prep_sparse.num_bits_state_prep)
    t_count_lte = 2 * lte.call_graph()[1][TGate()]
    t_count_lte_paper = 4 * prep_sparse.num_bits_state_prep  # inverted at zero cost
    delta_ineq = t_count_lte - t_count_lte_paper  # 4 * (prep_sparse.num_bits_state_prep + 1)
    adjusted_cost_qualtran = (
        cost - cost1a_mod + cost_uni_prep + refl_cost - delta_swap - delta_qrom - delta_ineq
    ) // 4
    assert adjusted_cost_qualtran == cost_of


@pytest.mark.notebook
def test_notebook():
    execute_notebook("sparse")
