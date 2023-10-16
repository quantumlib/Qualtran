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
from openfermion.resource_estimates.utils import QI

from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.chemistry.sparse import PrepareSparse, SelectSparse
from qualtran.resource_counting import get_bloq_counts_graph
from qualtran.testing import execute_notebook


def _make_sparse_select():
    from qualtran.bloqs.chemistry.sparse import SelectSparse

    return SelectSparse(10)


def _make_sparse_prepare():
    from qualtran.bloqs.chemistry.sparse import PrepareSparse

    return PrepareSparse(10, 1_000, 8)


def test_sparse_prepare():
    prep = PrepareSparse(10, 1_000, 8)


def test_sparse_prepare_bloq_counts():
    _, sigma = get_bloq_counts_graph(PrepareSparse(10, 1_000, 8))


def test_sparse_select_bloq_counts():
    _, sigma = get_bloq_counts_graph(SelectSparse(10))


@pytest.mark.parametrize(
    "num_spin_orb, num_non_zero, num_bits_rot_aa",
    ((30, 1_000, 4), (129, 10_000, 3), (213, 100_000, 3)),
)
def test_sparse_costs_against_openfermion(num_spin_orb, num_non_zero, num_bits_rot_aa):
    num_bits_state_prep = 12
    cost = 0
    _, sigma = get_bloq_counts_graph(SelectSparse(num_spin_orb))
    cost += sigma[TGate()]
    _, sigma = get_bloq_counts_graph(
        PrepareSparse(
            num_spin_orb,
            num_non_zero,
            num_bits_rot_aa=num_bits_rot_aa,
            num_bits_state_prep=num_bits_state_prep,
            k=32,  # harcoded in openfermion
        )
    )
    cost += sigma[TGate()]
    _, sigma = get_bloq_counts_graph(
        PrepareSparse(
            num_spin_orb,
            num_non_zero,
            num_bits_rot_aa=num_bits_rot_aa,
            num_bits_state_prep=num_bits_state_prep,
            adjoint=True,
            k=2 ** QI(num_non_zero)[0],  # determined from QI in openfermion
        )
    )
    cost += sigma[TGate()]
    unused_lambda = 10
    unused_de = 1e-3
    unused_stps = 10
    logd = (num_non_zero - 1).bit_length()
    refl_cost = 4 * (num_bits_state_prep + logd + 4)  # page 40 listing 4
    cost_of = cost_sparse(
        num_spin_orb, unused_lambda, num_non_zero, unused_de, num_bits_state_prep, unused_stps
    )[0]
    adjusted_cost_qualtran = (cost + refl_cost) // 4
    assert adjusted_cost_qualtran == cost_of


def test_sparse_select():
    sel = SelectSparse(10)


def test_notebook():
    execute_notebook("sparse")
