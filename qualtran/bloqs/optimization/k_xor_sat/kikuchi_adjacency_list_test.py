#  Copyright 2024 Google LLC
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
from unittest.mock import ANY

import pytest
import sympy

import qualtran.testing as qlt_testing
from qualtran.resource_counting import big_O, GateCounts, get_cost_value, QECGatesCost
from qualtran.symbolics import ceil, log2

from .kikuchi_adjacency_list import (
    _col_kth_nz,
    _col_kth_nz_symb,
    _kikuchi_nonzero_index,
    _kikuchi_nonzero_index_symb,
)


@pytest.mark.parametrize(
    "bloq_ex",
    [_col_kth_nz, _col_kth_nz_symb, _kikuchi_nonzero_index, _kikuchi_nonzero_index_symb],
    ids=lambda bloq_ex: bloq_ex.name,
)
def test_examples(bloq_autotester, bloq_ex):
    bloq_autotester(bloq_ex)


def test_cost_col_kth_nz():
    n, m, k, c, s = sympy.symbols("n m k c s", positive=True, integer=True)
    l = c * k
    logn = ceil(log2(n))
    logl = ceil(log2(l))

    bloq = _col_kth_nz_symb()
    cost = get_cost_value(bloq, QECGatesCost())
    assert cost == GateCounts(
        toffoli=(m + 1) * logn,
        cswap=4 * l * m * (logl + 1) * logn,
        and_bloq=(
            4 * m * (logn - 1)
            + (
                2
                * m
                * (
                    2 * l * ((2 * logn + 1) * (logl + 1))
                    + l
                    + k
                    + 2 * ((logn - 1) * (l + k - 1))
                    + 2 * ceil(log2(l + k))
                    - 4
                )
            )
            + m
        ),
        clifford=ANY,
        measurement=ANY,
    )
    assert big_O(cost.total_t_count()) == big_O(l * m * logn * logl)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('kikuchi_adjacency_list')
