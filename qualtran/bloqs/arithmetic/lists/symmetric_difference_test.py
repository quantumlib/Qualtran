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

import qualtran.testing as qlt_testing
from qualtran.bloqs.arithmetic.lists.symmetric_difference import (
    _symm_diff,
    _symm_diff_equal_size_symb,
    _symm_diff_symb,
)
from qualtran.resource_counting import big_O, GateCounts, get_cost_value, QECGatesCost
from qualtran.symbolics import ceil, log2


@pytest.mark.parametrize("bloq_ex", [_symm_diff, _symm_diff_symb, _symm_diff_equal_size_symb])
def test_examples(bloq_autotester, bloq_ex):
    bloq_autotester(bloq_ex)


@pytest.mark.parametrize("bloq_ex", [_symm_diff_symb, _symm_diff_equal_size_symb])
def test_cost(bloq_ex):
    bloq = bloq_ex()
    gc = get_cost_value(bloq, QECGatesCost())

    l, r = bloq.n_lhs, bloq.n_rhs  # assumption l >= r
    logn = bloq.dtype.num_qubits
    logl = ceil(log2(l))
    assert gc == GateCounts(
        cswap=2 * l * logn * (logl + 1),
        and_bloq=(
            2 * l * (2 * logn + 1) * (logl + 1)
            + l
            + r
            + 2 * ((logn - 1) * (l + r - 1))
            + 2 * ceil(log2(l + r))
            - 4
        ),
        clifford=ANY,
        measurement=ANY,
    )

    # \tilde{O}(l log n)
    # Page 38, Thm 4.17, proof para 3, 3rd last line.
    assert gc.total_t_count() in big_O(l * logn * logl**2)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('lists')
