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
from attrs import evolve

import qualtran.testing as qlt_testing
from qualtran.resource_counting import big_O, GateCounts, get_cost_value, QECGatesCost
from qualtran.symbolics import ceil, log2

from .kikuchi_adjacency_matrix import _kikuchi_matrix_entry, _kikuchi_matrix_entry_symb


@pytest.mark.parametrize("bloq_ex", [_kikuchi_matrix_entry, _kikuchi_matrix_entry_symb])
def test_examples(bloq_autotester, bloq_ex):
    if bloq_autotester.check_name == 'serialize':
        pytest.skip()

    bloq_autotester(bloq_ex)


def test_controlled_cost():
    bloq = _kikuchi_matrix_entry()
    _, sigma = bloq.call_graph(max_depth=2)
    _, ctrl_sigma = bloq.controlled().call_graph(max_depth=2)

    # should only differ in QROM call for loading absolute amplitudes
    a_minus_b = set(sigma.items()) - set(ctrl_sigma.items())
    b_minus_a = set(ctrl_sigma.items()) - set(sigma.items())
    assert len(a_minus_b) == 1
    assert len(b_minus_a) == 1

    ((qrom, na),) = a_minus_b
    ((ctrl_qrom, nb),) = b_minus_a
    assert na == nb
    assert evolve(qrom, num_controls=1) == ctrl_qrom  # type: ignore


def test_cost():
    bloq = _kikuchi_matrix_entry()

    gc = get_cost_value(bloq, QECGatesCost())
    assert gc == GateCounts(
        cswap=512, and_bloq=1301, clifford=12518, measurement=1301, rotation=ANY
    )


def test_cost_symb():
    bloq = _kikuchi_matrix_entry_symb()
    n, m, k, c = sympy.symbols("n m k c", positive=True, integer=True)

    l = c * k
    logl = ceil(log2(l))
    logn = ceil(log2(n))
    logm = ceil(log2(m))

    gc = get_cost_value(bloq, QECGatesCost())
    assert gc == GateCounts(
        cswap=4 * l * (logl + 1) * logn,
        and_bloq=(
            4 * l * ((2 * logn + 1) * (logl + 1))
            + 4 * l
            + 2 * m * (k * logn - 1)
            + 2 * m
            + 4 * ((2 * l - 1) * (logn - 1))
            + logm
            + 4 * ceil(log2(2 * l))
            - 10
        ),
        rotation=ANY,
        clifford=ANY,
        measurement=ANY,
    )

    assert big_O(gc.total_t_count()) == big_O(l * logn * logl + k * m * logn)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('kikuchi_adjacency_matrix')
