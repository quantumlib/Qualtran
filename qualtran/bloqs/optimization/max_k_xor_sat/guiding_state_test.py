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
from qualtran.bloqs.max_k_xor_sat.guiding_state import (
    _guiding_state,
    _guiding_state_symb,
    _guiding_state_symb_c,
    _simple_guiding_state,
    _simple_guiding_state_symb,
)
from qualtran.resource_counting import big_O, GateCounts, get_cost_value, QECGatesCost
from qualtran.symbolics import bit_length, ceil, log2


@pytest.mark.parametrize(
    "bloq_ex",
    [
        _simple_guiding_state,
        _simple_guiding_state_symb,
        _guiding_state,
        _guiding_state_symb,
        _guiding_state_symb_c,
    ],
    ids=lambda b: b.name,
)
def test_examples(bloq_autotester, bloq_ex):
    if bloq_autotester.check_name == 'serialize':
        pytest.skip()

    bloq_autotester(bloq_ex)


def test_t_cost_simple():
    bloq = _simple_guiding_state()
    gc = get_cost_value(bloq, QECGatesCost())
    B_GRAD = bloq.phasegrad_bitsize

    assert gc == GateCounts(and_bloq=24, toffoli=3 * (B_GRAD - 2), clifford=ANY, measurement=ANY)


def test_t_cost_simple_symb():
    bloq = _simple_guiding_state_symb()
    gc = get_cost_value(bloq, QECGatesCost())
    B_GRAD = bloq.phasegrad_bitsize

    n, m, k = bloq.inst.n, bloq.inst.m, bloq.inst.k
    klogn = k * ceil(log2(n))
    # https://github.com/quantumlib/Qualtran/issues/1341
    klogn_roundtrip = bit_length(2**klogn - 1)

    assert gc == GateCounts(
        # O(k m log n)
        and_bloq=4 * m + (2 * m + 1) * (klogn_roundtrip - 1) - 4,
        toffoli=2 * (B_GRAD - 2),
        clifford=ANY,
        measurement=ANY,
    )


def test_t_cost():
    bloq = _guiding_state()
    gc = get_cost_value(bloq, QECGatesCost())
    B_GRAD = bloq.simple_guiding_state.phasegrad_bitsize

    assert gc == GateCounts(
        and_bloq=352, toffoli=6 * (B_GRAD - 2), cswap=192, clifford=ANY, measurement=ANY
    )


@pytest.mark.parametrize("bloq_ex", [_guiding_state_symb, _guiding_state_symb_c])
def test_t_cost_symb_c(bloq_ex):
    bloq = bloq_ex()
    gc = get_cost_value(bloq, QECGatesCost())
    B_GRAD = bloq.simple_guiding_state.phasegrad_bitsize

    n, m, k = bloq.inst.n, bloq.inst.m, bloq.inst.k
    l, c = bloq.ell, bloq.c

    logn = ceil(log2(n))
    logl = ceil(log2(l))

    klogn = k * logn
    # https://github.com/quantumlib/Qualtran/issues/1341
    klogn_roundtrip = bit_length(2**klogn - 1)

    assert gc == GateCounts(
        and_bloq=(
            6 * l**2 * (2 * logn + 1)
            + l * logl
            + l
            + c * (4 * m + (2 * m + 1) * (klogn_roundtrip - 1) - 4)
            + (l - 1) * logn
            - 2
        ),
        toffoli=c * (2 * (B_GRAD - 2)),
        cswap=6 * l**2 * logn,
        clifford=ANY,
        measurement=ANY,
    )

    # verify big_O
    t_cost = gc.total_t_count()
    t_cost = sympy.sympify(t_cost)
    t_cost = t_cost.subs(klogn_roundtrip, klogn)
    t_cost = t_cost.simplify()
    assert t_cost in big_O(l * m * logn + l**2 * logn + B_GRAD * c)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('guiding_state')


@pytest.mark.notebook
def test_tutorial():
    qlt_testing.execute_notebook('guiding_state_tutorial')
