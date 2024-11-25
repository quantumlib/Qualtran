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

import numpy as np
import pytest
import sympy

import qualtran.testing as qlt_testing
from qualtran import QInt, QMontgomeryUInt, QUInt
from qualtran.bloqs.arithmetic.controlled_addition import _cadd_large, _cadd_small, CAdd
from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join


def test_examples(bloq_autotester):
    bloq_autotester(_cadd_small)
    bloq_autotester(_cadd_large)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('controlled_addition')


@pytest.mark.parametrize('control', range(2))
@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(['a_bits', 'b_bits'], [(a, b) for a in range(1, 5) for b in range(a, 5)])
def test_decomposition(control, dtype, a_bits, b_bits):
    b = CAdd(dtype(a_bits), dtype(b_bits), cv=control)
    qlt_testing.assert_valid_bloq_decomposition(b)
    qlt_testing.assert_equivalent_bloq_counts(b, [ignore_alloc_free, ignore_split_join])


@pytest.mark.parametrize("n", [*range(3, 10)])
def test_addition_gate_counts_controlled(n: int):
    add = CAdd(QUInt(n), cv=1)
    num_and = 2 * n - 1
    t_count = 4 * num_and
    assert add.bloq_counts() == add.decompose_bloq().bloq_counts(generalizer=ignore_split_join)
    assert add.t_complexity().t == t_count


@pytest.mark.slow
@pytest.mark.parametrize('control', range(2))
@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(['a_bits', 'b_bits'], [(a, b) for a in range(1, 5) for b in range(a, 5)])
def test_classical_action_unsigned(control, dtype, a_bits, b_bits):
    b = CAdd(dtype(a_bits), dtype(b_bits), cv=control)
    cb = b.decompose_bloq()
    for c in range(2):
        for x in range(2**a_bits):
            for y in range(2**b_bits):
                assert b.call_classically(ctrl=c, a=x, b=y) == cb.call_classically(
                    ctrl=c, a=x, b=y
                ), f'{c=} {x=} {y=}'


@pytest.mark.parametrize('control', range(2))
@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(['a_bits', 'b_bits'], [(a, b) for a in range(1, 5) for b in range(a, 5)])
def test_classical_action_unsigned_fast(control, dtype, a_bits, b_bits):
    b = CAdd(dtype(a_bits), dtype(b_bits), cv=control)
    cb = b.decompose_bloq()
    rng = np.random.default_rng(13432)
    for c in range(2):
        for x, y in zip(rng.choice(2**a_bits, 10), rng.choice(2**b_bits, 10)):
            assert b.call_classically(ctrl=c, a=x, b=y) == cb.call_classically(
                ctrl=c, a=x, b=y
            ), f'{c=} {x=} {y=}'


@pytest.mark.slow
@pytest.mark.parametrize('control', range(2))
@pytest.mark.parametrize(['a_bits', 'b_bits'], [(a, b) for a in range(2, 5) for b in range(a, 5)])
def test_classical_action_signed(control, a_bits, b_bits):
    b = CAdd(QInt(a_bits), QInt(b_bits), cv=control)
    cb = b.decompose_bloq()
    for c in range(2):
        for x in range(-(2 ** (a_bits - 1)), 2 ** (a_bits - 1)):
            for y in range(-(2 ** (b_bits - 1)), 2 ** (b_bits - 1)):
                assert b.call_classically(ctrl=c, a=x, b=y) == cb.call_classically(
                    ctrl=c, a=x, b=y
                ), f'{c=} {x=} {y=}'


@pytest.mark.parametrize('control', range(2))
@pytest.mark.parametrize(['a_bits', 'b_bits'], [(a, b) for a in range(2, 5) for b in range(a, 5)])
def test_classical_action_signed_fast(control, a_bits, b_bits):
    b = CAdd(QInt(a_bits), QInt(b_bits), cv=control)
    cb = b.decompose_bloq()
    rng = np.random.default_rng(13432)
    for c in range(2):
        for x, y in zip(rng.choice(2**a_bits, 10), rng.choice(2**b_bits, 10)):
            x -= 2 ** (a_bits - 1)
            y -= 2 ** (b_bits - 1)
            assert b.call_classically(ctrl=c, a=x, b=y) == cb.call_classically(
                ctrl=c, a=x, b=y
            ), f'{c=} {x=} {y=}'


@pytest.mark.parametrize('control', range(2))
@pytest.mark.parametrize('dtype', [QInt, QUInt, QMontgomeryUInt])
def test_symbolic_cost(control, dtype):
    n, m = sympy.symbols('n m')
    b = CAdd(dtype(n), dtype(m), control)
    cost = get_cost_value(b, QECGatesCost()).total_t_and_ccz_count()
    assert cost['n_t'] == 0
    assert cost['n_ccz'] == m + n - 1


@pytest.mark.parametrize('control', range(2))
@pytest.mark.parametrize('dtype', [QInt, QUInt, QMontgomeryUInt])
def test_consistent_tcomplexity(control, dtype):
    n, m = sympy.symbols('n m')
    b = CAdd(dtype(n), dtype(m), control)
    cost = get_cost_value(b, QECGatesCost()).total_t_and_ccz_count()
    assert cost['n_t'] == 0
    assert b.t_complexity().t == 4 * cost['n_ccz']
