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

import pytest
import sympy

from qualtran.bloqs.mod_arithmetic.mod_subtraction import ModNeg, CModNeg, _cmod_neg, _mod_neg
import qualtran.testing as qlt_testing
from qualtran import QUInt, QMontgomeryUInt
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join
from qualtran.resource_counting import query_costs, QECGatesCost, GateCounts

@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(['bitsize', 'prime'], [
    (p, n) for p in (13, 17, 23) for n in range(p.bit_length(), 10)
])
def test_valid_modneg_decomposition(dtype, bitsize, prime):
    b = ModNeg(dtype(bitsize), prime)
    qlt_testing.assert_valid_bloq_decomposition(b)
    qlt_testing.assert_equivalent_bloq_counts(b, [ignore_split_join, ignore_alloc_free])

@pytest.mark.parametrize('cv', range(2))
@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(['bitsize', 'prime'], [
    (p, n) for p in (13, 17, 23) for n in range(p.bit_length(), 10)
])
def test_valid_cmodneg_decomposition(dtype, bitsize, prime, cv):
    b = CModNeg(dtype(bitsize), prime, cv)
    qlt_testing.assert_valid_bloq_decomposition(b)
    qlt_testing.assert_equivalent_bloq_counts(b, [ignore_split_join, ignore_alloc_free])


@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(['bitsize', 'prime'], [
    (p, n) for p in (13, 17, 23) for n in range(p.bit_length(), 10)
])
def test_modneg_classical_action(dtype, bitsize, prime):
    b = ModNeg(dtype(bitsize), prime)
    cb = b.decompose_bloq()
    for x in range(prime):
        assert b.call_classically(x=x) == cb.call_classically(x=x) == ((-x)%prime,)

@pytest.mark.parametrize('cv', range(2))
@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(['bitsize', 'prime'], [
    (p, n) for p in (13, 17, 23) for n in range(p.bit_length(), 10)
])
def test_cmodneg_classical_action(dtype, bitsize, prime, cv):
    b = CModNeg(dtype(bitsize), prime, cv)
    cb = b.decompose_bloq()
    for control in range(2):
        for x in range(prime):
            assert b.call_classically(ctrl=control, x=x) == cb.call_classically(ctrl=control, x=x)


def test_modneg_cost():
    n, p = sympy.symbols('n p')
    b = ModNeg(QMontgomeryUInt(n), p)
    target_cost = QECGatesCost()
    cost:GateCounts = query_costs(b, [target_cost])[b][target_cost]
    num_toffoli = 3*(n-1)
    assert cost.total_t_count() == 4*num_toffoli

def test_cmodneg_cost():
    n, p = sympy.symbols('n p')
    for cv in range(2):
        b = CModNeg(QMontgomeryUInt(n), p, cv)
        target_cost = QECGatesCost()
        cost:GateCounts = query_costs(b, [target_cost])[b][target_cost]
        num_toffoli = 3*(n-1) + 1
        assert cost.total_t_count() == 4*num_toffoli


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('mod_subtraction')
    
@pytest.mark.parametrize('example', [_mod_neg, _cmod_neg])
def test_examples(bloq_autotester, example):
    bloq_autotester(example)
