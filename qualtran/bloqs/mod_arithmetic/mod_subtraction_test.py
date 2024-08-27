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

import itertools

import numpy as np
import pytest
import sympy

import qualtran.testing as qlt_testing
from qualtran import QMontgomeryUInt, QUInt
from qualtran.bloqs.mod_arithmetic.mod_subtraction import (
    _cmod_neg,
    _mod_neg,
    CModNeg,
    CModSub,
    ModNeg,
    ModSub,
)
from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join


@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(
    ['prime', 'bitsize'], [(p, n) for p in (13, 17, 23) for n in range(p.bit_length(), 10)]
)
def test_valid_modneg_decomposition(dtype, bitsize, prime):
    b = ModNeg(dtype(bitsize), prime)
    qlt_testing.assert_valid_bloq_decomposition(b)
    qlt_testing.assert_equivalent_bloq_counts(b, [ignore_split_join, ignore_alloc_free])


@pytest.mark.parametrize('cv', range(2))
@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(
    ['prime', 'bitsize'], [(p, n) for p in (13, 17, 23) for n in range(p.bit_length(), 10)]
)
def test_valid_cmodneg_decomposition(dtype, bitsize, prime, cv):
    b = CModNeg(dtype(bitsize), prime, cv)
    qlt_testing.assert_valid_bloq_decomposition(b)
    qlt_testing.assert_equivalent_bloq_counts(b, [ignore_split_join, ignore_alloc_free])


@pytest.mark.slow
@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(
    ['prime', 'bitsize'], [(p, n) for p in (13, 17, 23) for n in range(p.bit_length(), 10)]
)
def test_modneg_classical_action(dtype, bitsize, prime):
    b = ModNeg(dtype(bitsize), prime)
    cb = b.decompose_bloq()
    for x in range(prime):
        assert b.call_classically(x=x) == cb.call_classically(x=x) == ((-x) % prime,)


@pytest.mark.slow
@pytest.mark.parametrize('cv', range(2))
@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(
    ['prime', 'bitsize'], [(p, n) for p in (13, 17, 23) for n in range(p.bit_length(), 10)]
)
def test_cmodneg_classical_action(dtype, bitsize, prime, cv):
    b = CModNeg(dtype(bitsize), prime, cv)
    cb = b.decompose_bloq()
    for control in range(2):
        for x in range(prime):
            assert b.call_classically(ctrl=control, x=x) == cb.call_classically(ctrl=control, x=x)


def test_modneg_classical_action_action_fast():
    prime = 11
    b = ModNeg(QMontgomeryUInt(4), prime)
    cb = b.decompose_bloq()
    for x in range(prime):
        assert b.call_classically(x=x) == cb.call_classically(x=x) == ((-x) % prime,)


@pytest.mark.parametrize('cv', range(2))
def test_cmodneg_classical_action_fast(cv):
    prime = 11
    b = CModNeg(QMontgomeryUInt(4), prime, cv)
    cb = b.decompose_bloq()
    for control in range(2):
        for x in range(prime):
            assert b.call_classically(ctrl=control, x=x) == cb.call_classically(ctrl=control, x=x)


def test_modneg_cost():
    n, p = sympy.symbols('n p')
    b = ModNeg(QMontgomeryUInt(n), p)
    counts = get_cost_value(b, QECGatesCost()).total_t_and_ccz_count()
    # Litinski 2023 https://arxiv.org/abs/2306.08585
    # The construction in Figure 6b, has toffoli count of 3n which is what we use here.
    # Figure/Table 8. Lists n-qubit modular negation as 2n toffoli because it assumes the last $n$
    # toffolis are replaced by measurement based uncomputation. We don't use this optimization since
    # it introduces random phase flips.
    assert counts['n_t'] == 0, 'all toffoli'
    assert counts['n_ccz'] == 3 * (n - 1)


def test_cmodneg_cost():
    n, p = sympy.symbols('n p')
    for cv in range(2):
        b = CModNeg(QMontgomeryUInt(n), p, cv)
        counts = get_cost_value(b, QECGatesCost()).total_t_and_ccz_count()

    # Litinski 2023 https://arxiv.org/abs/2306.08585
    # Figure/Table 8. Lists n-qubit controlled modular negation as 3n toffoli.
    #   Note: While this bloq has the same toffoli count it uses a different decomposition.
    assert counts['n_t'] == 0, 'all toffoli'
    assert counts['n_ccz'] == 3 * (n - 1) + 1


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('mod_subtraction')


@pytest.mark.parametrize('example', [_mod_neg, _cmod_neg])
def test_examples(bloq_autotester, example):
    bloq_autotester(example)


def test_modsub_cost():
    n, p = sympy.symbols('n p')
    b = ModSub(QMontgomeryUInt(n), p)
    counts = get_cost_value(b, QECGatesCost()).total_t_and_ccz_count()

    # Litinski 2023 https://arxiv.org/abs/2306.08585
    # Figure/Table 8. Lists modular subtraction as 6n toffoli.
    assert counts['n_t'] == 0
    assert counts['n_ccz'] == 6 * n - 3


@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(
    ['prime', 'bitsize'], [(p, n) for p in (13, 17, 23) for n in range(p.bit_length(), 10)]
)
def test_modsub_decomposition(dtype, bitsize, prime):
    b = ModSub(dtype(bitsize), prime)
    qlt_testing.assert_valid_bloq_decomposition(b)


@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(
    ['prime', 'bitsize'], [(p, n) for p in (13, 17, 23) for n in range(p.bit_length(), 10)]
)
def test_modsub_bloq_counts(dtype, bitsize, prime):
    b = ModSub(dtype(bitsize), prime)
    qlt_testing.assert_equivalent_bloq_counts(b)


@pytest.mark.slow
@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(
    ['prime', 'bitsize'], [(p, n) for p in (13, 17, 23) for n in range(p.bit_length(), 6)]
)
def test_modsub_classical_action(dtype, bitsize, prime):
    b = ModSub(dtype(bitsize), prime)
    cb = b.decompose_bloq()
    for x, y in itertools.product(range(prime), repeat=2):
        assert b.call_classically(x=x, y=y) == cb.call_classically(x=x, y=y) == (x, (y - x) % prime)


@pytest.mark.slow
@pytest.mark.parametrize('prime', (10**9 + 7, 10**9 + 9))
@pytest.mark.parametrize('bitsize', (32, 33))
def test_modsub_classical_action_large(bitsize, prime):
    b = ModSub(QMontgomeryUInt(bitsize), prime)
    rng = np.random.default_rng(13324)
    qlt_testing.assert_consistent_classical_action(
        b, x=rng.choice(prime, 5).tolist(), y=rng.choice(prime, 5).tolist()
    )


def test_modsub_classical_action_fast():
    bitsize = 10
    prime = 541
    rng = np.random.default_rng(13214)
    b = ModSub(QUInt(bitsize), prime)
    cb = b.decompose_bloq()
    for x, y in rng.choice(prime, (10, 2)):
        assert b.call_classically(x=x, y=y) == cb.call_classically(x=x, y=y) == (x, (y - x) % prime)


def test_cmodsub_cost():
    n, p = sympy.symbols('n p')
    b = CModSub(QMontgomeryUInt(n), p)
    counts = get_cost_value(b, QECGatesCost()).total_t_and_ccz_count()

    # Litinski 2023 https://arxiv.org/abs/2306.08585
    # Figure/Table 8. Lists controlled modular subtraction as 7n toffoli.
    assert counts['n_t'] == 0
    assert counts['n_ccz'] == 7 * n - 1


@pytest.mark.parametrize('cv', range(2))
@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(
    ['prime', 'bitsize'], [(p, n) for p in (13, 17, 23) for n in range(p.bit_length(), 10)]
)
def test_cmodsub_decomposition(cv, dtype, bitsize, prime):
    b = CModSub(dtype(bitsize), prime, cv)
    qlt_testing.assert_valid_bloq_decomposition(b)


@pytest.mark.parametrize('cv', range(2))
@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(
    ['prime', 'bitsize'], [(p, n) for p in (13, 17, 23) for n in range(p.bit_length(), 10)]
)
def test_cmodsub_bloq_counts(cv, dtype, bitsize, prime):
    b = CModSub(dtype(bitsize), prime, cv)
    qlt_testing.assert_equivalent_bloq_counts(b)


@pytest.mark.slow
@pytest.mark.parametrize('cv', range(2))
@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(
    ['prime', 'bitsize'], [(p, n) for p in (13, 17) for n in range(p.bit_length(), 6)]
)
def test_cmodsub_classical_action(cv, dtype, bitsize, prime):
    b = CModSub(dtype(bitsize), prime, cv)
    qlt_testing.assert_consistent_classical_action(b, ctrl=range(2), x=range(prime), y=range(prime))


@pytest.mark.slow
@pytest.mark.parametrize('prime', (10**9 + 7, 10**9 + 9))
@pytest.mark.parametrize('bitsize', (32, 33))
def test_cmodsub_classical_action_large(bitsize, prime):
    b = CModSub(QMontgomeryUInt(bitsize), prime)
    rng = np.random.default_rng(13324)
    qlt_testing.assert_consistent_classical_action(
        b, ctrl=(1,), x=rng.choice(prime, 5).tolist(), y=rng.choice(prime, 5).tolist()
    )


def test_cmodsub_classical_action_fast():
    bitsize = 10
    prime = 541
    rng = np.random.default_rng(13214)
    b = CModSub(QUInt(bitsize), prime)
    cb = b.decompose_bloq()
    for x, y in rng.choice(prime, (10, 2)):
        assert (
            b.call_classically(ctrl=1, x=x, y=y)
            == cb.call_classically(ctrl=1, x=x, y=y)
            == (1, x, (y - x) % prime)
        )
