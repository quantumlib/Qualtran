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

import attrs
import numpy as np
import pytest
import sympy

import qualtran.testing as qlt_testing
from qualtran import QMontgomeryUInt, QUInt
from qualtran.bloqs.mod_arithmetic.mod_addition import CtrlScaleModAdd
from qualtran.bloqs.mod_arithmetic.mod_multiplication import (
    _dirtyoutofplacemontgomerymodmul_medium,
    _dirtyoutofplacemontgomerymodmul_small,
    _moddbl_large,
    _moddbl_small,
    _modmul,
    _modmul_symb,
    CModMulK,
    DirtyOutOfPlaceMontgomeryModMul,
    ModDbl,
    SingleWindowModMul,
)
from qualtran.resource_counting import get_cost_value, QECGatesCost, SympySymbolAllocator
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join


@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(
    ['prime', 'bitsize'], [(p, n) for p in (13, 17, 23) for n in range(p.bit_length(), 10)]
)
def test_moddbl_classical_action(dtype, bitsize, prime):
    b = ModDbl(dtype(bitsize), mod=prime)
    qlt_testing.assert_consistent_classical_action(b, x=range(prime))


@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(
    ['prime', 'bitsize'], [(p, n) for p in (13, 17, 23) for n in range(p.bit_length(), 10)]
)
def test_moddbl_decomposition(dtype, bitsize, prime):
    b = ModDbl(dtype(bitsize), prime)
    qlt_testing.assert_valid_bloq_decomposition(b)


@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(
    ['prime', 'bitsize'], [(p, n) for p in (13, 17, 23) for n in range(p.bit_length(), 10)]
)
def test_moddbl_bloq_counts(dtype, bitsize, prime):
    b = ModDbl(dtype(bitsize), prime)
    qlt_testing.assert_equivalent_bloq_counts(b, [ignore_alloc_free, ignore_split_join])


def test_moddbl_cost():
    n, p = sympy.symbols('n p')
    b = ModDbl(QMontgomeryUInt(n), p)
    cost = get_cost_value(b, QECGatesCost()).total_t_and_ccz_count()

    # Litinski 2023 https://arxiv.org/abs/2306.08585
    # Figure/Table 8. Lists modular doubling as 2n toffoli.
    assert cost['n_t'] == 0
    assert cost['n_ccz'] == 2 * n + 1


@pytest.mark.slow
@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(
    ['prime', 'bitsize', 'k'],
    [(p, n, k) for p in (13, 17, 23) for n in range(p.bit_length(), 10) for k in range(1, p)],
)
def test_cmodmulk_classical_action(dtype, bitsize, prime, k):
    b = CModMulK(dtype(bitsize), k=k, mod=prime)
    qlt_testing.assert_consistent_classical_action(b, ctrl=(0, 1), x=range(prime))


@pytest.mark.parametrize('k', range(1, 13))
def test_cmodmulk_classical_action_fast(k):
    b = CModMulK(QMontgomeryUInt(4), k=k, mod=13)
    qlt_testing.assert_consistent_classical_action(b, ctrl=(0, 1), x=range(13))


@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(
    ['prime', 'bitsize', 'k'],
    [(p, n, k) for p in (13, 17, 23) for n in range(p.bit_length(), 10) for k in range(1, p)],
)
def test_cmodmulk_decomposition(dtype, bitsize, prime, k):
    b = CModMulK(dtype(bitsize), k, prime)
    qlt_testing.assert_valid_bloq_decomposition(b)


@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize(
    ['prime', 'bitsize', 'k'],
    [(p, n, k) for p in (13, 17, 23) for n in range(p.bit_length(), 10) for k in range(1, p)],
)
def test_cmodmulk_bloq_counts(dtype, bitsize, prime, k):
    b = CModMulK(dtype(bitsize), k, prime)
    ssa = SympySymbolAllocator()
    my_k = ssa.new_symbol('k')

    def generalizer(bloq):
        if isinstance(bloq, CtrlScaleModAdd):
            return attrs.evolve(bloq, k=my_k)
        return bloq

    qlt_testing.assert_equivalent_bloq_counts(
        b, [ignore_alloc_free, ignore_split_join, generalizer]
    )


def test_examples_moddbl_small(bloq_autotester):
    bloq_autotester(_moddbl_small)


def test_examples_moddbl_large(bloq_autotester):
    bloq_autotester(_moddbl_large)


def test_examples_modmul_symb(bloq_autotester):
    bloq_autotester(_modmul_symb)


def test_examples_modmul(bloq_autotester):
    bloq_autotester(_modmul)


def test_examples_dirtyoutofplacemontgomerymodmul_small(bloq_autotester):
    bloq_autotester(_dirtyoutofplacemontgomerymodmul_small)


def test_examples_dirtyoutofplacemontgomerymodmul_medium(bloq_autotester):
    bloq_autotester(_dirtyoutofplacemontgomerymodmul_medium)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('mod_multiplication')


@pytest.mark.parametrize('p', (7, 9, 11))
@pytest.mark.parametrize('uncompute', [True, False])
@pytest.mark.parametrize(
    ['n', 'm'], [(n, m) for n in range(6, 10) for m in range(1, n + 1) if n % m == 0]
)
def test_dirtyoutofplacemontgomerymodmul_decomposition(n, m, p, uncompute):
    b = DirtyOutOfPlaceMontgomeryModMul(n, m, p, uncompute)
    qlt_testing.assert_valid_bloq_decomposition(b)


@pytest.mark.parametrize('p', (7, 9, 11))
@pytest.mark.parametrize('uncompute', [True, False])
@pytest.mark.parametrize(
    ['n', 'm'], [(n, m) for n in range(6, 10) for m in range(1, n + 1) if n % m == 0]
)
def test_dirtyoutofplacemontgomerymodmul_bloq_counts(n, m, p, uncompute):
    b = DirtyOutOfPlaceMontgomeryModMul(n, m, p, uncompute)
    qlt_testing.assert_equivalent_bloq_counts(b, [ignore_alloc_free, ignore_split_join])


@pytest.mark.parametrize('uncompute', [True, False])
def test_dirtyoutofplacemontgomerymodmul_symbolic_cost(uncompute):
    n, m, p = sympy.symbols('n m p', integer=True)

    # In Litinski 2023 https://arxiv.org/abs/2306.08585 a window size of 4 is used.
    # The cost function generally has floor/ceil division that disappear for bitsize=0 mod 4.
    # This is why instead of using bitsize=n directly, we use bitsize=4*m=n.
    b = DirtyOutOfPlaceMontgomeryModMul(4 * m, 4, p, uncompute)
    cost = get_cost_value(b, QECGatesCost()).total_t_and_ccz_count()
    assert cost['n_t'] == 0

    # Litinski 2023 https://arxiv.org/abs/2306.08585
    # Figure/Table 8. Lists modular multiplication as 2.25n^2+9n toffoli.
    # The following formula is 2.25n^2+7.25n-1 written with rationals because sympy comparison fails with floats.
    assert isinstance(cost['n_ccz'], sympy.Expr)
    assert (
        cost['n_ccz'].subs(m, n / 4).expand()
        == sympy.Rational(9, 4) * n**2 + sympy.Rational(29, 4) * n - 1
    )


@pytest.mark.parametrize('p', (3, 5, 7))
@pytest.mark.parametrize(
    ['n', 'm'], [(n, m) for n in range(5, 8) for m in range(1, n + 1) if n % m == 0]
)
def test_dirtyoutofplacemontgomerymodmul_classical_action(n, m, p):
    b = DirtyOutOfPlaceMontgomeryModMul(n, m, p, False)
    qlt_testing.assert_consistent_classical_action(b, x=range(1, p), y=range(1, p))


@pytest.mark.parametrize('p', (3, 5, 7))
@pytest.mark.parametrize(
    ['n', 'm'], [(n, m) for n in range(5, 8) for m in range(1, n + 1) if n % m == 0]
)
def test_singlewindowmodmul_decomposition(n, m, p):
    b = SingleWindowModMul(window_size=m, bitsize=n, mod=p)
    qlt_testing.assert_valid_bloq_decomposition(b)


@pytest.mark.parametrize('p', (3, 5, 7))
@pytest.mark.parametrize(
    ['n', 'm'], [(n, m) for n in range(5, 8) for m in range(1, n + 1) if n % m == 0]
)
def test_singlewindowmodmul_bloq_counts(n, m, p):
    b = SingleWindowModMul(window_size=m, bitsize=n, mod=p)
    qlt_testing.assert_equivalent_bloq_counts(b, [ignore_alloc_free, ignore_split_join])


@pytest.mark.slow
@pytest.mark.parametrize('p', (3, 5, 7))
@pytest.mark.parametrize(
    ['n', 'm'], [(n, m) for n in range(5, 8) for m in range(1, n + 1) if n % m == 0]
)
def test_singlewindowmodmul_classical_action(n, m, p):
    b = SingleWindowModMul(window_size=m, bitsize=n, mod=p)
    cb = b.decompose_bloq()
    for x in range(1, min(p, 2**m)):
        for y in range(1, p):
            for target in range(2**n):
                bloq_res = b.call_classically(
                    x=np.array(QUInt(m).to_bits(x)),
                    y=y,
                    target=np.array(QUInt(n + m).to_bits(target)),
                    qrom_index=0,
                )
                decomposed_res = cb.call_classically(
                    x=np.array(QUInt(m).to_bits(x)),
                    y=y,
                    target=np.array(QUInt(n + m).to_bits(target)),
                    qrom_index=0,
                )
                np.testing.assert_equal(bloq_res[0], decomposed_res[0])  # x
                assert bloq_res[1] == decomposed_res[1]  # y
                np.testing.assert_equal(bloq_res[2], decomposed_res[2])  # target
                assert bloq_res[3] == decomposed_res[3]  # qrom_index


def test_singlewindowmodmul_classical_action_fast():
    n = 4
    m = 2
    p = 5
    b = SingleWindowModMul(window_size=m, bitsize=n, mod=p)
    cb = b.decompose_bloq()
    for x in range(1, min(p, 2**m)):
        for y in range(1, p):
            for target in range(2**n):
                bloq_res = b.call_classically(
                    x=np.array(QUInt(m).to_bits(x)),
                    y=y,
                    target=np.array(QUInt(n + m).to_bits(target)),
                    qrom_index=0,
                )
                decomposed_res = cb.call_classically(
                    x=np.array(QUInt(m).to_bits(x)),
                    y=y,
                    target=np.array(QUInt(n + m).to_bits(target)),
                    qrom_index=0,
                )
                np.testing.assert_equal(bloq_res[0], decomposed_res[0])  # x
                assert bloq_res[1] == decomposed_res[1]  # y
                np.testing.assert_equal(bloq_res[2], decomposed_res[2])  # target
                assert bloq_res[3] == decomposed_res[3]  # qrom_index
