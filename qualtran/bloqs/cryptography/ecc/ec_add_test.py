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
import sympy

import qualtran.testing as qlt_testing
from qualtran._infra.data_types import QMontgomeryUInt
from qualtran.bloqs.cryptography.ecc.ec_add import (
    _ec_add,
    _ec_add_small,
    _ECAddStepFive,
    _ECAddStepFour,
    _ECAddStepOne,
    _ECAddStepSix,
    _ECAddStepThree,
    _ECAddStepTwo,
    ECAdd,
)
from qualtran.resource_counting._bloq_counts import QECGatesCost
from qualtran.resource_counting._costing import get_cost_value
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join


@pytest.mark.parametrize(
    ['n', 'm'], [(n, m) for n in range(7, 8) for m in range(1, n + 1) if n % m == 0]
)
@pytest.mark.parametrize('a,b', [(15, 13), (2, 10)])
@pytest.mark.parametrize('x,y', [(15, 13), (0, 0)])
def test_ec_add_steps_classical_fast(n, m, a, b, x, y):
    p = 17
    lam_num = (3 * a**2) % p
    lam_denom = (2 * b) % p
    lam_r = 0 if b == 0 else (lam_num * pow(lam_denom, -1, mod=p)) % p

    dtype = QMontgomeryUInt(n, p)
    a = dtype.uint_to_montgomery(a)
    b = dtype.uint_to_montgomery(b)
    x = dtype.uint_to_montgomery(x)
    y = dtype.uint_to_montgomery(y)
    lam_r = dtype.uint_to_montgomery(lam_r) if lam_r != 0 else p

    bloq = _ECAddStepOne(n=n, mod=p)
    ret1 = bloq.call_classically(a=a, b=b, x=x, y=y)
    ret2 = bloq.decompose_bloq().call_classically(a=a, b=b, x=x, y=y)
    assert ret1 == ret2

    step_1 = _ECAddStepOne(n=n, mod=p).on_classical_vals(a=a, b=b, x=x, y=y)
    bloq = _ECAddStepTwo(n=n, mod=p, window_size=m)
    ret1 = bloq.call_classically(
        f1=step_1['f1'], ctrl=step_1['ctrl'], a=a, b=b, x=x, y=y, lam_r=lam_r
    )
    ret2 = bloq.decompose_bloq().call_classically(
        f1=step_1['f1'], ctrl=step_1['ctrl'], a=a, b=b, x=x, y=y, lam_r=lam_r
    )
    assert ret1 == ret2

    step_2 = _ECAddStepTwo(n=n, mod=p, window_size=m).on_classical_vals(
        f1=step_1['f1'], ctrl=step_1['ctrl'], a=a, b=b, x=x, y=y, lam_r=lam_r
    )
    bloq = _ECAddStepThree(n=n, mod=p, window_size=m)
    ret1 = bloq.call_classically(
        ctrl=step_2['ctrl'],
        a=step_2['a'],
        b=step_2['b'],
        x=step_2['x'],
        y=step_2['y'],
        lam=step_2['lam'],
    )
    ret2 = bloq.decompose_bloq().call_classically(
        ctrl=step_2['ctrl'],
        a=step_2['a'],
        b=step_2['b'],
        x=step_2['x'],
        y=step_2['y'],
        lam=step_2['lam'],
    )
    assert ret1 == ret2

    step_3 = _ECAddStepThree(n=n, mod=p, window_size=m).on_classical_vals(
        ctrl=step_2['ctrl'],
        a=step_2['a'],
        b=step_2['b'],
        x=step_2['x'],
        y=step_2['y'],
        lam=step_2['lam'],
    )
    bloq = _ECAddStepFour(n=n, mod=p, window_size=m)
    ret1 = bloq.call_classically(x=step_3['x'], y=step_3['y'], lam=step_3['lam'])
    ret2 = bloq.decompose_bloq().call_classically(x=step_3['x'], y=step_3['y'], lam=step_3['lam'])
    assert ret1 == ret2

    step_4 = _ECAddStepFour(n=n, mod=p, window_size=m).on_classical_vals(
        x=step_3['x'], y=step_3['y'], lam=step_3['lam']
    )
    bloq = _ECAddStepFive(n=n, mod=p, window_size=m)
    ret1 = bloq.call_classically(
        ctrl=step_3['ctrl'],
        a=step_3['a'],
        b=step_3['b'],
        x=step_4['x'],
        y=step_4['y'],
        lam_r=step_2['lam_r'],
        lam=step_4['lam'],
    )
    ret2 = bloq.decompose_bloq().call_classically(
        ctrl=step_3['ctrl'],
        a=step_3['a'],
        b=step_3['b'],
        x=step_4['x'],
        y=step_4['y'],
        lam_r=step_2['lam_r'],
        lam=step_4['lam'],
    )
    assert ret1 == ret2

    step_5 = _ECAddStepFive(n=n, mod=p, window_size=m).on_classical_vals(
        ctrl=step_3['ctrl'],
        a=step_3['a'],
        b=step_3['b'],
        x=step_4['x'],
        y=step_4['y'],
        lam_r=step_2['lam_r'],
        lam=step_4['lam'],
    )
    bloq = _ECAddStepSix(n=n, mod=p)
    ret1 = bloq.call_classically(
        f1=step_2['f1'],
        f2=step_1['f2'],
        f3=step_1['f3'],
        f4=step_1['f4'],
        ctrl=step_5['ctrl'],
        a=step_5['a'],
        b=step_5['b'],
        x=step_5['x'],
        y=step_5['y'],
    )
    ret2 = bloq.decompose_bloq().call_classically(
        f1=step_2['f1'],
        f2=step_1['f2'],
        f3=step_1['f3'],
        f4=step_1['f4'],
        ctrl=step_5['ctrl'],
        a=step_5['a'],
        b=step_5['b'],
        x=step_5['x'],
        y=step_5['y'],
    )
    assert ret1 == ret2


@pytest.mark.slow
@pytest.mark.parametrize(
    ['n', 'm'], [(n, m) for n in range(7, 9) for m in range(1, n + 1) if n % m == 0]
)
@pytest.mark.parametrize(
    'a,b',
    [
        (15, 13),
        (2, 10),
        (8, 3),
        (12, 1),
        (6, 6),
        (5, 8),
        (10, 15),
        (1, 12),
        (3, 0),
        (1, 5),
        (10, 2),
        (0, 0),
    ],
)
@pytest.mark.parametrize('x,y', [(15, 13), (5, 8), (10, 15), (1, 12), (3, 0), (1, 5), (10, 2)])
def test_ec_add_steps_classical(n, m, a, b, x, y):
    p = 17
    lam_num = (3 * a**2) % p
    lam_denom = (2 * b) % p
    lam_r = 0 if b == 0 else (lam_num * pow(lam_denom, -1, mod=p)) % p

    dtype = QMontgomeryUInt(n, p)
    a = dtype.uint_to_montgomery(a)
    b = dtype.uint_to_montgomery(b)
    x = dtype.uint_to_montgomery(x)
    y = dtype.uint_to_montgomery(y)
    lam_r = dtype.uint_to_montgomery(lam_r) if lam_r != 0 else p

    bloq = _ECAddStepOne(n=n, mod=p)
    ret1 = bloq.call_classically(a=a, b=b, x=x, y=y)
    ret2 = bloq.decompose_bloq().call_classically(a=a, b=b, x=x, y=y)
    assert ret1 == ret2

    step_1 = _ECAddStepOne(n=n, mod=p).on_classical_vals(a=a, b=b, x=x, y=y)
    bloq = _ECAddStepTwo(n=n, mod=p, window_size=m)
    ret1 = bloq.call_classically(
        f1=step_1['f1'], ctrl=step_1['ctrl'], a=a, b=b, x=x, y=y, lam_r=lam_r
    )
    ret2 = bloq.decompose_bloq().call_classically(
        f1=step_1['f1'], ctrl=step_1['ctrl'], a=a, b=b, x=x, y=y, lam_r=lam_r
    )
    assert ret1 == ret2

    step_2 = _ECAddStepTwo(n=n, mod=p, window_size=m).on_classical_vals(
        f1=step_1['f1'], ctrl=step_1['ctrl'], a=a, b=b, x=x, y=y, lam_r=lam_r
    )
    bloq = _ECAddStepThree(n=n, mod=p, window_size=m)
    ret1 = bloq.call_classically(
        ctrl=step_2['ctrl'],
        a=step_2['a'],
        b=step_2['b'],
        x=step_2['x'],
        y=step_2['y'],
        lam=step_2['lam'],
    )
    ret2 = bloq.decompose_bloq().call_classically(
        ctrl=step_2['ctrl'],
        a=step_2['a'],
        b=step_2['b'],
        x=step_2['x'],
        y=step_2['y'],
        lam=step_2['lam'],
    )
    assert ret1 == ret2

    step_3 = _ECAddStepThree(n=n, mod=p, window_size=m).on_classical_vals(
        ctrl=step_2['ctrl'],
        a=step_2['a'],
        b=step_2['b'],
        x=step_2['x'],
        y=step_2['y'],
        lam=step_2['lam'],
    )
    bloq = _ECAddStepFour(n=n, mod=p, window_size=m)
    ret1 = bloq.call_classically(x=step_3['x'], y=step_3['y'], lam=step_3['lam'])
    ret2 = bloq.decompose_bloq().call_classically(x=step_3['x'], y=step_3['y'], lam=step_3['lam'])
    assert ret1 == ret2

    step_4 = _ECAddStepFour(n=n, mod=p, window_size=m).on_classical_vals(
        x=step_3['x'], y=step_3['y'], lam=step_3['lam']
    )
    bloq = _ECAddStepFive(n=n, mod=p, window_size=m)
    ret1 = bloq.call_classically(
        ctrl=step_3['ctrl'],
        a=step_3['a'],
        b=step_3['b'],
        x=step_4['x'],
        y=step_4['y'],
        lam_r=step_2['lam_r'],
        lam=step_4['lam'],
    )
    ret2 = bloq.decompose_bloq().call_classically(
        ctrl=step_3['ctrl'],
        a=step_3['a'],
        b=step_3['b'],
        x=step_4['x'],
        y=step_4['y'],
        lam_r=step_2['lam_r'],
        lam=step_4['lam'],
    )
    assert ret1 == ret2

    step_5 = _ECAddStepFive(n=n, mod=p, window_size=m).on_classical_vals(
        ctrl=step_3['ctrl'],
        a=step_3['a'],
        b=step_3['b'],
        x=step_4['x'],
        y=step_4['y'],
        lam_r=step_2['lam_r'],
        lam=step_4['lam'],
    )
    bloq = _ECAddStepSix(n=n, mod=p)
    ret1 = bloq.call_classically(
        f1=step_2['f1'],
        f2=step_1['f2'],
        f3=step_1['f3'],
        f4=step_1['f4'],
        ctrl=step_5['ctrl'],
        a=step_5['a'],
        b=step_5['b'],
        x=step_5['x'],
        y=step_5['y'],
    )
    ret2 = bloq.decompose_bloq().call_classically(
        f1=step_2['f1'],
        f2=step_1['f2'],
        f3=step_1['f3'],
        f4=step_1['f4'],
        ctrl=step_5['ctrl'],
        a=step_5['a'],
        b=step_5['b'],
        x=step_5['x'],
        y=step_5['y'],
    )
    assert ret1 == ret2


@pytest.mark.parametrize(
    ['n', 'm'], [(n, m) for n in range(7, 8) for m in range(1, n + 1) if n % m == 0]
)
@pytest.mark.parametrize('a,b', [(15, 13), (2, 10)])
@pytest.mark.parametrize('x,y', [(15, 13), (0, 0)])
def test_ec_add_classical_fast(n, m, a, b, x, y):
    p = 17
    bloq = ECAdd(n=n, mod=p, window_size=m)
    lam_num = (3 * a**2) % p
    lam_denom = (2 * b) % p
    lam_r = p if b == 0 else (lam_num * pow(lam_denom, -1, mod=p)) % p
    dtype = QMontgomeryUInt(n, p)
    ret1 = bloq.call_classically(
        a=dtype.uint_to_montgomery(a),
        b=dtype.uint_to_montgomery(b),
        x=dtype.uint_to_montgomery(x),
        y=dtype.uint_to_montgomery(y),
        lam_r=dtype.uint_to_montgomery(lam_r),
    )
    ret2 = bloq.decompose_bloq().call_classically(
        a=dtype.uint_to_montgomery(a),
        b=dtype.uint_to_montgomery(b),
        x=dtype.uint_to_montgomery(x),
        y=dtype.uint_to_montgomery(y),
        lam_r=dtype.uint_to_montgomery(lam_r),
    )
    assert ret1 == ret2


@pytest.mark.slow
@pytest.mark.parametrize(
    ['n', 'm'], [(n, m) for n in range(7, 9) for m in range(1, n + 1) if n % m == 0]
)
@pytest.mark.parametrize(
    'a,b',
    [
        (15, 13),
        (2, 10),
        (8, 3),
        (12, 1),
        (6, 6),
        (5, 8),
        (10, 15),
        (1, 12),
        (3, 0),
        (1, 5),
        (10, 2),
        (0, 0),
    ],
)
@pytest.mark.parametrize('x,y', [(15, 13), (5, 8), (10, 15), (1, 12), (3, 0), (1, 5), (10, 2)])
def test_ec_add_classical(n, m, a, b, x, y):
    p = 17
    bloq = ECAdd(n=n, mod=p, window_size=m)
    lam_num = (3 * a**2) % p
    lam_denom = (2 * b) % p
    lam_r = p if b == 0 else (lam_num * pow(lam_denom, -1, mod=p)) % p
    dtype = QMontgomeryUInt(n, p)
    ret1 = bloq.call_classically(
        a=dtype.uint_to_montgomery(a),
        b=dtype.uint_to_montgomery(b),
        x=dtype.uint_to_montgomery(x),
        y=dtype.uint_to_montgomery(y),
        lam_r=dtype.uint_to_montgomery(lam_r),
    )
    ret2 = bloq.decompose_bloq().call_classically(
        a=dtype.uint_to_montgomery(a),
        b=dtype.uint_to_montgomery(b),
        x=dtype.uint_to_montgomery(x),
        y=dtype.uint_to_montgomery(y),
        lam_r=dtype.uint_to_montgomery(lam_r),
    )
    assert ret1 == ret2


@pytest.mark.parametrize('p', (7, 9, 11))
@pytest.mark.parametrize(
    ['n', 'window_size'],
    [
        (n, window_size)
        for n in range(5, 8)
        for window_size in range(1, n + 1)
        if n % window_size == 0
    ],
)
def test_ec_add_decomposition(n, window_size, p):
    b = ECAdd(n=n, window_size=window_size, mod=p)
    qlt_testing.assert_valid_bloq_decomposition(b)


@pytest.mark.parametrize('p', (7, 9, 11))
@pytest.mark.parametrize(
    ['n', 'window_size'],
    [
        (n, window_size)
        for n in range(5, 8)
        for window_size in range(1, n + 1)
        if n % window_size == 0
    ],
)
def test_ec_add_bloq_counts(n, window_size, p):
    b = ECAdd(n=n, window_size=window_size, mod=p)
    qlt_testing.assert_equivalent_bloq_counts(b, [ignore_alloc_free, ignore_split_join])


def test_ec_add_symbolic_cost():
    n, m, p = sympy.symbols('n m p', integer=True)

    # In Litinski 2023 https://arxiv.org/abs/2306.08585 a window size of 4 is used.
    # The cost function generally has floor/ceil division that disappear for bitsize=0 mod 4.
    # This is why instead of using bitsize=n directly, we use bitsize=4*m=n.
    b = ECAdd(n=4 * m, window_size=4, mod=p)
    cost = get_cost_value(b, QECGatesCost()).total_t_and_ccz_count()
    # We have some T gates since we use CSwapApprox instead of n CSWAPs in KaliskiModInverse.
    total_toff = (cost['n_t'] / 4 + cost['n_ccz']) * sympy.Integer(1)
    total_toff = total_toff.subs(m, n / 4).expand()

    # Litinski 2023 https://arxiv.org/abs/2306.08585
    # Based on the counts from Figures 3, 5, and 8 the toffoli count for ECAdd is 126.5n^2 + 189n.
    # The following formula is 126.5n^2 + 215.5n - 34. We account for the discrepancy in the
    # coefficient of n by a reduction in the toffoli cost of Montgomery ModMult, an increase in the
    # toffoli cost for Kaliski Mod Inverse, n extra toffolis in ModNeg, 2n extra toffolis to do n
    # 3-controlled toffolis in step 2, and a few extra gates added to fix bugs found in the circuit
    # (see class docstrings). The expression is written with rationals because sympy comparison
    # fails with floats.
    assert total_toff == sympy.Rational(253, 2) * n**2 + sympy.Rational(431, 2) * n - 34


def test_ec_add(bloq_autotester):
    bloq_autotester(_ec_add)


def test_ec_add_small(bloq_autotester):
    bloq_autotester(_ec_add_small)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('ec_add')


def test_ec_add_small_gate_cost():
    bloq = _ec_add_small.make()
    assert get_cost_value(bloq, QECGatesCost()).toffoli == 29
