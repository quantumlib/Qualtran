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

import math

import pytest
import sympy

import qualtran.testing as qlt_testing
from qualtran import QMontgomeryUInt
from qualtran.bloqs.mod_arithmetic import mod_division
from qualtran.bloqs.mod_arithmetic.mod_division import _kaliskimodinverse_example, KaliskiModInverse
from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join


@pytest.mark.parametrize('bitsize', [5, 6])
@pytest.mark.parametrize('mod', [3, 5, 7, 11, 13, 15])
def test_kaliski_mod_inverse_classical_action(bitsize, mod):
    blq = KaliskiModInverse(bitsize, mod)
    cblq = blq.decompose_bloq()
    dtype = QMontgomeryUInt(bitsize, mod)
    R = pow(2, bitsize, mod)
    for x in range(1, mod):
        if math.gcd(x, mod) != 1:
            continue
        x_montgomery = dtype.uint_to_montgomery(x)
        res = blq.call_classically(x=x_montgomery)

        assert res == cblq.call_classically(x=x_montgomery)
        assert len(res) == 2
        assert res[0] == dtype.montgomery_inverse(x_montgomery)
        assert dtype.montgomery_product(int(res[0]), x_montgomery) == R
        assert blq.adjoint().call_classically(x=res[0], junk=res[1]) == (x_montgomery,)


@pytest.mark.parametrize('bitsize', [5, 6])
@pytest.mark.parametrize('mod', [3, 5, 7, 11, 13, 15])
def test_kaliski_mod_inverse_classical_action_zero(bitsize, mod):
    blq = KaliskiModInverse(bitsize, mod)
    cblq = blq.decompose_bloq()
    # When x = 0 the terminal condition is achieved at the first iteration, this corresponds to
    # m_0 = is_terminal_0 = 1 and all other bits = 0.
    junk = 2 ** (4 * bitsize - 1) + 2 ** (2 * bitsize - 1)
    assert blq.call_classically(x=0) == cblq.call_classically(x=0) == (0, junk)
    assert blq.adjoint().call_classically(x=0, junk=junk) == (0,)


@pytest.mark.parametrize('bitsize', [5, 6])
@pytest.mark.parametrize('mod', [3, 5, 7, 11, 13, 15])
def test_kaliski_mod_inverse_decomposition(bitsize, mod):
    b = KaliskiModInverse(bitsize, mod)
    qlt_testing.assert_valid_bloq_decomposition(b)


@pytest.mark.parametrize('bitsize', [5, 6])
@pytest.mark.parametrize('mod', [3, 5, 7, 11, 13, 15])
def test_kaliski_mod_bloq_counts(bitsize, mod):
    b = KaliskiModInverse(bitsize, mod)
    qlt_testing.assert_equivalent_bloq_counts(b, [ignore_alloc_free, ignore_split_join])


def test_kaliski_symbolic_cost():
    n, p = sympy.symbols('n p')
    b = KaliskiModInverse(n, p)
    total_toff = get_cost_value(b, QECGatesCost()).total_toffoli_only()
    total_toff = sympy.expand(total_toff)

    # The toffoli cost from Litinski https://arxiv.org/abs/2306.08585 is 26n^2 + 2n.
    # The cost of Kaliski is 2*n*(cost of an iteration) + (cost of computing $p - x$)
    #
    #   - The cost of of computing  $p-x$ in Litinski is 2n (Neg -> Add(p)). In our
    #       construction this is just $n-1$ (BitwiseNot -> Add(p+1)).
    #   - The cost of an iteration in Litinski $13n$ since they ignore constants.
    #       Our construction is exactly the same but we also count the constants
    #       which amout to $3$. for a total cost of $13n + 4$.
    # For example the cost of ModDbl is 2n+1. In their figure 8, they report
    # it as just $2n$. ModDbl gets executed within the 2n loop so its contribution
    # to the overal cost should be 4n^2 + 2n instead of just 4n^2.
    assert total_toff == 26 * n**2 + 9 * n - 1


def test_kaliskimodinverse_example(bloq_autotester):
    bloq_autotester(_kaliskimodinverse_example)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('mod_division')


def test_kaliski_iteration_decomposition():
    mod = 7
    bitsize = 5
    b = mod_division._KaliskiIteration(bitsize, mod)
    cb = b.decompose_bloq()
    for x in range(mod):
        u = mod
        v = x
        r = 0
        s = 1
        f = 1

        for _ in range(2 * bitsize):
            inputs = {'u': u, 'v': v, 'r': r, 's': s, 'm': 0, 'f': f, 'is_terminal': 0}
            res = b.call_classically(**inputs)
            assert res == cb.call_classically(**inputs), f'{inputs=}'
            u, v, r, s, _, f, _ = res  # type: ignore

    qlt_testing.assert_valid_bloq_decomposition(b)
    qlt_testing.assert_equivalent_bloq_counts(b, generalizer=(ignore_alloc_free, ignore_split_join))


def test_kaliski_steps():
    bitsize = 5
    mod = 7
    steps = [
        mod_division._KaliskiIterationStep1(bitsize),
        mod_division._KaliskiIterationStep2(bitsize),
        mod_division._KaliskiIterationStep3(bitsize),
        mod_division._KaliskiIterationStep4(bitsize),
        mod_division._KaliskiIterationStep5(bitsize),
        mod_division._KaliskiIterationStep6(bitsize, mod),
    ]
    csteps = [b.decompose_bloq() for b in steps]

    # check decomposition is valid.
    for step in steps:
        qlt_testing.assert_valid_bloq_decomposition(step)
        qlt_testing.assert_equivalent_bloq_counts(
            step, generalizer=(ignore_alloc_free, ignore_split_join)
        )

    # check that for all inputs all 2n iteration work when excuted directly on the 6 steps
    # and their decompositions.
    for x in range(mod):
        u, v, r, s, f = mod, x, 0, 1, 1

        for _ in range(2 * bitsize):
            a = b = m = is_terminal = 0

            res = steps[0].call_classically(v=v, m=m, f=f, is_terminal=is_terminal)
            assert res == csteps[0].call_classically(v=v, m=m, f=f, is_terminal=is_terminal)
            v, m, f, is_terminal = res  # type: ignore

            res = steps[1].call_classically(u=u, v=v, b=b, a=a, m=m, f=f)
            assert res == csteps[1].call_classically(u=u, v=v, b=b, a=a, m=m, f=f)
            u, v, b, a, m, f = res  # type: ignore

            res = steps[2].call_classically(u=u, v=v, b=b, a=a, m=m, f=f)
            assert res == csteps[2].call_classically(u=u, v=v, b=b, a=a, m=m, f=f)
            u, v, b, a, m, f = res  # type: ignore

            res = steps[3].call_classically(u=u, v=v, r=r, s=s, a=a)
            assert res == csteps[3].call_classically(u=u, v=v, r=r, s=s, a=a)
            u, v, r, s, a = res  # type: ignore

            res = steps[4].call_classically(u=u, v=v, r=r, s=s, b=b, f=f)
            assert res == csteps[4].call_classically(u=u, v=v, r=r, s=s, b=b, f=f)
            u, v, r, s, b, f = res  # type: ignore

            res = steps[5].call_classically(u=u, v=v, r=r, s=s, b=b, a=a, m=m, f=f)
            assert res == csteps[5].call_classically(u=u, v=v, r=r, s=s, b=b, a=a, m=m, f=f)
            u, v, r, s, b, a, m, f = res  # type: ignore
