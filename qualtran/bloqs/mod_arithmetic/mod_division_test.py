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
from qualtran.bloqs.mod_arithmetic.mod_division import _kaliskimodinverse_example, KaliskiModInverse
from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join


@pytest.mark.parametrize('bitsize', [5, 6])
@pytest.mark.parametrize('mod', [3, 5, 7, 11, 13, 15])
def test_kaliski_mod_inverse_classical_action(bitsize, mod):
    blq = KaliskiModInverse(bitsize, mod)
    cblq = blq.decompose_bloq()
    dtype = QMontgomeryUInt(bitsize)
    R = pow(2, bitsize, mod)
    for x in range(1, mod):
        if math.gcd(x, mod) != 1:
            continue
        x_montgomery = dtype.uint_to_montgomery(x, mod)
        res = blq.call_classically(x=x_montgomery)
        assert res == cblq.call_classically(x=x_montgomery)
        assert len(res) == 2
        assert res[0] == dtype.montgomery_inverse(x_montgomery, mod)
        assert dtype.montgomery_product(int(res[0]), x_montgomery, mod) == R
        assert blq.adjoint().call_classically(x=res[0], m=res[1]) == (x_montgomery,)


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
    cost = get_cost_value(b, QECGatesCost()).total_t_and_ccz_count()
    # We have some T gates since we use CSwapApprox instead of n CSWAPs.
    total_toff = (cost['n_t'] / 4 + cost['n_ccz']) * sympy.Integer(1)
    total_toff = total_toff.expand()

    # The toffoli cost from Litinski https://arxiv.org/abs/2306.08585 is 26n^2 + 2n.
    # The cost of Kaliski is 2*n*(cost of an iteration) + (cost of computing $p - x$)
    #
    #   - The cost of of computing  $p-x$ in Litinski is 2n (Neg -> Add(p)). In our
    #       construction this is just $n-1$ (BitwiseNot -> Add(p+1)).
    #   - The cost of an iteration in Litinski $13n$ since they ignore constants.
    #       Our construction is exactly the same but we also count the constants
    #       which amout to $3$. for a total cost of $13n + 3$.
    # For example the cost of ModDbl is 2n+1. In their figure 8, they report
    # it as just $2n$. ModDbl gets executed within the 2n loop so its contribution
    # to the overal cost should be 4n^2 + 2n instead of just 4n^2.
    assert total_toff == 26 * n**2 + 7 * n - 1


def test_kaliskimodinverse_example(bloq_autotester):
    bloq_autotester(_kaliskimodinverse_example)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('mod_division')
