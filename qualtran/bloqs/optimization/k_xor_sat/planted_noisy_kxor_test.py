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

from qualtran.drawing import show_call_graph
from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join
from qualtran.symbolics import ln, log2

from .kxor_instance import KXorInstance
from .planted_noisy_kxor import (
    _solve_planted,
    _solve_planted_symbolic,
    AliceTheorem,
    PlantedNoisyKXOR,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.mark.xfail
def test_alice_thm_symb():
    n, m = sympy.symbols("n m", positive=True, integer=True)
    k = sympy.symbols("k", positive=True, integer=True, even=True)
    rho = sympy.symbols(r"\rho", positive=True, real=True)
    c = sympy.symbols(r"c", positive=True, integer=True)
    thm = AliceTheorem(n=n, k=k, ell=c * k, kappa=0.99 * rho, eps=0.005)
    _ = thm.C_kappa()
    _ = thm.min_m()
    _ = thm.fail_prob()


@pytest.mark.parametrize("bloq_ex", [_solve_planted, _solve_planted_symbolic])
def test_examples(bloq_autotester, bloq_ex):
    if bloq_autotester.check_name == 'serialize':
        pytest.skip()

    bloq_autotester(bloq_ex)


def test_call_graph():
    _solve_planted().call_graph()


def test_call_graph_symb():
    algo = _solve_planted_symbolic()
    g, sigma = algo.call_graph(generalizer=[ignore_split_join, ignore_alloc_free])
    show_call_graph(g)


def example_random_instance(*, k=4, n=100, m=1000, c=2, rho=0.8, seed=120) -> PlantedNoisyKXOR:
    # generate instance
    rng = np.random.default_rng(seed)
    ell = c * k
    inst = KXorInstance.random_instance(n=n, m=m, k=k, planted_advantage=rho, rng=rng)
    algo_bloq = PlantedNoisyKXOR.from_inst(inst=inst, ell=ell, rho=rho, zeta=1 / ln(n), rng=rng)

    return algo_bloq


def test_gate_cost():
    bloq = example_random_instance()
    gc = get_cost_value(bloq, QECGatesCost())
    t_cost = gc.total_t_count(ts_per_cswap=4)

    n = bloq.inst_guide.n
    m = bloq.inst_guide.m + bloq.inst_solve.m
    ell = bloq.ell
    c = ell // bloq.inst_guide.k

    big_O_expected = n ** (ell / 4) * (m**0.5) * ell**ell * log2(n) ** (c // 2)
    print()
    print(t_cost)
    print(t_cost / big_O_expected)
    print(big_O_expected)
    print(t_cost / big_O_expected * bloq.guiding_state_overlap)
    print(1 / bloq.guiding_state_overlap)
    print(1 / bloq.guiding_state_overlap_guarantee.overlap_probability**0.5)


@pytest.mark.parametrize("n", [40, 50, 60, 70, 80, 90, 100])
@pytest.mark.parametrize("k", [4, 8])
@pytest.mark.parametrize("c", [2, 3, 4])
def test_more_costs(n, k, c):
    if c * k == 32 and n <= 40:
        pytest.skip("too small n")

    bloq = example_random_instance(k=k, c=c, n=n, m=n, seed=142)
    cost = get_cost_value(bloq, QECGatesCost())
    print(cost)


@pytest.mark.slow
@pytest.mark.parametrize("n", [10**4, 10**5])
def test_large(n):
    k = 4
    c = 32 // 4
    bloq = example_random_instance(k=k, c=c, n=n, m=n * 10, seed=142)
    cost = get_cost_value(bloq, QECGatesCost())
    print(cost)
