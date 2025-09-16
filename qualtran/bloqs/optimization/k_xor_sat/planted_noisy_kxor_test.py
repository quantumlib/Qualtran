#  Copyright 2025 Google LLC
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
from qualtran.bloqs.optimization.k_xor_sat.kxor_instance import KXorInstance
from qualtran.bloqs.optimization.k_xor_sat.planted_noisy_kxor import (
    _solve_planted,
    _solve_planted_symbolic,
    AliceTheorem,
    PlantedNoisyKXOR,
)
from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.symbolics import ln


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
    bloq_autotester(bloq_ex)


def example_random_instance(
    *, k=4, n=100, m=1000, c=2, rho=0.8, rng: np.random.Generator
) -> PlantedNoisyKXOR:
    # generate instance
    ell = c * k
    inst = KXorInstance.random_instance(n=n, m=m, k=k, planted_advantage=rho, rng=rng)
    algo_bloq = PlantedNoisyKXOR.from_inst(inst=inst, ell=ell, rho=rho, zeta=1 / ln(n), rng=rng)

    return algo_bloq


def test_gate_cost():
    rng = np.random.default_rng(42)
    bloq = example_random_instance(rng=rng)
    gc = get_cost_value(bloq, QECGatesCost())
    t_cost = gc.total_t_count(ts_per_cswap=4)
    assert t_cost == 104471529365303256253


@pytest.mark.parametrize("n", [40, 50, 60, 70, 80, 90, 100])
@pytest.mark.parametrize("k", [4, 8])
@pytest.mark.parametrize("c", [2, 3, 4])
def test_more_costs(n, k, c):
    if c * k == 32 and n <= 40:
        pytest.skip("too small n")

    bloq = example_random_instance(k=k, c=c, n=n, m=n, rng=np.random.default_rng(142))
    _cost = get_cost_value(bloq, QECGatesCost())


@pytest.mark.slow
@pytest.mark.parametrize("n", [10**4])
def test_large(n):
    k = 4
    c = 32 // 4
    bloq = example_random_instance(k=k, c=c, n=n, m=n * 10, rng=np.random.default_rng(142))
    _cost = get_cost_value(bloq, QECGatesCost())


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('planted_noisy_kxor')


@pytest.mark.notebook
def test_tutorial():
    qlt_testing.execute_notebook('tutorial_planted_noisy_kxor')
