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
import random
from typing import Dict, Tuple, Union

import cirq
import numpy as np
import pytest
import sympy

from qualtran import Bloq
from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.bookkeeping import ArbitraryClifford
from qualtran.bloqs.swap_network.cswap_approx import (
    _approx_cswap_large,
    _approx_cswap_small,
    _approx_cswap_symb,
    CSwapApprox,
)
from qualtran.cirq_interop.t_complexity_protocol import t_complexity, TComplexity
from qualtran.testing import assert_valid_bloq_decomposition, execute_notebook

random.seed(12345)


def test_cswap_approx_decomp():
    csa = CSwapApprox(10)
    assert_valid_bloq_decomposition(csa)


def test_cswap_approx_decomposition():
    csa = CSwapApprox(4)
    circuit = (
        csa.as_composite_bloq().to_cirq_circuit()
        + csa.adjoint().as_composite_bloq().to_cirq_circuit()
    )
    initial_state = cirq.testing.random_superposition(2**9, random_state=1234)
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, initial_state=initial_state)
    np.testing.assert_allclose(result.final_state_vector, initial_state)


@pytest.mark.parametrize('n', [5, 32])
def test_approx_cswap_t_count(n):
    cswap = CSwapApprox(bitsize=n)
    cswap_d = cswap.decompose_bloq()

    assert cswap.t_complexity() == cswap_d.t_complexity()


def get_t_count_and_clifford(
    bc: Dict[Bloq, Union[int, sympy.Expr]]
) -> Tuple[Union[int, sympy.Expr], Union[int, sympy.Expr]]:
    """Get the t count and clifford cost from bloq count."""
    cliff_cost = sum([v for k, v in bc.items() if isinstance(k, ArbitraryClifford)])
    t_cost = sum([v for k, v in bc.items() if isinstance(k, TGate)])
    return t_cost, cliff_cost


@pytest.mark.parametrize("n", [*range(1, 6)])
def test_t_complexity_cswap_approx(n):
    actual = t_complexity(CSwapApprox(n))
    assert actual == TComplexity(t=4 * n, clifford=22 * n - 1)


@pytest.mark.parametrize("n", [*range(2, 6)])
def test_cswap_approx_bloq_counts(n):
    csa = CSwapApprox(n)
    bc = csa.bloq_counts()
    t_cost, cliff_cost = get_t_count_and_clifford(bc)
    assert csa.t_complexity().clifford == cliff_cost
    assert csa.t_complexity().t == t_cost


def test_approx_cswap_small(bloq_autotester):
    bloq_autotester(_approx_cswap_small)


def test_approx_cswap_symb(bloq_autotester):
    bloq_autotester(_approx_cswap_symb)


def test_approx_cswap_large(bloq_autotester):
    bloq_autotester(_approx_cswap_large)


@pytest.mark.notebook
def test_notebook():
    execute_notebook('swap_network')
