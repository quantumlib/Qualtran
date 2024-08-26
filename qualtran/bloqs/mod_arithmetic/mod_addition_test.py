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

import itertools

import numpy as np
import pytest
import sympy

from qualtran import QMontgomeryUInt, QUInt
from qualtran.bloqs.arithmetic import Add
from qualtran.bloqs.mod_arithmetic import CModAdd, CModAddK, CtrlScaleModAdd, ModAdd, ModAddK
from qualtran.bloqs.mod_arithmetic.mod_addition import _cmodadd_example
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.resource_counting import GateCounts, get_cost_value, QECGatesCost
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join
from qualtran.testing import (
    assert_equivalent_bloq_counts,
    assert_valid_bloq_decomposition,
    execute_notebook,
)


def identity_map(n: int):
    """Returns a dict of size `2**n` mapping each integer in range [0, 2**n) to itself."""
    return {i: i for i in range(2**n)}


def test_add_mod_n_protocols():
    with pytest.raises(ValueError, match="must be between"):
        _ = ModAddK(3, 10)
    add_one = ModAddK(3, 5, 1)
    add_two = ModAddK(3, 5, 2, cvs=[1, 0])

    assert add_one == ModAddK(3, 5, 1)
    assert add_one != add_two
    assert hash(add_one) != hash(add_two)
    assert add_two.cvs == (1, 0)


def add_constant_mod_n_ref_t_complexity_(b: ModAddK) -> TComplexity:
    # Rough cost as given in https://arxiv.org/abs/1905.09749
    return 5 * Add(QUInt(b.bitsize)).t_complexity()


@pytest.mark.parametrize('bitsize', [3, 9])
def test_add_mod_n_gate_counts(bitsize):
    bloq = ModAddK(bitsize, mod=8, add_val=2, cvs=[0, 1])
    assert bloq.t_complexity() == add_constant_mod_n_ref_t_complexity_(bloq)


def test_ctrl_scale_mod_add():
    bloq = CtrlScaleModAdd(k=123, mod=13 * 17, bitsize=8)

    counts = bloq.bloq_counts()
    ((bloq, n),) = counts.items()
    assert n == 8


def test_ctrl_mod_add_k():
    bloq = CModAddK(k=123, mod=13 * 17, bitsize=8)

    counts = bloq.bloq_counts()
    ((bloq, n),) = counts.items()
    assert n == 5


@pytest.mark.parametrize('bitsize,p', [(1, 1), (2, 3), (5, 8)])
def test_mod_add_valid_decomp(bitsize, p):
    bloq = ModAdd(bitsize=bitsize, mod=p)
    assert_valid_bloq_decomposition(bloq)


@pytest.mark.parametrize('bitsize', list(range(1, 6)) + [sympy.Symbol('n')])
def test_mod_add_symbolic_cost(bitsize):
    tcomplexity = ModAdd(bitsize, sympy.Symbol('p')).t_complexity()
    assert tcomplexity.t == 16 * bitsize - 4  # 4n toffoli
    assert tcomplexity.rotations == 0


@pytest.mark.parametrize(
    ['prime', 'bitsize'],
    [(p, bitsize) for p in [11, 13, 31] for bitsize in range(1 + p.bit_length(), 8)],
)
def test_classical_action_mod_add(prime, bitsize):
    b = ModAdd(bitsize=bitsize, mod=prime)
    cb = b.decompose_bloq()
    valid_range = range(prime)
    for x in valid_range:
        for y in valid_range:
            assert b.call_classically(x=x, y=y) == cb.call_classically(x=x, y=y)


@pytest.mark.slow
@pytest.mark.parametrize('control', range(2))
@pytest.mark.parametrize(
    ['prime', 'bitsize'],
    [(p, bitsize) for p in [11, 13, 31] for bitsize in range(1 + p.bit_length(), 8)],
)
@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
def test_classical_action_cmodadd(control, prime, dtype, bitsize):
    b = CModAdd(dtype(bitsize), mod=prime, cv=control)
    cb = b.decompose_bloq()
    valid_range = range(prime)
    for c in range(2):
        for x, y in itertools.product(valid_range, repeat=2):
            assert b.call_classically(ctrl=c, x=x, y=y) == cb.call_classically(ctrl=c, x=x, y=y)


@pytest.mark.parametrize('control', range(2))
@pytest.mark.parametrize('bitsize', [4, 5])
def test_classical_action_cmodadd_fast(control, bitsize):
    prime = 11
    b = CModAdd(QMontgomeryUInt(bitsize), mod=prime, cv=control)
    cb = b.decompose_bloq()
    rng = np.random.default_rng(341)
    for c in range(2):
        for x, y in rng.choice(prime, (10, 2)):
            assert b.call_classically(ctrl=c, x=x, y=y) == cb.call_classically(ctrl=c, x=x, y=y)


@pytest.mark.parametrize('control', range(2))
@pytest.mark.parametrize(
    ['prime', 'bitsize'],
    [(p, bitsize) for p in [11, 13, 31] for bitsize in range(1 + p.bit_length(), 8)],
)
@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
def test_cmodadd_decomposition(control, prime, dtype, bitsize):
    b = CModAdd(dtype(bitsize), mod=prime, cv=control)
    assert_valid_bloq_decomposition(b)
    assert_equivalent_bloq_counts(b, [ignore_alloc_free, ignore_split_join])


@pytest.mark.parametrize('control', range(2))
@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
def test_cmodadd_cost(control, dtype):
    prime = sympy.Symbol('p')
    n = sympy.Symbol('n')
    b = CModAdd(dtype(n), mod=prime, cv=control)
    cost: GateCounts = get_cost_value(b, QECGatesCost())
    n_toffolis = 5 * n + 1
    assert cost.total_t_count() == 4 * n_toffolis


def test_cmodadd_example(bloq_autotester):
    bloq_autotester(_cmodadd_example)


@pytest.mark.notebook
def test_notebook():
    execute_notebook('mod_addition')


def test_cmod_add_complexity_vs_ref():
    n, k = sympy.symbols('n k', integer=True, positive=True)
    bloq = CModAdd(QUInt(n), mod=k)
    counts = get_cost_value(bloq, QECGatesCost()).total_t_and_ccz_count()
    assert counts['n_t'] == 0, 'all toffoli'

    # Litinski 2023 https://arxiv.org/abs/2306.08585
    # Figure/Table 8. Lists n-qubit controlled modular addition as 5n toffoli.
    #     Note: We have an extra toffoli due to how our OutOfPlaceAdder works.
    assert counts['n_ccz'] == 5 * n + 1
