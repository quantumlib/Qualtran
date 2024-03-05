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

from typing import Optional

import attrs
import numpy as np
import pytest
import sympy

import qualtran.testing as qlt_testing
from qualtran import Bloq
from qualtran.bloqs.factoring.mod_add import CtrlScaleModAdd
from qualtran.bloqs.factoring.mod_mul import _modmul, _modmul_symb, CtrlModMul, MontgomeryModDbl
from qualtran.bloqs.util_bloqs import Allocate, Free
from qualtran.resource_counting import SympySymbolAllocator
from qualtran.testing import assert_valid_bloq_decomposition


def test_consistent_classical():
    rs = np.random.RandomState(52)
    primes = [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
    ]

    # 100 random attribute choices.
    for _ in range(100):
        # Choose a mod in a range, set bitsize=n big enough to fit.
        p, q = rs.choice(primes, 2)
        mod = int(p) * int(q)
        n = int(np.ceil(np.log2(mod)))
        n = rs.randint(n, n + 10)

        # choose a random constant and variable within mod
        k = rs.randint(1, mod)
        x = rs.randint(1, mod)

        try:
            pow(k, -1, mod=mod)
        except ValueError as e:
            if str(e) == 'base is not invertible for the given modulus':
                continue
            raise e

        bloq = CtrlModMul(k=k, mod=mod, bitsize=n)

        # ctrl on
        ret1 = bloq.call_classically(ctrl=1, x=x)
        ret2 = bloq.decompose_bloq().call_classically(ctrl=1, x=x)
        assert ret1 == ret2

        # ctrl off
        ret1 = bloq.call_classically(ctrl=0, x=x)
        ret2 = bloq.decompose_bloq().call_classically(ctrl=0, x=x)
        assert ret1 == ret2


def test_modmul_symb_manual():
    k, N, n_x = sympy.symbols('k N n_x')
    bloq = CtrlModMul(k=k, mod=N, bitsize=n_x)
    assert bloq.short_name() == 'x *= k % N'

    # it's all fixed constants, but check it works anyways
    counts = bloq.bloq_counts()
    assert len(counts) > 0

    ctrl, x = bloq.call_classically(ctrl=1, x=sympy.Symbol('x'))
    assert str(x) == 'Mod(k*x, N)'

    ctrl, x = bloq.call_classically(ctrl=0, x=sympy.Symbol('x'))
    assert str(x) == 'x'


def test_consistent_counts():
    bloq = CtrlModMul(k=123, mod=13 * 17, bitsize=8)
    counts1 = bloq.bloq_counts()

    ssa = SympySymbolAllocator()
    my_k = ssa.new_symbol('k')

    def generalize(b: Bloq) -> Optional[Bloq]:
        if isinstance(b, CtrlScaleModAdd):
            return attrs.evolve(b, k=my_k)

        if isinstance(b, (Free, Allocate)):
            return
        return b

    counts2 = bloq.decompose_bloq().bloq_counts(generalizer=generalize)

    assert counts1 == counts2


@pytest.mark.parametrize('bitsize,p', [(1, 1), (2, 3), (5, 8)])
def test_montgomery_mod_dbl_decomp(bitsize, p):
    bloq = MontgomeryModDbl(bitsize=bitsize, p=p)
    assert_valid_bloq_decomposition(bloq)


def test_modul(bloq_autotester):
    bloq_autotester(_modmul)


def test_modul_symb(bloq_autotester):
    bloq_autotester(_modmul_symb)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('mod_mul')
