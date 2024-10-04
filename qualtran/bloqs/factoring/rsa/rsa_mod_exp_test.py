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

from typing import cast, Optional

import attrs
import numpy as np
import pytest
import sympy

from qualtran import Bloq, QUInt
from qualtran.bloqs.bookkeeping import Join, Split
from qualtran.bloqs.factoring.rsa.rsa_mod_exp import _modexp, ModExp
from qualtran.bloqs.mod_arithmetic import CModMulK
from qualtran.drawing import Text
from qualtran.resource_counting import SympySymbolAllocator
from qualtran.symbolics.types import HasLength
from qualtran.testing import execute_notebook


# TODO: Fix ModExp and improve this test
def test_mod_exp_consistent_classical():
    rs = np.random.RandomState(52)

    # 100 random attribute choices.
    for _ in range(100):
        # Sample moduli in a range. Set x_bitsize=n big enough to fit.
        mod = 7 * 13
        n = int(np.ceil(np.log2(mod)))

        # Choose an exponent in a range. Set exp_bitsize=ne bit enough to fit.
        exponent = rs.randint(1, 2**n)
        ne = 2 * n

        # Choose a base smaller than mod.
        base = rs.randint(1, mod)
        while np.gcd(base, mod) != 1:
            base = rs.randint(1, mod)

        bloq = ModExp(base=base, exp_bitsize=ne, x_bitsize=n, mod=mod)
        ret1 = bloq.call_classically(exponent=QUInt(ne).to_bits_array(exponent)[0], x=1)
        ret2 = bloq.decompose_bloq().call_classically(
            exponent=QUInt(ne).to_bits_array(exponent)[0], x=1
        )
        assert len(ret1) == len(ret2)
        for i in range(len(ret1)):
            np.testing.assert_array_equal(ret1[i], ret2[i])


def test_mod_exp_consistent_counts():
    bloq = ModExp(base=11, exp_bitsize=3, x_bitsize=10, mod=50)

    counts1 = bloq.bloq_counts()

    ssa = SympySymbolAllocator()
    my_k = ssa.new_symbol('k')

    def generalize(b: Bloq) -> Optional[Bloq]:
        if isinstance(b, CModMulK):
            # Symbolic k in `CModMulK`.
            return attrs.evolve(b, k=my_k)
        if isinstance(b, (Split, Join)):
            # Ignore these
            return None
        return b

    counts2 = bloq.decompose_bloq().bloq_counts(generalizer=generalize)
    assert counts1 == counts2


def test_mod_exp_t_complexity():
    bloq = ModExp(base=11, exp_bitsize=3, x_bitsize=10, mod=50)
    tcomp = bloq.t_complexity()
    assert tcomp.t > 0


def test_modexp(bloq_autotester):
    bloq_autotester(_modexp)


@pytest.mark.notebook
def test_intro_notebook():
    execute_notebook('factoring-via-modexp')
