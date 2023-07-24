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
import sympy

from qualtran import Bloq
from qualtran.bloqs.factoring.mod_exp import ModExp
from qualtran.bloqs.factoring.mod_mul import CtrlModMul
from qualtran.bloqs.util_bloqs import Join, Split
from qualtran.resource_counting import get_cbloq_bloq_counts, SympySymbolAllocator
from qualtran.testing import execute_notebook


def _make_modexp():
    from qualtran.bloqs.factoring.mod_exp import ModExp

    return ModExp(base=3, mod=15, exp_bitsize=3, x_bitsize=2048)


def test_mod_exp_consistent_classical():
    rs = np.random.RandomState(52)

    # 100 random attribute choices.
    for _ in range(100):
        # Sample moduli in a range. Set x_bitsize=n big enough to fit.
        mod = rs.randint(4, 123)
        n = int(np.ceil(np.log2(mod)))
        n = rs.randint(n, n + 10)

        # Choose an exponent in a range. Set exp_bitsize=ne bit enough to fit.
        exponent = rs.randint(1, 20)
        ne = int(np.ceil(np.log2(exponent)))
        ne = rs.randint(ne, ne + 10)

        # Choose a base smaller than mod.
        base = rs.randint(1, mod)

        bloq = ModExp(base=base, exp_bitsize=ne, x_bitsize=n, mod=mod)
        ret1 = bloq.call_classically(exponent=exponent)
        ret2 = bloq.decompose_bloq().call_classically(exponent=exponent)
        assert ret1 == ret2


def test_mod_exp_symbolic():
    g, N, n_e, n_x = sympy.symbols('g N n_e, n_x')
    modexp = ModExp(base=g, mod=N, exp_bitsize=n_e, x_bitsize=n_x)
    assert modexp.short_name() == 'g^e % N'
    counts = modexp.bloq_counts(SympySymbolAllocator())
    counts_by_bloq = {bloq.pretty_name(): n for n, bloq in counts}
    assert counts_by_bloq['|1>'] == 1
    assert counts_by_bloq['CtrlModMul'] == n_e

    b, x = modexp.call_classically(exponent=sympy.Symbol('b'))
    assert str(x) == 'Mod(g**b, N)'


def test_mod_exp_consistent_counts():

    bloq = ModExp(base=8, exp_bitsize=3, x_bitsize=10, mod=50)
    counts1 = bloq.bloq_counts(SympySymbolAllocator())

    ssa = SympySymbolAllocator()
    my_k = ssa.new_symbol('k')

    def generalize(b: Bloq) -> Optional[Bloq]:
        if isinstance(b, CtrlModMul):
            # Symbolic k in `CtrlModMul`.
            return attrs.evolve(b, k=my_k)
        if isinstance(b, (Split, Join)):
            # Ignore these
            return
        return b

    counts2 = get_cbloq_bloq_counts(bloq.decompose_bloq(), generalizer=generalize)

    assert counts1 == counts2


def test_intro_notebook():
    execute_notebook('factoring-via-modexp')


def test_ref_notebook():
    execute_notebook('ref-factoring')
