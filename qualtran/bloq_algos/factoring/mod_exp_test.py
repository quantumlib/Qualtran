from typing import Optional

import attrs
import numpy as np
import sympy

from qualtran import Bloq
from qualtran.bloq_algos.factoring.mod_exp import ModExp
from qualtran.bloq_algos.factoring.mod_mul import CtrlModMul
from qualtran.jupyter_tools import execute_notebook
from qualtran.quantum_graph.bloq_counts import get_cbloq_bloq_counts, SympySymbolAllocator
from qualtran.quantum_graph.util_bloqs import Join, Split


def _make_modexp():
    from qualtran.bloq_algos.factoring.mod_exp import ModExp

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
    assert counts[0][0] == 1, 'int state'
    assert counts[1][0] == n_e, 'mod muls'

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

    assert set(counts1) == set(counts2)


def test_intro_notebook():
    execute_notebook('factoring-via-modexp')


def test_ref_notebook():
    execute_notebook('ref-factoring')
