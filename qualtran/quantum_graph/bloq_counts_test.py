from functools import cached_property
from typing import Dict, Optional, Tuple

import attrs
import networkx as nx
import sympy
from attrs import frozen

from qualtran import Bloq, Signature
from qualtran.bloq_algos.basic_gates import TGate
from qualtran.quantum_graph.bloq_counts import get_bloq_counts_graph, SympySymbolAllocator
from qualtran.quantum_graph.util_bloqs import ArbitraryClifford, Join, Split


@frozen
class BigBloq(Bloq):
    bitsize: int

    @cached_property
    def registers(self) -> 'Signature':
        return Signature.build(x=self.bitsize)

    def bloq_counts(self, ssa: SympySymbolAllocator):
        return [(sympy.log(self.bitsize), SubBloq(unrelated_param=0.5))]


@frozen
class DecompBloq(Bloq):
    bitsize: int

    @cached_property
    def registers(self) -> 'Signature':
        return Signature.build(x=self.bitsize)

    def build_composite_bloq(self, bb: 'BloqBuilder', x: 'SoquetT') -> Dict[str, 'SoquetT']:
        qs = bb.split(x)
        for i in range(self.bitsize):
            (qs[i],) = bb.add(SubBloq(unrelated_param=i / 12), q=qs[i])

        return {'x': bb.join(qs)}


@frozen
class SubBloq(Bloq):
    unrelated_param: float

    @cached_property
    def registers(self) -> 'Signature':
        return Signature.build(q=1)

    def bloq_counts(self, ssa: SympySymbolAllocator):
        return [(3, TGate())]


def get_big_bloq_counts_graph_1(bloq: Bloq) -> Tuple[nx.DiGraph, Dict[Bloq, int]]:
    ss = SympySymbolAllocator()
    n_c = ss.new_symbol('n_c')

    def generalize(bloq: Bloq) -> Optional[Bloq]:
        if isinstance(bloq, ArbitraryClifford):
            return attrs.evolve(bloq, n=n_c)

        return bloq

    return get_bloq_counts_graph(bloq, generalize, ss)


def test_bloq_counts_method():
    graph, sigma = get_big_bloq_counts_graph_1(BigBloq(100))
    assert len(sigma) == 1
    expr = sigma[TGate()]
    assert str(expr) == '3*log(100)'


def test_bloq_counts_decomp():
    graph, sigma = get_bloq_counts_graph(DecompBloq(10))
    assert len(sigma) == 3  # includes split and join
    expr = sigma[TGate()]
    assert str(expr) == '30'

    def generalize(bloq):
        if isinstance(bloq, (Split, Join)):
            return None
        return bloq

    graph, sigma = get_bloq_counts_graph(DecompBloq(10), generalize)
    assert len(sigma) == 1
    expr = sigma[TGate()]
    assert str(expr) == '30'
