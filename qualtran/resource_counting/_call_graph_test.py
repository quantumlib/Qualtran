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

from collections import defaultdict
from functools import cached_property
from typing import Dict, Iterable, Optional, Sequence, Tuple

import attrs
import networkx as nx
import pytest
import sympy
from attrs import field, frozen

import qualtran.testing as qlt_testing
from qualtran import Bloq, BloqBuilder, Signature, Soquet, SoquetT
from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.bookkeeping import ArbitraryClifford, Join, Split
from qualtran.resource_counting import (
    BloqCountDictT,
    BloqCountT,
    get_bloq_call_graph,
    get_bloq_callee_counts,
    MutableBloqCountDictT,
    SympySymbolAllocator,
)
from qualtran.resource_counting.generalizers import generalize_rotation_angle
from qualtran.symbolics import SymbolicInt


@frozen
class BigBloq(Bloq):
    bitsize: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(x=self.bitsize)

    def build_call_graph(self, ssa: Optional['SympySymbolAllocator']) -> 'BloqCountDictT':
        return {SubBloq(unrelated_param=0.5): sympy.log(self.bitsize)}


@frozen
class DecompBloq(Bloq):
    bitsize: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(x=self.bitsize)

    def build_composite_bloq(self, bb: 'BloqBuilder', x: 'Soquet') -> Dict[str, 'SoquetT']:
        qs = bb.split(x)
        for i in range(self.bitsize):
            qs[i] = bb.add(SubBloq(unrelated_param=i / 12), q=qs[i])

        return {'x': bb.join(qs)}


@frozen
class SubBloq(Bloq):
    unrelated_param: float

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(q=1)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {TGate(): 3}


def get_big_bloq_counts_graph_1(bloq: Bloq) -> Tuple[nx.DiGraph, Dict[Bloq, SymbolicInt]]:
    ss = SympySymbolAllocator()
    n_c = ss.new_symbol('n_c')

    def generalize(bloq: Bloq) -> Optional[Bloq]:
        if isinstance(bloq, ArbitraryClifford):
            return attrs.evolve(bloq, n=n_c)

        return bloq

    return get_bloq_call_graph(bloq, generalize, ss)


def test_bloq_counts_method():
    graph, sigma = get_big_bloq_counts_graph_1(BigBloq(100))
    assert len(sigma) == 1
    expr = sigma[TGate()]
    assert str(expr) == '3*log(100)'


def test_get_bloq_callee_counts():
    bloq = BigBloq(100)
    callee_counts = get_bloq_callee_counts(bloq)
    assert callee_counts == [(SubBloq(unrelated_param=0.5), sympy.log(100))]

    bloq = DecompBloq(10)
    callee_counts = get_bloq_callee_counts(bloq)
    assert len(callee_counts) == 10 + 2  # 2 for split/join

    bloq = SubBloq(unrelated_param=0.5)
    callee_counts = get_bloq_callee_counts(bloq)
    assert callee_counts == [(TGate(), 3)]

    callee_counts = get_bloq_callee_counts(TGate())
    assert callee_counts == []


def test_bloq_counts_decomp():
    graph, sigma = get_bloq_call_graph(DecompBloq(10))
    assert len(sigma) == 3  # includes split and join
    expr = sigma[TGate()]
    assert str(expr) == '30'

    def generalize(bloq):
        if isinstance(bloq, (Split, Join)):
            return None
        return bloq

    graph, sigma = get_bloq_call_graph(DecompBloq(10), generalize)
    assert len(sigma) == 1
    expr = sigma[TGate()]
    assert str(expr) == '30'


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('call_graph')


def _to_tuple(x: Iterable[BloqCountT]) -> Sequence[BloqCountT]:
    """mypy compatible converter for OnlyCallGraphBloqShim.callees."""
    return tuple(x)


@frozen
class OnlyCallGraphBloqShim(Bloq):
    name: str
    callees: Sequence[BloqCountT] = field(converter=_to_tuple, factory=tuple)

    @property
    def signature(self) -> 'Signature':
        return Signature([])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        counts: 'MutableBloqCountDictT' = defaultdict(int)
        for bloq, count in self.callees:
            counts[bloq] += count
        return counts

    def __str__(self):
        return self.name


def make_diamond_graph():
    c = OnlyCallGraphBloqShim('c')
    b1 = OnlyCallGraphBloqShim('b1', callees=[(c, 1)])
    b2 = OnlyCallGraphBloqShim('b2', callees=[(c, 1)])
    a = OnlyCallGraphBloqShim('a', callees=[(b1, 1), (b2, 1)])

    def combine_bs(bloq):
        if bloq.name.startswith('b'):
            return OnlyCallGraphBloqShim('b', callees=[(c, 1)])
        return bloq

    return a, combine_bs


def test_diamond_graph():
    bloq, _ = make_diamond_graph()
    graph, sigma = bloq.call_graph()
    edgeset = {(n1.name, n2.name, graph.edges[n1, n2]['n']) for n1, n2 in graph.edges}
    sigma = {bloq.name: val for bloq, val in sigma.items()}

    assert edgeset == {('a', 'b1', 1), ('a', 'b2', 1), ('b1', 'c', 1), ('b2', 'c', 1)}
    assert sigma == {'c': 2}


def test_diamond_graph_generalize():
    bloq, combine_bs = make_diamond_graph()
    graph, sigma = bloq.call_graph(generalizer=combine_bs)
    edgeset = {(n1.name, n2.name, graph.edges[n1, n2]['n']) for n1, n2 in graph.edges}
    sigma = {bloq.name: val for bloq, val in sigma.items()}

    assert edgeset == {('a', 'b', 2), ('b', 'c', 1)}
    assert sigma == {'c': 2}


def make_funnel_graph():
    c = OnlyCallGraphBloqShim('c')
    b = OnlyCallGraphBloqShim('b', callees=[(c, 1)])
    a1 = OnlyCallGraphBloqShim('a1', callees=[(b, 1)])
    a2 = OnlyCallGraphBloqShim('a2', callees=[(b, 1)])
    x = OnlyCallGraphBloqShim('x', callees=[(a1, 1), (a2, 1)])

    def combine_as(bloq):
        if bloq.name.startswith('a'):
            return OnlyCallGraphBloqShim('a', callees=[(b, 1)])
        return bloq

    return x, combine_as


def test_funnel_graph():
    bloq, _ = make_funnel_graph()
    graph, sigma = bloq.call_graph()
    edgeset = {(n1.name, n2.name, graph.edges[n1, n2]['n']) for n1, n2 in graph.edges}
    sigma = {bloq.name: val for bloq, val in sigma.items()}

    assert edgeset == {
        ('x', 'a1', 1),
        ('x', 'a2', 1),
        ('a1', 'b', 1),
        ('a2', 'b', 1),
        ('b', 'c', 1),
    }
    assert sigma == {'c': 2}


def test_funnel_graph_generalize():
    bloq, combine_as = make_funnel_graph()
    graph, sigma = bloq.call_graph(generalizer=combine_as)
    edgeset = {(n1.name, n2.name, graph.edges[n1, n2]['n']) for n1, n2 in graph.edges}
    sigma = {bloq.name: val for bloq, val in sigma.items()}

    assert edgeset == {('x', 'a', 2), ('a', 'b', 1), ('b', 'c', 1)}
    assert sigma == {'c': 2}


def test_multiple_overlaping_call_graph_entries():
    b = OnlyCallGraphBloqShim(name='multi', callees=[(TGate(), 1), (TGate(), 2)])
    counts = get_bloq_callee_counts(b)
    assert counts == [(TGate(), 3)]

    b = OnlyCallGraphBloqShim(name='sep', callees=[(TGate(), 1), (TGate().adjoint(), 2)])
    counts = get_bloq_callee_counts(b)
    # TODO: set needed because of https://github.com/quantumlib/Qualtran/issues/845
    assert set(counts) == {(TGate(), 1), (TGate().adjoint(), 2)}

    b = OnlyCallGraphBloqShim(name='for_gen', callees=[(TGate(), 1), (TGate().adjoint(), 2)])
    counts = get_bloq_callee_counts(b, generalizer=generalize_rotation_angle)
    assert counts == [(TGate(), 3)]
