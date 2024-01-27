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

from functools import cached_property
from typing import Dict, List, Optional, Sequence, Set, Tuple

import attrs
import networkx as nx
import sympy
from attrs import field, frozen

import qualtran.testing as qlt_testing
from qualtran import Bloq, BloqBuilder, Signature, SoquetT
from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.util_bloqs import ArbitraryClifford, Join, Split
from qualtran.resource_counting import BloqCountT, get_bloq_call_graph, SympySymbolAllocator
from qualtran.resource_counting import BloqCount, AddCostVal, CostKV


@frozen
class BigBloq(Bloq):
    bitsize: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(x=self.bitsize)

    def build_call_graph(self, ssa: Optional['SympySymbolAllocator']) -> Set['BloqCountT']:
        return {(SubBloq(unrelated_param=0.5), sympy.log(self.bitsize))}


@frozen
class DecompBloq(Bloq):
    bitsize: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(x=self.bitsize)

    def build_composite_bloq(self, bb: 'BloqBuilder', x: 'SoquetT') -> Dict[str, 'SoquetT']:
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

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(TGate(), 3)}


def get_big_bloq_counts_graph_1(bloq: Bloq) -> Tuple[nx.DiGraph, Dict[Bloq, int]]:
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


def test_notebook():
    qlt_testing.execute_notebook('bloq_counts')


@frozen
class OnlyCallGraphBloqShim(Bloq):
    name: str
    callees: Sequence[BloqCountT] = field(converter=tuple, factory=tuple)

    @property
    def signature(self) -> 'Signature':
        return Signature([])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return set(self.callees)

    def my_static_costs(self) -> List[CostKV]:
        """Anything that you can't deduce by recursion. For example, you shouldn't include
        the T count here because that should be found by diving into the call graph
        until you get to the T bloq and you can add it all up.

        This could be something like success probability or 'overhead'. For examples sake,
        I've moved clifford accounting to an additional cost instead of capturing it in the
        call graph.
        """
        return []

    def my_leaf_costs(self) -> List[CostKV]:
        """The costs if this is a leaf bloq; i.e. sorry: we're not going to deduce costs
        from the call graph for you.

        We'll include my_addtl_costs even if you're a leaf; so don't repeat yourself.

        By default, this records that we do--in fact-- take one bloq count of ourself. So
        if you end on a bloq you know it at least takes one of itself.

        If there are success probabilities or max_width or overhead complications from
        callees, it's lost (unless you override this). Is this a problem? Should we error
        if this is not explicitly overridden and we end up making it a leaf bloq?

        Consider the Tof+T cost model. If we built the big-ol cost model, then we'd just
        get a T count. If you included a toffoli count (naively, e.g. in my_addtl_costs) you'd
        get the 4*tof t-gates "double counted". If your call graph had *only* tof, then
        this would work fine because tof would be a leaf.

        The solution is to change the call-graph generation logic to consider Tof a leaf
        node (as well as T if it shows up in other branches of the DAG).
        """
        return [(BloqCount(self), AddCostVal(1))]

    def pretty_name(self):
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
