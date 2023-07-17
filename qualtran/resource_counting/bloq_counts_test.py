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
from typing import Dict, Optional, Set, Tuple

import attrs
import networkx as nx
import sympy
from attrs import frozen

from qualtran import Bloq, BloqBuilder, Signature, SoquetT
from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.util_bloqs import ArbitraryClifford, Join, Split
from qualtran.resource_counting import BloqCountT, get_bloq_counts_graph, SympySymbolAllocator


@frozen
class BigBloq(Bloq):
    bitsize: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(x=self.bitsize)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set['BloqCountT']:
        return {(sympy.log(self.bitsize), SubBloq(unrelated_param=0.5))}


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

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set['BloqCountT']:
        return {(3, TGate())}


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
