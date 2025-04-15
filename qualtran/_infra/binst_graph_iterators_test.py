#  Copyright 2024 Google LLC
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
import sympy
from attrs import frozen

from qualtran import (
    Bloq,
    BloqBuilder,
    BloqInstance,
    LeftDangle,
    QAny,
    QBit,
    RightDangle,
    Signature,
    SoquetT,
)
from qualtran._infra.binst_graph_iterators import greedy_topological_sort
from qualtran.bloqs.basic_gates import CNOT, IntState, Swap
from qualtran.bloqs.bookkeeping import Allocate, Free


@frozen
class MultiAlloc(Bloq):
    rounds: int = 2

    @property
    def signature(self) -> Signature:
        return Signature.build(q=1)

    def build_composite_bloq(self, bb: BloqBuilder, q: SoquetT) -> dict[str, SoquetT]:
        for _ in range(self.rounds):
            a = bb.allocate(1)
            a, q = bb.add(CNOT(), ctrl=a, target=q)
            bb.free(a)

        return {"q": q}


def test_greedy_topological_sort():
    bloq = MultiAlloc()
    binst_graph = bloq.decompose_bloq()._binst_graph
    greedy_toposort = [*greedy_topological_sort(binst_graph)]
    assert greedy_toposort == [
        LeftDangle,
        BloqInstance(bloq=Allocate(dtype=QAny(bitsize=1)), i=0),
        BloqInstance(bloq=CNOT(), i=1),
        BloqInstance(bloq=Free(dtype=QBit()), i=2),
        BloqInstance(bloq=Allocate(dtype=QAny(bitsize=1)), i=3),
        BloqInstance(bloq=CNOT(), i=4),
        BloqInstance(bloq=Free(dtype=QBit()), i=5),
        RightDangle,
    ]


def test_topo_sort_with_symbolic_registers():
    # when _priority returns a symbolic value, networkx will try to use
    # it in a comparison and you would get
    # TypeError: cannot determine truth value of Relational.

    n = sympy.Symbol('n')
    bb = BloqBuilder()

    # This isn't usually a problem for thru-registers since sympy will
    # simplify n-n to zero, which can be compared. Test against a sided
    # symbolic register
    x = bb.add(IntState(5, n))

    y = bb.add_register('y', n)
    x, y = bb.add(Swap(n), x=x, y=y)
    x, y = bb.add(Swap(n), x=x, y=y)
    cbloq = bb.finalize(x=x, y=y)
    res = list(greedy_topological_sort(cbloq._binst_graph))
    assert len(res) > 0
