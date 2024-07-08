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
from qualtran.bloqs.basic_gates import CNOT
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
