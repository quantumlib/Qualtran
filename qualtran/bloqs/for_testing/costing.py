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
from typing import List, Sequence, Set

from attrs import field, frozen

from qualtran import Bloq, Signature
from qualtran.resource_counting import (
    AddCostVal,
    BloqCount,
    BloqCountT,
    CLIFFORD_COST,
    CostKV,
    MaxCostVal,
    MaxQubits,
    MulCostVal,
    SuccessProb,
    SympySymbolAllocator,
)


@frozen
class CostingBloq(Bloq):
    """A bloq that lets you set the costs via attributes."""

    name: str
    num_qubits: int
    callees: Sequence[BloqCountT] = field(converter=tuple, factory=tuple)
    static_costs: Sequence[CostKV] = field(converter=tuple, factory=tuple)
    leaf_costs: Sequence[CostKV] | None = field(
        converter=lambda x: tuple(x) if x is not None else x, default=None
    )

    def signature(self) -> 'Signature':
        return Signature.build(register=self.num_qubits)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return set(self.callees)

    def my_static_costs(self) -> List[CostKV]:
        return [(MaxQubits(), MaxCostVal(self.num_qubits))] + list(self.static_costs)

    def my_leaf_costs(self) -> List[CostKV]:
        if self.leaf_costs is None:
            return [(BloqCount(self), AddCostVal(1)), (MaxQubits(), MaxCostVal(self.num_qubits))]
        return list(self.leaf_costs)

    def pretty_name(self):
        return self.name

    def __str__(self):
        return self.name


def make_example_1() -> Bloq:
    tgate = CostingBloq('TGate', num_qubits=1)
    tof = CostingBloq(
        'Tof', num_qubits=3, callees=[(tgate, 4)], static_costs=[(CLIFFORD_COST, AddCostVal(7))]
    )
    add = CostingBloq('Add', num_qubits=8, callees=[(tof, 8)])
    comp = CostingBloq(
        'Compare', num_qubits=9, callees=[(tof, 8)], static_costs=[(SuccessProb(), MulCostVal(0.9))]
    )
    modadd = CostingBloq('ModAdd', num_qubits=8, callees=[(add, 1), (comp, 2)])
    return modadd
