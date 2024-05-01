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
from typing import Any, Sequence, Set, Tuple, Union

from attrs import field, frozen

from qualtran import Bloq, Signature
from qualtran.resource_counting import BloqCountT, CostKey, SympySymbolAllocator


@frozen
class CostingBloq(Bloq):
    """A bloq that lets you set the costs via attributes."""

    name: str
    num_qubits: int
    callees: Sequence[BloqCountT] = field(converter=tuple, factory=tuple)
    static_costs: Sequence[Tuple[CostKey, Any]] = field(converter=tuple, factory=tuple)

    @property
    def signature(self) -> 'Signature':
        return Signature.build(register=self.num_qubits)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return set(self.callees)

    def my_static_costs(self, cost_key: 'CostKey'):
        return dict(self.static_costs).get(cost_key, NotImplemented)

    def pretty_name(self):
        return self.name

    def __str__(self):
        return self.name


def make_example_costing_bloqs():
    from qualtran.bloqs.basic_gates import Hadamard, TGate, Toffoli

    func1 = CostingBloq(
        'Func1', num_qubits=10, callees=[(TGate(), 10), (TGate().adjoint(), 10), (Hadamard(), 10)]
    )
    func2 = CostingBloq('Func2', num_qubits=3, callees=[(Toffoli(), 100)])
    algo = CostingBloq('Algo', num_qubits=100, callees=[(func1, 1), (func2, 1)])
    return algo
