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

import abc
from typing import Generic, TYPE_CHECKING

from attrs import frozen

from ._cost_val import AddCostVal, CostValT, MaxCostVal, MulCostVal

if TYPE_CHECKING:
    from qualtran import Bloq


class CostKey(Generic[CostValT], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def identity_val(self) -> CostValT:
        ...


@frozen
class BloqCount(CostKey[AddCostVal]):
    bloq: 'Bloq'

    def identity_val(self) -> AddCostVal:
        return AddCostVal(0)

    def __str__(self):
        return f'{self.bloq} count'


@frozen
class AnyCount(CostKey[AddCostVal]):
    cost_name: str

    def identity_val(self) -> AddCostVal:
        return AddCostVal(0)

    def __str__(self):
        return f'{self.cost_name} count'


CLIFFORD_COST = AnyCount("clifford")


@frozen
class MaxQubits(CostKey[MaxCostVal]):
    """A cost representing the maximum qubits required."""

    def identity_val(self) -> MaxCostVal:
        return MaxCostVal.minval()

    def __str__(self):
        return 'max qubits'


@frozen
class SuccessProb(CostKey[MulCostVal]):
    def identity_val(self) -> MulCostVal:
        return MulCostVal(1.0)

    def __str__(self):
        return 'success prob'
