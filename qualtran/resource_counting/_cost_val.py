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
import math
from typing import TypeVar

from attrs import frozen


class CostVal(metaclass=abc.ABCMeta):
    """Values that can do val += y * n

    "+=" should combine two costs; "*n" should do the combination n times;
    For additive costs, these are normal addition and multiplication.
    For multiplicitive costs, these would be multiplying and power.
    For "max" costs, this is max() and identity.
    """

    @abc.abstractmethod
    def __mul__(self, other):
        ...

    @abc.abstractmethod
    def __iadd__(self, other):
        ...


CostValT = TypeVar('CostValT', bound=CostVal)


@frozen
class AddCostVal(CostVal):
    qty: int

    def __mul__(self, other: int):
        return AddCostVal(self.qty * other)

    def __iadd__(self, other: 'AddCostVal'):
        return AddCostVal(self.qty + other.qty)

    def __str__(self):
        return f'{self.qty}'


@frozen
class MulCostVal(CostVal):
    val: float

    def __mul__(self, other: int):
        return MulCostVal(self.val**other)

    def __iadd__(self, other: 'MulCostVal'):
        assert isinstance(other, MulCostVal)
        return MulCostVal(self.val * other.val)

    def __str__(self):
        return f'{self.val}'


@frozen
class MaxCostVal(CostVal):
    val: float

    @classmethod
    def minval(cls) -> 'MaxCostVal':
        return MaxCostVal(-math.inf)

    def __mul__(self, other: int):
        # doing repeated "max" is the same as doing one max.
        return MaxCostVal(self.val)

    def __iadd__(self, other: 'MaxCostVal'):
        assert isinstance(other, MaxCostVal)
        return MaxCostVal(max(self.val, other.val))

    def __str__(self):
        return f'{self.val}'
