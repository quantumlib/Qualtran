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
from typing import Union

from attrs import field, frozen, validators


@frozen
class MagicCount:
    """A count of magic states.

    Attributes:
        n_t: The number of T states.
        n_ccz: The number of CCZ states.
    """

    n_t: float = field(
        default=0.0, converter=float, repr=lambda x: f'{x:g}', validator=validators.ge(0)
    )
    n_ccz: float = field(
        default=0.0, converter=float, repr=lambda x: f'{x:g}', validator=validators.ge(0)
    )

    def __add__(self, other: 'MagicCount') -> 'MagicCount':
        return MagicCount(n_t=self.n_t + other.n_t, n_ccz=self.n_ccz + other.n_ccz)

    def __mul__(self, other: Union[float, int]) -> 'MagicCount':
        return MagicCount(n_t=self.n_t * other, n_ccz=self.n_ccz * other)

    def __rmul__(self, other: Union[float, int]) -> 'MagicCount':
        return self.__mul__(other)
