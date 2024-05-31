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
from typing import Union

from attrs import frozen
from sympy import Expr

from qualtran import Bloq, QAny, Register, Signature
from qualtran.cirq_interop.t_complexity_protocol import TComplexity


@frozen
class ArbitraryClifford(Bloq):
    """A bloq representing an arbitrary `n`-qubit clifford operation.

    In the surface code architecture, clifford operations are generally considered
    cheaper than non-clifford gates. Each clifford also has roughly the same cost independent
    of what particular operation it is doing.

    You can use this to bloq to represent an arbitrary clifford operation e.g. in bloq_counts
    resource estimates where the details are unimportant for the resource estimation task
    at hand.
    """

    n: Union[int, Expr]

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('x', QAny(bitsize=self.n))])

    def _t_complexity_(self) -> 'TComplexity':
        return TComplexity(clifford=1)

    def __str__(self):
        return f'ArbitraryClifford(n={self.n})'
