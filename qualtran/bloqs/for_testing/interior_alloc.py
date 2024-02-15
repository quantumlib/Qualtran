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
from typing import Dict, Union

import sympy
from attrs import frozen

from qualtran import Bloq, BloqBuilder, Signature, Soquet, SoquetT
from qualtran.bloqs.basic_gates.swap import Swap


@frozen
class InteriorAlloc(Bloq):
    """A bloq that performs an allocation and de-allocation on an interior wire.

    This means the maximum number of qubits used is larger than the sum of the register bitsizes.
    """

    n: Union[int, sympy.Symbol]

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(x=self.n, y=self.n)

    def build_composite_bloq(self, bb: 'BloqBuilder', x: Soquet, y: Soquet) -> Dict[str, 'SoquetT']:
        middle = bb.allocate(self.n)
        x, middle = bb.add(Swap(self.n), x=x, y=middle)
        middle, y = bb.add(Swap(self.n), x=middle, y=y)
        bb.free(middle)
        return {'x': x, 'y': y}
