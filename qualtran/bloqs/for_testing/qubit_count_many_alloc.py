#  Copyright 2025 Google LLC
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
from typing import Dict

import attrs

from qualtran import Bloq, BloqBuilder, Signature, Soquet, SoquetT
from qualtran.bloqs.basic_gates import CNOT, TGate


@attrs.frozen
class TestManyAllocOnce(Bloq):
    """Allocate an ancilla once, and re-use it explicitly

    See qualtran.resource.counting._qubit_counts_test:test_many_alloc
    """

    n: int

    @property
    def signature(self) -> Signature:
        return Signature.build(x=self.n)

    def build_composite_bloq(self, bb: 'BloqBuilder', *, x: 'Soquet') -> dict[str, 'SoquetT']:
        x = bb.split(x)
        anc = bb.allocate()
        for i in range(self.n):
            x[i], anc = bb.add(CNOT(), ctrl=x[i], target=anc)
            anc = bb.add(TGate(), q=anc)
            x[i], anc = bb.add(CNOT(), ctrl=x[i], target=anc)
        bb.free(anc)
        return {'x': bb.join(x)}


@attrs.frozen
class TestManyAllocMany(Bloq):
    """Allocate a new ancilla for each pseudo-subcall.

    See qualtran.resource.counting._qubit_counts_test:test_many_alloc
    """

    n: int

    @property
    def signature(self) -> Signature:
        return Signature.build(x=self.n)

    def build_composite_bloq(self, bb: 'BloqBuilder', *, x: 'Soquet') -> dict[str, 'SoquetT']:
        x = bb.split(x)
        for i in range(self.n):
            anc = bb.allocate()
            x[i], anc = bb.add(CNOT(), ctrl=x[i], target=anc)
            anc = bb.add(TGate(), q=anc)
            x[i], anc = bb.add(CNOT(), ctrl=x[i], target=anc)
            bb.free(anc)
        return {'x': bb.join(x)}


@attrs.frozen
class _Inner(Bloq):
    """Inner part, used by `TestManyAllocAbstracted`"""

    @property
    def signature(self) -> Signature:
        return Signature.build(x=1)

    def build_composite_bloq(self, bb: 'BloqBuilder', *, x: 'Soquet') -> Dict[str, 'SoquetT']:
        anc = bb.allocate()
        x, anc = bb.add(CNOT(), ctrl=x, target=anc)
        anc = bb.add(TGate(), q=anc)
        x, anc = bb.add(CNOT(), ctrl=x, target=anc)
        bb.free(anc)
        return {'x': x}


@attrs.frozen
class TestManyAllocAbstracted(Bloq):
    """Factor allocation into subbloq

    See qualtran.resource.counting._qubit_counts_test:test_many_alloc
    """

    n: int

    @property
    def signature(self) -> Signature:
        return Signature.build(x=self.n)

    def build_composite_bloq(self, bb: 'BloqBuilder', *, x: 'Soquet') -> dict[str, 'SoquetT']:
        x = bb.split(x)
        for i in range(self.n):
            x[i] = bb.add(_Inner(), x=x[i])
        return {'x': bb.join(x)}
