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
from typing import Set

from attrs import frozen

from qualtran import Bloq, QBit, QUInt, Register, Signature
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class MultiCToffoli(Bloq):
    n: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('ctrl', QBit(), shape=(self.n,)), Register('target', QBit())])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(Toffoli(), self.n - 2)}


@frozen
class AddK(Bloq):
    n: int
    k: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QUInt(self.n))])


@frozen
class Sub(Bloq):
    n: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QUInt(self.n)), Register('y', QUInt(self.n))])


@frozen
class Lt(Bloq):
    n: int
    signed: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [Register('x', QUInt(self.n)), Register('y', QUInt(self.n)), Register('out', QBit())]
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # litinski
        return {(Toffoli(), self.n)}


@frozen
class CHalf(Bloq):
    n: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('ctrl', QBit()), Register('x', QUInt(self.n))])


@frozen
class Negate(Bloq):
    n: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QUInt(self.n))])
