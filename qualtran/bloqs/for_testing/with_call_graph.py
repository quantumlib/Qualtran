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
from typing import TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, Signature
from qualtran.bloqs.for_testing.atom import TestAtom
from qualtran.bloqs.for_testing.with_decomposition import TestParallelCombo, TestSerialCombo

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@frozen
class TestBloqWithCallGraph(Bloq):
    """A bloq that declares its callees (only)."""

    @cached_property
    def signature(self) -> Signature:
        return Signature.build()

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        n = ssa.new_symbol('n')
        return {TestParallelCombo(): 1, TestSerialCombo(): 1, TestAtom(): n}
