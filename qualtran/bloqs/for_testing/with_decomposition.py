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
from typing import Dict

from attrs import frozen

from qualtran import Bloq, Signature, Soquet
from qualtran.bloqs.for_testing.atom import TestAtom


@frozen
class TestSerialCombo(Bloq):
    """Made up of three bloqs in serial order."""

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(reg=1)

    def build_composite_bloq(self, bb: 'BloqBuilder', reg: 'SoquetT') -> Dict[str, 'Soquet']:
        for i in range(3):
            reg = bb.add(TestAtom(tag=f'atom{i}'), q=reg)
        return {'reg': reg}


@frozen
class TestParallelCombo(Bloq):
    """Made up of three bloqs that happen in parallel."""

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(reg=3)

    def build_composite_bloq(self, bb: 'BloqBuilder', reg: 'SoquetT') -> Dict[str, 'Soquet']:
        reg = bb.split(reg)
        for i in range(len(reg)):
            reg[i] = bb.add(TestAtom(), q=reg[i])

        return {'reg': bb.join(reg)}
