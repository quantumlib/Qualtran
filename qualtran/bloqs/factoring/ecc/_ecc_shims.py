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

from attrs import frozen

from qualtran import Bloq, DecomposeTypeError, QBit, Register, Side, Signature, Soquet
from qualtran.drawing import RarrowTextBox, WireSymbol


@frozen
class MeasureQFT(Bloq):
    n: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QBit(), shape=(self.n,), side=Side.LEFT)])

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError('MeasureQFT is a placeholder, atomic bloq.')

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name == 'x':
            return RarrowTextBox('MeasQFT')

    def __str__(self):
        return 'MeasureQFT'

    def cost_attrs(self):
        return [('n', self.n)]
