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
from collections import defaultdict
from functools import cached_property
from typing import Set

from attrs import frozen

from qualtran import Bloq, QBit, QUInt, Register, Signature, Soquet
from qualtran.bloqs.arithmetic import Add, AddK
from qualtran.bloqs.arithmetic._shims import CHalf, Lt, MultiCToffoli, Negate, Sub
from qualtran.bloqs.basic_gates import CNOT, CSwap, Swap, Toffoli
from qualtran.drawing import Circle, TextBox, WireSymbol
from qualtran.resource_counting.symbolic_counting_utils import log2


@frozen
class ModAdd(Bloq):
    n: int
    mod: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QUInt(self.n)), Register('y', QUInt(self.n))])


@frozen
class ModSub(Bloq):
    n: int
    mod: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QUInt(self.n)), Register('y', QUInt(self.n))])


@frozen
class CModSub(Bloq):
    n: int
    mod: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [Register('ctrl', QBit()), Register('x', QUInt(self.n)), Register('y', QUInt(self.n))]
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Roetteler
        return {(Toffoli(), 16 * self.n * log2(self.n) - 23.8 * self.n)}

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name == 'ctrl':
            return Circle()
        elif soq.reg.name == 'x':
            return TextBox('x')
        elif soq.reg.name == 'y':
            return TextBox('x-y')

    def __str__(self):
        return self.__class__.__name__


@frozen
class CModAdd(Bloq):
    n: int
    mod: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [Register('ctrl', QBit()), Register('x', QUInt(self.n)), Register('y', QUInt(self.n))]
        )


@frozen
class _ModInvInner(Bloq):
    n: int
    mod: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QUInt(self.n)), Register('out', QUInt(self.n))])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        listing = [
            (MultiCToffoli(self.n + 1), 1),
            (CNOT(), 1),
            (Toffoli(), 1),
            (MultiCToffoli(n=3), 1),
            (CNOT(), 2),
            (Lt(self.n), 1),
            (CSwap(self.n), 2),
            (Sub(self.n), 1),
            (Add(QUInt(self.n)), 1),
            (CNOT(), 1),
            (ModDbl(self.n, self.mod), 1),
            (CHalf(self.n), 1),
            (CSwap(self.n), 2),
            (CNOT(), 1),
        ]
        summer = defaultdict(lambda: 0)
        for bloq, n in listing:
            summer[bloq] += n
        return set(summer.items())

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name == 'x':
            return TextBox('x')
        elif soq.reg.name == 'out':
            return TextBox('$x^{-1}$')

    def __str__(self):
        return self.__class__.__name__


@frozen
class ModInv(Bloq):
    n: int
    mod: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QUInt(self.n)), Register('out', QUInt(self.n))])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Roetteler
        # return {(Toffoli(), 32 * self.n**2 * log2(self.n))}
        return {
            (_ModInvInner(n=self.n, mod=self.mod), 2 * self.n),
            (Negate(self.n), 1),
            (AddK(self.n, k=self.mod), 1),
            (Swap(self.n), 1),
        }

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name == 'x':
            return TextBox('x')
        elif soq.reg.name == 'out':
            return TextBox('$x^{-1}$')

    def __str__(self):
        return self.__class__.__name__


@frozen
class ModMul(Bloq):
    n: int
    mod: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('x', QUInt(self.n)),
                Register('y', QUInt(self.n)),
                Register('out', QUInt(self.n)),
            ]
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Roetteler montgomery
        return {(Toffoli(), 16 * self.n**2 * log2(self.n) - 26.3 * self.n**2)}

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name in ['x', 'y']:
            return TextBox(soq.reg.name)
        elif soq.reg.name == 'out':
            return TextBox('x*y')

    def __str__(self):
        return self.__class__.__name__


@frozen
class ModDbl(Bloq):
    n: int
    mod: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QUInt(self.n)), Register('out', QUInt(self.n))])

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name == 'x':
            return TextBox('x')
        elif soq.reg.name == 'out':
            return TextBox('$2x$')

    def __str__(self):
        return self.__class__.__name__


@frozen
class ModNeg(Bloq):
    n: int
    mod: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QUInt(self.n))])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # litinski
        return {
            (MultiCToffoli(self.n), 2),
            (CNOT(), self.n),
            (AddK(self.n, k=self.mod).controlled(), 1),
        }

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name == 'x':
            return TextBox('$-x$')

    def __str__(self):
        return self.__class__.__name__


@frozen
class CModNeg(Bloq):
    n: int
    mod: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('ctrl', QBit()), Register('x', QUInt(self.n))])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Roetteler
        return {(Toffoli(), 8 * self.n * log2(self.n) - 14.5 * self.n)}

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name == 'ctrl':
            return Circle()
        elif soq.reg.name == 'x':
            return TextBox('$-x$')

    def __str__(self):
        return self.__class__.__name__
