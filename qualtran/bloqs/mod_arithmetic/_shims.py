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
"""This module has a selection of minimally-implemented modular arithmetic primitives.

These bloqs serve as the callees in the call graphs of the algorithms found
in `qualtran.bloq.factoring`. They are place-holders, so we don't have undefined symbols
and can still merge the high-level algorithms. These shims will be fleshed out
and moved to their final organizational location soon (written: 2024-05-06).
"""


from collections import defaultdict
from functools import cached_property
from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, QBit, QUInt, Register, Signature
from qualtran.bloqs.arithmetic import Add, AddK, Subtract
from qualtran.bloqs.arithmetic._shims import CHalf, Lt, MultiCToffoli, Negate
from qualtran.bloqs.basic_gates import CNOT, CSwap, Swap, Toffoli
from qualtran.drawing import Circle, Text, TextBox, WireSymbol
from qualtran.symbolics import ceil, log2

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


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
        return {(Toffoli(), ceil(16 * self.n * log2(self.n) - 23.8 * self.n))}

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text("")
        if reg.name == 'ctrl':
            return Circle()
        elif reg.name == 'x':
            return TextBox('x')
        elif reg.name == 'y':
            return TextBox('x-y')
        raise ValueError(f'Unrecognized register name {reg.name}')

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
        # This listing is based off of Haner 2023, fig 15. The order of operations
        # matches the order in the figure
        listing = [
            (MultiCToffoli(self.n + 1), 1),
            (CNOT(), 1),
            (Toffoli(), 1),
            (MultiCToffoli(n=3), 1),
            (CNOT(), 2),
            (Lt(self.n), 1),
            (CSwap(self.n), 2),
            (Subtract(QUInt(self.n)), 1),
            (Add(QUInt(self.n)), 1),
            (CNOT(), 1),
            (ModDbl(self.n, self.mod), 1),
            (CHalf(self.n), 1),
            (CSwap(self.n), 2),
            (CNOT(), 1),
        ]
        # Since the listing is time-ordered and the call graph protocol expects
        # unique bloq keys, we group counts by bloqs.
        summer: Dict[Bloq, int] = defaultdict(lambda: 0)
        for bloq, n in listing:
            summer[bloq] += n
        return set(summer.items())

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text("")
        if reg.name == 'x':
            return TextBox('x')
        elif reg.name == 'out':
            return TextBox('$x^{-1}$')
        raise ValueError(f'Unrecognized register name {reg.name}')

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

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text("")
        if reg.name == 'x':
            return TextBox('x')
        elif reg.name == 'out':
            return TextBox('$x^{-1}$')
        raise ValueError(f'Unrecognized register name {reg.name}')

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
        return {(Toffoli(), ceil(16 * self.n**2 * log2(self.n) - 26.3 * self.n**2))}

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text("")
        if reg.name in ['x', 'y']:
            return TextBox(reg.name)
        elif reg.name == 'out':
            return TextBox('x*y')
        raise ValueError(f'Unrecognized register name {reg.name}')

    def __str__(self):
        return self.__class__.__name__


@frozen
class ModDbl(Bloq):
    n: int
    mod: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QUInt(self.n)), Register('out', QUInt(self.n))])

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text("")
        if reg.name == 'x':
            return TextBox('x')
        elif reg.name == 'out':
            return TextBox('$2x$')
        raise ValueError(f'Unrecognized register name {reg.name}')

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

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text("")
        if reg.name == 'x':
            return TextBox('$-x$')
        raise ValueError(f'Unrecognized register name {reg.name}')

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
        return {(Toffoli(), ceil(8 * self.n * log2(self.n) - 14.5 * self.n))}

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text("")
        if reg.name == 'ctrl':
            return Circle()
        elif reg.name == 'x':
            return TextBox('$-x$')
        raise ValueError(f'Unrecognized register name {reg.name}')

    def __str__(self):
        return self.__class__.__name__
