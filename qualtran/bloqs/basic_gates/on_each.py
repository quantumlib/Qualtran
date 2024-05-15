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

"""Classes to apply single qubit bloq to multiple qubits."""
from functools import cached_property
from typing import cast, Dict, Optional, Set, Tuple

import attrs
import sympy

from qualtran import Bloq, BloqBuilder, QAny, Register, Signature, Soquet, SoquetT
from qualtran.drawing import Text, WireSymbol
from qualtran.drawing.musical_score import TextBox
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
from qualtran.symbolics import SymbolicInt


@attrs.frozen
class OnEach(Bloq):
    """Add a single-qubit (unparameterized) bloq on each of n qubits.

    Args:
        n: the number of qubits to add the bloq to.
        gate: A single qubit gate. The single qubit register must be named q.

    Registers:
     - q: an n-qubit register.
    """

    n: SymbolicInt
    gate: Bloq

    def __attrs_post_init__(self):
        assert len(self.gate.signature) == 1, "Gate must only have a single register."
        assert self.gate.signature[0].bitsize == 1, "Must be single qubit gate."
        assert self.gate.signature[0].name == 'q', "Register must be named q."

    @cached_property
    def signature(self) -> Signature:
        reg = Register('q', QAny(bitsize=self.n))
        return Signature([reg])

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> WireSymbol:
        if reg is None:
            return Text(rf'{self.gate.wire_symbol(reg=None)}⨂{self.n}')
        return TextBox(cast(Text, self.gate.wire_symbol(reg=None)).text)

    def build_composite_bloq(self, bb: BloqBuilder, *, q: Soquet) -> Dict[str, SoquetT]:
        if isinstance(self.n, sympy.Expr):
            raise ValueError(f'Symbolic n not allowed {self.n}')
        qs = bb.split(q)
        for i in range(self.n):
            qs[i] = bb.add(self.gate, q=qs[i])
        return {'q': bb.join(qs)}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(self.gate, self.n)}
