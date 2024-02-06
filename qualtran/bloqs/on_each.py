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
from typing import Dict

import attrs

from qualtran import Bloq, BloqBuilder, GateWithRegisters, QAny, Register, Signature, SoquetT
from qualtran._infra.quantum_graph import Soquet
from qualtran.drawing import WireSymbol
from qualtran.drawing.musical_score import TextBox


@attrs.frozen
class OnEach(GateWithRegisters):
    """Add a single-qubit (unparameterized) bloq on each of n qubits.

    Args:
        n: the number of qubits to add the bloq to.
        gate: A single qubit gate. The single qubit register must be named q.

    Registers:
     - q: an n-qubit register.
    """

    n: int
    gate: Bloq

    def __attrs_post_init__(self):
        assert len(self.gate.signature) == 1, "Gate must only have a single register."
        assert self.gate.signature[0].bitsize == 1, "Must be single qubit gate."
        assert self.gate.signature[0].name == 'q', "Register must be named q."

    @cached_property
    def signature(self) -> Signature:
        reg = Register('q', QAny(bitsize=self.n))
        return Signature([reg])

    def short_name(self) -> str:
        return rf'{self.gate.short_name()}⨂{self.n}'

    def wire_symbol(self, soq: Soquet) -> WireSymbol:
        return TextBox(self.gate.short_name())

    def build_composite_bloq(self, bb: BloqBuilder, *, q: SoquetT) -> Dict[str, SoquetT]:
        qs = bb.split(q)
        for i in range(self.n):
            qs[i] = bb.add(self.gate, q=qs[i])
        return {'q': bb.join(qs)}
