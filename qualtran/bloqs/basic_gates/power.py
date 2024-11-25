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
from typing import Dict, TYPE_CHECKING

import numpy as np
from attrs import frozen

from qualtran import Bloq, BloqBuilder, GateWithRegisters, Side, Signature, SoquetT
from qualtran.symbolics import is_symbolic, SymbolicInt

if TYPE_CHECKING:
    import cirq

    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@frozen
class Power(GateWithRegisters):
    """Wrapper that repeats the given `bloq` `power` times.

    `Bloq` must have only THRU registers.

    Args:
        bloq: Bloq to repeat
        power: Number of times to repeat the Bloq

    Registers:
        Same as `self.bloq.signature`
    """

    bloq: Bloq
    power: SymbolicInt

    def __attrs_post_init__(self):
        if any(reg.side != Side.THRU for reg in self.bloq.signature):
            raise ValueError('Bloq to repeat must have only THRU registers')

        if not is_symbolic(self.power) and (
            not isinstance(self.power, (int, np.integer)) or self.power < 1
        ):
            raise ValueError(f'{self.power=} must be a positive integer.')

    def adjoint(self) -> 'Bloq':
        return Power(self.bloq.adjoint(), self.power)

    @cached_property
    def signature(self) -> Signature:
        return self.bloq.signature

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        if not isinstance(self.power, int):
            raise ValueError(f'Symbolic power {self.power} not supported')
        for _ in range(self.power):
            soqs = bb.add_d(self.bloq, **soqs)
        return soqs

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {self.bloq: self.power}

    def __pow__(self, power) -> 'Power':
        bloq = self.bloq.adjoint() if power < 0 else self.bloq
        return Power(bloq, self.power * abs(power))

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        import cirq

        info = cirq.circuit_diagram_info(self.bloq, args, default=None)

        if info is None:
            info = super()._circuit_diagram_info_(args)

        wire_symbols = [f'{symbol}^{self.power}' for symbol in info.wire_symbols]

        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)
