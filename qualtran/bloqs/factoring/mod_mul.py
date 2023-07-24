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
from typing import Dict, Optional, Set, Union

import sympy
from attrs import frozen

from qualtran import Bloq, BloqBuilder, Signature, Soquet, SoquetT
from qualtran.bloqs.basic_gates import CSwap
from qualtran.bloqs.factoring.mod_add import CtrlScaleModAdd
from qualtran.drawing import Circle, directional_text_box, WireSymbol
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class CtrlModMul(Bloq):
    """Perform controlled `x *= k mod m` for constant k, m and variable x.

    Args:
        k: The integer multiplicative constant.
        mod: The integer modulus.
        bitsize: The size of the `x` register.

    Registers:
     - ctrl: The control bit
     - x: The integer being multiplied
    """

    k: Union[int, sympy.Expr]
    mod: Union[int, sympy.Expr]
    bitsize: Union[int, sympy.Expr]

    def __attrs_post_init__(self):
        if isinstance(self.k, sympy.Expr):
            return
        if isinstance(self.mod, sympy.Expr):
            return

        assert self.k < self.mod

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(ctrl=1, x=self.bitsize)

    def _Add(self, k: Union[int, sympy.Expr]):
        """Helper method to forward attributes to `CtrlScaleModAdd`."""
        return CtrlScaleModAdd(k=k, bitsize=self.bitsize, mod=self.mod)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrl: 'SoquetT', x: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        k = self.k
        neg_k_inv = -pow(k, -1, mod=self.mod)

        # We store the result of the CtrlScaleModAdd into this new register
        # and then clear the original `x` register by multiplying in the inverse.
        y = bb.allocate(self.bitsize)

        # y += x*k
        ctrl, x, y = bb.add(self._Add(k=k), ctrl=ctrl, x=x, y=y)
        # x += y * (-k^-1)
        ctrl, y, x = bb.add(self._Add(k=neg_k_inv), ctrl=ctrl, x=y, y=x)

        # y contains the answer and x is empty.
        # In [GE2019], it is asserted that the registers can be swapped via bookkeeping.
        # This is not correct: we do not want to swap the registers if the control bit
        # is not set.
        ctrl, x, y = bb.add(CSwap(self.bitsize), ctrl=ctrl, x=x, y=y)
        bb.free(y)
        return {'ctrl': ctrl, 'x': x}

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set['BloqCountT']:
        if ssa is None:
            raise ValueError(f"{self} requires a SympySymbolAllocator")
        k = ssa.new_symbol('k')
        return {(2, self._Add(k=k)), (1, CSwap(self.bitsize))}

    def on_classical_vals(self, ctrl, x) -> Dict[str, ClassicalValT]:
        if ctrl == 0:
            return {'ctrl': ctrl, 'x': x}

        assert ctrl == 1, ctrl
        return {'ctrl': ctrl, 'x': (x * self.k) % self.mod}

    def short_name(self) -> str:
        return f'x *= {self.k} % {self.mod}'

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name == 'ctrl':
            return Circle(filled=True)
        if soq.reg.name == 'x':
            return directional_text_box(f'*={self.k}', side=soq.reg.side)
