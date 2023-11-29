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

import attrs
import sympy
from attrs import frozen

from qualtran import Bloq, bloq_example, BloqBuilder, BloqDocSpec, Signature, Soquet, SoquetT
from qualtran.bloqs.basic_gates import CSwap
from qualtran.bloqs.factoring.mod_add import CtrlScaleModAdd
from qualtran.drawing import Circle, directional_text_box, WireSymbol
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join
from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class CtrlModMul(Bloq):
    """Perform controlled `x *= k mod m` for constant k, m and variable x.

    Args:
        k: The integer multiplicative constant.
        mod: The integer modulus.
        bitsize: The size of the `x` register.

    Registers:
        ctrl: The control bit
        x: The integer being multiplied
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

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        k = ssa.new_symbol('k')
        return {(self._Add(k=k), 2), (CSwap(self.bitsize), 1)}

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


_K = sympy.Symbol('k_mul')


def _generalize_k(b: Bloq) -> Optional[Bloq]:
    if isinstance(b, CtrlScaleModAdd):
        return attrs.evolve(b, k=_K)

    return b


@bloq_example(generalizer=(ignore_split_join, ignore_alloc_free, _generalize_k))
def _modmul() -> CtrlModMul:
    modmul = CtrlModMul(k=123, mod=13 * 17, bitsize=8)
    return modmul


@bloq_example(generalizer=(ignore_split_join, ignore_alloc_free, _generalize_k))
def _modmul_symb() -> CtrlModMul:
    import sympy

    k, N, n_x = sympy.symbols('k N n_x')
    modmul_symb = CtrlModMul(k=k, mod=N, bitsize=n_x)
    return modmul_symb


_MODMUL_DOC = BloqDocSpec(
    bloq_cls=CtrlModMul,
    import_line='from qualtran.bloqs.factoring.mod_mul import CtrlModMul',
    examples=(_modmul_symb, _modmul),
)
