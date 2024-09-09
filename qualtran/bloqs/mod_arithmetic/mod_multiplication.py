#  Copyright 2024 Google LLC
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

import math
import numbers
from functools import cached_property
from typing import cast, Dict, Optional, Tuple, Union

import attrs
import numpy as np
import sympy
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QBit,
    QMontgomeryUInt,
    QUInt,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic.addition import AddK
from qualtran.bloqs.basic_gates import CNOT, CSwap, XGate
from qualtran.bloqs.mod_arithmetic.mod_addition import CtrlScaleModAdd
from qualtran.drawing import Circle, directional_text_box, Text, WireSymbol
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.symbolics import is_symbolic


@frozen
class ModDbl(Bloq):
    r"""An n-bit modular doubling gate.

    Implements $\ket{x} \rightarrow \ket{2x \mod p}$ using $2n$ Toffoli gates.

    Args:
        dtype: Dtype of the number to double.
        p: The modulus for the doubling.

    Registers:
        x: The register containing the number to double.

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585)
        Fig 6d and 8
    """

    dtype: Union[QUInt, QMontgomeryUInt]
    mod: int = attrs.field()

    @mod.validator
    def _validate_mod(self, attribute, value):
        assert isinstance(value, numbers.Integral) or is_symbolic(value)
        if isinstance(value, numbers.Integral):
            assert value % 2 == 1

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', self.dtype)])

    def on_classical_vals(self, x: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        if x < self.mod:
            x = (x + x) % self.mod
        return {'x': x}

    def build_composite_bloq(self, bb: 'BloqBuilder', x: Soquet) -> Dict[str, 'SoquetT']:
        # Allocate ancilla bits for sign and double.
        lower_bit = bb.allocate(n=1)
        sign = bb.allocate(n=1)

        # Convert x to an n + 2-bit integer by attaching two |0⟩ qubits as the least and most
        # significant bits.
        x_split = bb.split(x)
        x = bb.join(
            np.concatenate([[sign], x_split, [lower_bit]]),
            dtype=attrs.evolve(self.dtype, bitsize=self.dtype.bitsize + 2),
        )

        # Add constant -p to the x register.
        x = bb.add(AddK(bitsize=self.dtype.bitsize + 2, k=-self.mod, signed=False), x=x)

        # Split the three bit pieces again so that we can use the sign to control our constant
        # addition circuit.
        x_split = bb.split(x)
        sign = x_split[0]
        x = bb.join(x_split[1:], dtype=attrs.evolve(self.dtype, bitsize=self.dtype.bitsize + 1))

        # Add constant p to the x register if the result of the last modular reduction is negative.
        (sign,), x = bb.add(
            AddK(bitsize=self.dtype.bitsize + 1, k=self.mod, signed=False, cvs=(1,)),
            ctrls=(sign,),
            x=x,
        )

        # Split the lower bit ancilla from the x register for use in resetting the other ancilla bit
        # before freeing them both.
        x_split = bb.split(x)
        lower_bit = x_split[-1]
        lower_bit = bb.add(XGate(), q=lower_bit)
        lower_bit, sign = bb.add(CNOT(), ctrl=lower_bit, target=sign)
        lower_bit = bb.add(XGate(), q=lower_bit)

        free_bit = x_split[0]
        x = bb.join(np.concatenate([x_split[1:-1], [lower_bit]]), dtype=self.dtype)

        # Free the ancilla bits.
        bb.free(free_bit)
        bb.free(sign)

        # Return the output registers.
        return {'x': x}

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text(f'x = 2 * x mod {self.mod}')
        return super().wire_symbol(reg, idx)

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        return {
            AddK(self.dtype.bitsize + 2, -self.mod, signed=False): 1,
            AddK(self.dtype.bitsize + 1, self.mod, cvs=(1,), signed=False): 1,
            CNOT(): 1,
            XGate(): 2,
        }


@bloq_example
def _moddbl_small() -> ModDbl:
    moddbl_small = ModDbl(QUInt(4), 13)
    return moddbl_small


@bloq_example
def _moddbl_large() -> ModDbl:
    prime = 10**9 + 7
    moddbl_large = ModDbl(QUInt(32), prime)
    return moddbl_large


_MOD_DBL_DOC = BloqDocSpec(bloq_cls=ModDbl, examples=[_moddbl_small, _moddbl_large])


@frozen
class CModMulK(Bloq):
    r"""Perform controlled modular multiplication by a constant.

    Applies $\ket{c}\ket{c} \rightarrow \ket{c} \ket{x*k^c \mod p}$.

    Args:
        dtype: Dtype of the register.
        k: The integer multiplicative constant.
        mod: The integer modulus.

    Registers:
        ctrl: The control bit
        x: The integer being multiplied
    """

    dtype: Union[QUInt, QMontgomeryUInt]
    k: Union[int, sympy.Expr]
    mod: Union[int, sympy.Expr]

    def __attrs_post_init__(self):
        if is_symbolic(self.k, self.mod):
            return
        assert 0 < self.k < self.mod
        assert math.gcd(cast(int, self.k), cast(int, self.mod)) == 1

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('ctrl', QBit()), Register('x', self.dtype)])

    def _Add(self, k: Union[int, sympy.Expr]):
        """Helper method to forward attributes to `CtrlScaleModAdd`."""
        return CtrlScaleModAdd(k=k, bitsize=self.dtype.bitsize, mod=self.mod)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrl: 'SoquetT', x: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        k = self.k
        if isinstance(self.mod, sympy.Expr) or isinstance(k, sympy.Expr):
            neg_k_inv = sympy.Mod(sympy.Pow(k, -1), self.mod)
        else:
            neg_k_inv = -pow(k, -1, mod=self.mod)

        # We store the result of the CtrlScaleModAdd into this new register
        # and then clear the original `x` register by multiplying in the inverse.
        y = bb.allocate(self.dtype.bitsize)

        # y += x*k
        ctrl, x, y = bb.add(self._Add(k=k), ctrl=ctrl, x=x, y=y)
        # x += y * (-k^-1)
        ctrl, y, x = bb.add(self._Add(k=neg_k_inv), ctrl=ctrl, x=y, y=x)

        # y contains the answer and x is empty.
        # In [GE2019], it is asserted that the registers can be swapped via bookkeeping.
        # This is not correct: we do not want to swap the registers if the control bit
        # is not set.
        ctrl, x, y = bb.add(CSwap(self.dtype.bitsize), ctrl=ctrl, x=x, y=y)
        bb.free(y)
        return {'ctrl': ctrl, 'x': x}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        k = ssa.new_symbol('k')
        return {self._Add(k=k): 2, CSwap(self.dtype.bitsize): 1}

    def on_classical_vals(self, ctrl, x) -> Dict[str, ClassicalValT]:
        if ctrl and x < self.mod:
            return {'ctrl': ctrl, 'x': (x * self.k) % self.mod}
        return {'ctrl': ctrl, 'x': x}

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text(f'x *= {self.k} % {self.mod}')
        if reg.name == 'ctrl':
            return Circle(filled=True)
        if reg.name == 'x':
            return directional_text_box(f'*={self.k}', side=reg.side)
        raise ValueError(f"Unknown register name: {reg.name}")


_K = sympy.Symbol('k_mul')


def _generalize_k(b: Bloq) -> Optional[Bloq]:
    if isinstance(b, CtrlScaleModAdd):
        return attrs.evolve(b, k=_K)

    return b


@bloq_example(generalizer=(ignore_split_join, ignore_alloc_free, _generalize_k))
def _modmul() -> CModMulK:
    modmul = CModMulK(QUInt(8), k=123, mod=13 * 17)
    return modmul


@bloq_example(generalizer=(ignore_split_join, ignore_alloc_free, _generalize_k))
def _modmul_symb() -> CModMulK:
    import sympy

    k, N, n_x = sympy.symbols('k N n_x')
    modmul_symb = CModMulK(QUInt(n_x), k=k, mod=N)
    return modmul_symb


_C_MOD_MUL_K_DOC = BloqDocSpec(bloq_cls=CModMulK, examples=(_modmul_symb, _modmul))
