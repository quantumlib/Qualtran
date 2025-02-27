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
import math
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
    DecomposeTypeError,
    QUInt,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran._infra.registers import Side
from qualtran.bloqs.basic_gates.z_basis import IntState
from qualtran.bloqs.mod_arithmetic import CModMulK
from qualtran.drawing import Text, WireSymbol
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.symbolics import is_symbolic
from qualtran.symbolics.types import SymbolicInt


@frozen
class ModExp(Bloq):
    r"""Perform $b^e \mod{m}$ for constant `base` $b$, `mod` $m$, and quantum `exponent` $e$.

    Modular exponentiation is the main computational primitive for quantum factoring algorithms.
    We follow [GE2019]'s "reference implementation" for factoring. See `ModExp.make_for_shor`
    to set the class attributes for a factoring run.

    This bloq decomposes into controlled modular exponentiation for each exponent bit.

    Args:
        base: The integer base of the exponentiation
        mod: The integer modulus
        exp_bitsize: The size of the `exponent` thru-register
        x_bitsize: The size of the `x` right-register

    Registers:
        exponent: The exponent
        x [right]: The output register containing the result of the exponentiation

    References:
        [How to factor 2048 bit RSA integers in 8 hours using 20 million noisy qubits](https://arxiv.org/abs/1905.09749).
        Gidney and Ekerå. 2019.
    """

    base: 'SymbolicInt'
    mod: 'SymbolicInt'
    exp_bitsize: 'SymbolicInt'
    x_bitsize: 'SymbolicInt'

    def __attrs_post_init__(self):
        if not is_symbolic(self.base, self.mod):
            assert math.gcd(cast(int, self.base), cast(int, self.mod)) == 1

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('exponent', QUInt(self.exp_bitsize)),
                Register('x', QUInt(self.x_bitsize), side=Side.RIGHT),
            ]
        )

    @classmethod
    def make_for_shor(
        cls,
        big_n: 'SymbolicInt',
        g: Optional['SymbolicInt'] = None,
        rs: Optional[np.random.RandomState] = None,
    ):
        """Factory method that sets up the modular exponentiation for a factoring run.

        Args:
            big_n: The large composite number N. Used to set `mod`. Its bitsize is used
                to set `x_bitsize` and `exp_bitsize`.
            g: Optional base of the exponentiation. If `None`, we pick a random base.
            rs: Optional random state which can be seeded to make base generation deterministic.
        """
        if is_symbolic(big_n):
            little_n = sympy.ceiling(sympy.log(big_n, 2))
        else:
            little_n = int(math.ceil(math.log2(big_n)))
        if g is None:
            if is_symbolic(big_n):
                g = sympy.symbols('g')
            else:
                if rs is None:
                    rs = np.random.RandomState()
                while True:
                    g = rs.randint(2, int(big_n))
                    if math.gcd(g, int(big_n)) == 1:
                        break
        return cls(base=g, mod=big_n, exp_bitsize=2 * little_n, x_bitsize=little_n)

    def _CtrlModMul(self, k: 'SymbolicInt'):
        """Helper method to return a `CModMulK` with attributes forwarded."""
        return CModMulK(QUInt(self.x_bitsize), k=k, mod=self.mod)

    def build_composite_bloq(self, bb: 'BloqBuilder', exponent: 'Soquet') -> Dict[str, 'SoquetT']:
        if is_symbolic(self.exp_bitsize):
            raise DecomposeTypeError(f"Cannot decompose {self} with symbolic `exp_bitsize`.")
        # https://en.wikipedia.org/wiki/Modular_exponentiation#Right-to-left_binary_method
        x = bb.add(IntState(val=1, bitsize=self.x_bitsize))
        exponent = bb.split(exponent)

        base = self.base % self.mod
        for j in range(self.exp_bitsize - 1, 0 - 1, -1):
            exponent[j], x = bb.add(self._CtrlModMul(k=base), ctrl=exponent[j], x=x)
            base = (base * base) % self.mod

        return {'exponent': bb.join(exponent, dtype=QUInt(self.exp_bitsize)), 'x': x}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        k = ssa.new_symbol('k')
        return {self._CtrlModMul(k=k): self.exp_bitsize, IntState(val=1, bitsize=self.x_bitsize): 1}

    def on_classical_vals(self, exponent) -> Dict[str, Union['ClassicalValT', sympy.Expr]]:
        return {'exponent': exponent, 'x': (self.base**exponent) % self.mod}

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text(f'{self.base}^e % {self.mod}')
        return super().wire_symbol(reg, idx)


_K = sympy.Symbol('k_exp')


def _generalize_k(b: Bloq) -> Optional[Bloq]:
    if isinstance(b, CModMulK):
        return attrs.evolve(b, k=_K)

    return b


@bloq_example(generalizer=(ignore_split_join, _generalize_k))
def _modexp_small() -> ModExp:
    """A small-exponent modular exponentiation demo."""
    modexp_small = ModExp(base=4, mod=15, exp_bitsize=3, x_bitsize=2048)
    return modexp_small


@bloq_example(generalizer=(ignore_split_join, _generalize_k))
def _modexp() -> ModExp:
    """An example modular exponentiation to factor 13 * 17."""
    modexp = ModExp.make_for_shor(big_n=13 * 17, g=9)
    return modexp


@bloq_example
def _modexp_symb() -> ModExp:
    """Modular exponentiation with symbolic attributes."""
    g, N, n_e, n_x = sympy.symbols('g N n_e, n_x')
    modexp_symb = ModExp(base=g, mod=N, exp_bitsize=n_e, x_bitsize=n_x)
    return modexp_symb


_RSA_MODEXP_DOC = BloqDocSpec(bloq_cls=ModExp, examples=(_modexp_small, _modexp, _modexp_symb))
