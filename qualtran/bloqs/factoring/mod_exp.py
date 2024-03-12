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
    Side,
    Signature,
    SoquetT,
)
from qualtran.bloqs.basic_gates import IntState
from qualtran.bloqs.factoring.mod_mul import CtrlModMul
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
from qualtran.resource_counting.generalizers import ignore_split_join


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
        [GE2019] How to factor 2048 bit RSA integers in 8 hours using 20 million noisy qubits.
        [arxiv:1905.09749](https://arxiv.org/abs/1905.09749). Gidney and Ekerå. 2019.
    """

    base: Union[int, sympy.Expr]
    mod: Union[int, sympy.Expr]
    exp_bitsize: Union[int, sympy.Expr]
    x_bitsize: Union[int, sympy.Expr]

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('exponent', QUInt(self.exp_bitsize)),
                Register('x', QUInt(self.x_bitsize), side=Side.RIGHT),
            ]
        )

    @classmethod
    def make_for_shor(cls, big_n: int, g: Optional[int] = None):
        """Factory method that sets up the modular exponentiation for a factoring run.

        Args:
            big_n: The large composite number N. Used to set `mod`. Its bitsize is used
                to set `x_bitsize` and `exp_bitsize`.
            g: Optional base of the exponentiation. If `None`, we pick a random base.
        """
        if isinstance(big_n, sympy.Expr):
            little_n = sympy.ceiling(sympy.log(big_n, 2))
        else:
            little_n = int(np.ceil(np.log2(big_n)))
        if g is None:
            g = np.random.randint(big_n)
        return cls(base=g, mod=big_n, exp_bitsize=2 * little_n, x_bitsize=little_n)

    def _CtrlModMul(self, k: Union[int, sympy.Expr]):
        """Helper method to return a `CtrlModMul` with attributes forwarded."""
        return CtrlModMul(k=k, bitsize=self.x_bitsize, mod=self.mod)

    def build_composite_bloq(self, bb: 'BloqBuilder', exponent: 'SoquetT') -> Dict[str, 'SoquetT']:
        if isinstance(self.exp_bitsize, sympy.Expr):
            raise DecomposeTypeError("`exp_bitsize` must be a concrete value.")
        x = bb.add(IntState(val=1, bitsize=self.x_bitsize))
        exponent = bb.split(exponent)

        # https://en.wikipedia.org/wiki/Modular_exponentiation#Right-to-left_binary_method
        base = self.base
        for j in range(self.exp_bitsize - 1, 0 - 1, -1):
            exponent[j], x = bb.add(self._CtrlModMul(k=base), ctrl=exponent[j], x=x)
            base = base * base % self.mod

        return {'exponent': bb.join(exponent), 'x': x}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        k = ssa.new_symbol('k')
        return {
            (IntState(val=1, bitsize=self.x_bitsize), 1),
            (self._CtrlModMul(k=k), self.exp_bitsize),
        }

    def on_classical_vals(self, exponent: int):
        return {'exponent': exponent, 'x': (self.base**exponent) % self.mod}

    def short_name(self) -> str:
        return f'{self.base}^e % {self.mod}'


_K = sympy.Symbol('k_exp')


def _generalize_k(b: Bloq) -> Optional[Bloq]:
    if isinstance(b, CtrlModMul):
        return attrs.evolve(b, k=_K)

    return b


@bloq_example(generalizer=(ignore_split_join, _generalize_k))
def _modexp_small() -> ModExp:
    modexp_small = ModExp(base=3, mod=15, exp_bitsize=3, x_bitsize=2048)
    return modexp_small


@bloq_example(generalizer=(ignore_split_join, _generalize_k))
def _modexp() -> ModExp:
    modexp = ModExp.make_for_shor(big_n=15 * 17, g=9)
    return modexp


@bloq_example
def _modexp_symb() -> ModExp:
    g, N, n_e, n_x = sympy.symbols('g N n_e, n_x')
    modexp_symb = ModExp(base=g, mod=N, exp_bitsize=n_e, x_bitsize=n_x)
    return modexp_symb


_MODEXP_DOC = BloqDocSpec(
    bloq_cls=ModExp,
    import_line='from qualtran.bloqs.factoring.mod_exp import ModExp',
    examples=(_modexp_symb, _modexp_small, _modexp),
)
