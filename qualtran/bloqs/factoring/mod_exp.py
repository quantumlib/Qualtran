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

import numpy as np
import sympy
from attrs import frozen

from qualtran import Bloq, BloqBuilder, Register, Side, Signature, SoquetT
from qualtran.bloqs.basic_gates import IntState
from qualtran.bloqs.factoring.mod_mul import CtrlModMul
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


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
     - exponent: The exponent
     - x [right]: The output register containing the result of the exponentiation

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
                Register('exponent', bitsize=self.exp_bitsize),
                Register('x', bitsize=self.x_bitsize, side=Side.RIGHT),
            ]
        )

    @classmethod
    def make_for_shor(cls, big_n: int, g=None):
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
        x = bb.add(IntState(val=1, bitsize=self.x_bitsize))
        exponent = bb.split(exponent)

        # https://en.wikipedia.org/wiki/Modular_exponentiation#Right-to-left_binary_method
        base = self.base
        for j in range(self.exp_bitsize - 1, 0 - 1, -1):
            exponent[j], x = bb.add(self._CtrlModMul(k=base), ctrl=exponent[j], x=x)
            base = base * base % self.mod

        return {'exponent': bb.join(exponent), 'x': x}

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set['BloqCountT']:
        if ssa is None:
            raise ValueError(f"{self} requires a SympySymbolAllocator")
        k = ssa.new_symbol('k')
        return {
            (1, IntState(val=1, bitsize=self.x_bitsize)),
            (self.exp_bitsize, self._CtrlModMul(k=k)),
        }

    def on_classical_vals(self, exponent: int):
        return {'exponent': exponent, 'x': (self.base**exponent) % self.mod}

    def short_name(self) -> str:
        return f'{self.base}^e % {self.mod}'
