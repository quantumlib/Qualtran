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
import random
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
from qualtran.bloqs.arithmetic import Add
from qualtran.bloqs.arithmetic.subtraction import SubtractFrom
from qualtran.bloqs.basic_gates.swap import Swap
from qualtran.bloqs.basic_gates.z_basis import IntState
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean
from qualtran.bloqs.mod_arithmetic import CModMulK
from qualtran.bloqs.mod_arithmetic.mod_addition import ModAdd
from qualtran.bloqs.mod_arithmetic.mod_subtraction import ModSub
from qualtran.drawing import Text, WireSymbol
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.symbolics import is_symbolic
from qualtran.symbolics.types import Shaped, SymbolicInt


@frozen
class ModExp(Bloq):
    r"""Perform $b^e \mod{m}$ for constant `base` $b$, `mod` $m$, and quantum `exponent` $e$.

    Modular exponentiation is the main computational primitive for quantum factoring algorithms.
    We follow [GE2019]'s "reference implementation" for factoring. See `ModExp.make_for_shor`
    to set the class attributes for a factoring run.

    This bloq decomposes into controlled modular exponentiation for each exponent bit.

    Args:
        base: The integer base of the exponentiation.
        mod: The integer modulus.
        exp_bitsize: The size of the `exponent` thru-register.
        x_bitsize: The size of the `x` right-register.
        exp_window_size: The window size of windowed arithmetic on the controlled modular
            multiplications.
        mult_window_size: The window size of windowed arithmetic on the modular product additions.

    Registers:
        exponent: The exponent
        x: The output register containing the result of the exponentiation

    References:
        [How to factor 2048 bit RSA integers in 8 hours using 20 million noisy qubits](https://arxiv.org/abs/1905.09749).
        Gidney and Ekerå. 2019.

        [Windowed quantum arithmetic](https://arxiv.org/abs/1905.07682).
        Craig Gidney. 2019.
    """

    base: 'SymbolicInt'
    mod: 'SymbolicInt'
    exp_bitsize: 'SymbolicInt'
    x_bitsize: 'SymbolicInt'
    exp_window_size: Optional['SymbolicInt'] = None
    mult_window_size: Optional['SymbolicInt'] = None

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
    def make_for_shor(cls, big_n: 'SymbolicInt', g: Optional['SymbolicInt'] = None, exp_window_size: Optional['SymbolicInt'] = None, mult_window_size: Optional['SymbolicInt'] = None):
        """Factory method that sets up the modular exponentiation for a factoring run.

        Args:
            big_n: The large composite number N. Used to set `mod`. Its bitsize is used
                to set `x_bitsize` and `exp_bitsize`.
            g: Optional base of the exponentiation. If `None`, we pick a random base.
        """
        if is_symbolic(big_n):
            little_n = sympy.ceiling(sympy.log(big_n, 2))
        else:
            little_n = int(math.ceil(math.log2(big_n)))
        if g is None:
            if is_symbolic(big_n):
                g = sympy.symbols('g')
            else:
                while True:
                    g = random.randint(2, int(big_n))
                    if math.gcd(g, int(big_n)) == 1:
                        break
        return cls(base=g, mod=big_n, exp_bitsize=2 * little_n, x_bitsize=little_n, exp_window_size=exp_window_size, mult_window_size=mult_window_size)

    def qrom(self, data):
        if is_symbolic(self.exp_bitsize) or is_symbolic(self.exp_window_size):
            log_block_sizes = None
            if is_symbolic(self.exp_bitsize) and not is_symbolic(self.exp_window_size):
                # We assume that bitsize is much larger than window_size
                log_block_sizes = (0,)
            return QROAMClean(
                [
                    Shaped((2**(self.exp_window_size+self.mult_window_size),)),
                ],
                selection_bitsizes=(self.exp_window_size, self.mult_window_size),
                target_bitsizes=(self.x_bitsize,),
                log_block_sizes=log_block_sizes,
            )

        return QROAMClean(
            [data],
            selection_bitsizes=(self.exp_window_size, self.mult_window_size),
            target_bitsizes=(self.x_bitsize,),
        )


    def _CtrlModMul(self, k: 'SymbolicInt'):
        """Helper method to return a `CModMulK` with attributes forwarded."""
        return CModMulK(QUInt(self.x_bitsize), k=k, mod=self.mod)

    def build_composite_bloq(self, bb: 'BloqBuilder', exponent: 'Soquet') -> Dict[str, 'SoquetT']:
        if is_symbolic(self.exp_bitsize):
                raise DecomposeTypeError(f"Cannot decompose {self} with symbolic `exp_bitsize`.")
        x = bb.add(IntState(val=1, bitsize=self.x_bitsize))
        exponent = bb.split(exponent)

        if self.exp_window_size is not None and self.mult_window_size is not None:
            k = self.base

            a = bb.split(x)
            b = bb.add(IntState(val=0, bitsize=self.x_bitsize))
            
            ei = np.split(np.array(exponent), self.exp_bitsize // self.exp_window_size)
            for i in range(self.exp_bitsize // self.exp_window_size):
                kes = [pow(k, 2**i * x_e, self.mod) for x_e in range(2**self.exp_window_size)]
                kes_inv = [pow(x_e, -1, self.mod) for x_e in kes]

                mi = np.split(np.array(a), self.x_bitsize // self.mult_window_size)
                for j in range(self.x_bitsize // self.mult_window_size):
                    data = list([(ke * f * 2**j) % self.mod for f in range(2**self.mult_window_size)] for ke in kes)
                    ei_i = bb.join(ei[(self.exp_bitsize // self.exp_window_size) - i - 1], QUInt((self.exp_window_size)))
                    mi_i = bb.join(mi[(self.x_bitsize // self.mult_window_size) - j - 1], QUInt((self.mult_window_size)))
                    ei_i, mi_i, t, *junk = bb.add(self.qrom(data), selection0=ei_i, selection1=mi_i)
                    t, b = bb.add(ModAdd(self.x_bitsize, self.mod), x=t, y=b)
                    junk_mapping = {f'junk_target{i}_': junk[i] for i in range(len(junk))}
                    ei_i, mi_i = bb.add(self.qrom(data).adjoint(), selection0=ei_i, selection1=mi_i, target0_=t, **junk_mapping)
                    ei[(self.exp_bitsize // self.exp_window_size) - i - 1] = bb.split(ei_i)
                    mi[(self.x_bitsize // self.mult_window_size) - j - 1] = bb.split(mi_i)

                a = np.concatenate(mi, axis=None)
                a = bb.join(a, QUInt(self.x_bitsize))
                
                b = bb.split(b)
                mi = np.split(np.array(b), self.x_bitsize // self.mult_window_size)
                for j in range(self.x_bitsize // self.mult_window_size):
                    data = list([(ke_inv * f * 2**j) % self.mod for f in range(2**self.mult_window_size)] for ke_inv in kes_inv)
                    ei_i = bb.join(ei[(self.exp_bitsize // self.exp_window_size) - i - 1], QUInt((self.exp_window_size)))
                    mi_i = bb.join(mi[(self.x_bitsize // self.mult_window_size) - j - 1], QUInt((self.mult_window_size)))
                    ei_i, mi_i, t, *junk = bb.add(self.qrom(data), selection0=ei_i, selection1=mi_i)
                    t, a = bb.add(ModSub(QUInt(self.x_bitsize), self.mod), x=t, y=a)
                    junk_mapping = {f'junk_target{i}_': junk[i] for i in range(len(junk))}
                    ei_i, mi_i = bb.add(self.qrom(data).adjoint(), selection0=ei_i, selection1=mi_i, target0_=t, **junk_mapping)
                    ei[(self.exp_bitsize // self.exp_window_size) - i - 1] = bb.split(ei_i)
                    mi[(self.x_bitsize // self.mult_window_size) - j - 1] = bb.split(mi_i)
                
                b = np.concatenate(mi, axis=None)
                
                b = bb.join(b, QUInt(self.x_bitsize))

                a, b = bb.add(Swap(self.x_bitsize), x=a, y=b)

                a = bb.split(a)
                    
            x = bb.join(a, QUInt(self.x_bitsize))
            exponent = np.concatenate(ei, axis=None)
            bb.free(b, dirty=True)
        else:
            # https://en.wikipedia.org/wiki/Modular_exponentiation#Right-to-left_binary_method
            base = self.base % self.mod
            for j in range(self.exp_bitsize - 1, 0 - 1, -1):
                exponent[j], x = bb.add(self._CtrlModMul(k=base), ctrl=exponent[j], x=x)
                base = (base * base) % self.mod

        return {'exponent': bb.join(exponent, dtype=QUInt(self.exp_bitsize)), 'x': x}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        if self.exp_window_size is not None and self.mult_window_size is not None:
            cg = {IntState(val=1, bitsize=self.x_bitsize): 1, Swap(self.x_bitsize): self.exp_bitsize // self.exp_window_size}

            k = self.base
            for i in range(self.exp_bitsize // self.exp_window_size):
                kes = [pow(k, 2**i * x_e, self.mod) for x_e in range(2**self.exp_window_size)]
                kes_inv = [pow(x_e, -1, self.mod) for x_e in kes]

                for j in range(self.x_bitsize // self.mult_window_size):
                    data = list([(ke * f * 2**j) % self.mod for f in range(2**self.mult_window_size)] for ke in kes)
                    cg[self.qrom(data)] = cg.get(self.qrom(data), 0) + 1
                    cg[ModAdd(self.x_bitsize, self.mod)] = cg.get(ModAdd(self.x_bitsize, self.mod), 0) + 1
                    cg[self.qrom(data).adjoint()] = cg.get(self.qrom(data).adjoint(), 0) + 1

                for j in range(self.x_bitsize // self.mult_window_size):
                    data = list([(ke_inv * f * 2**j) % self.mod for f in range(2**self.mult_window_size)] for ke_inv in kes_inv)
                    cg[self.qrom(data)] = cg.get(self.qrom(data), 0) + 1
                    cg[ModSub(QUInt(self.x_bitsize), self.mod)] = cg.get(ModSub(QUInt(self.x_bitsize), self.mod), 0) + 1
                    cg[self.qrom(data).adjoint()] = cg.get(self.qrom(data).adjoint(), 0) + 1

            return cg
        else:
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
    modexp_small = ModExp(base=4, mod=15, exp_bitsize=3, x_bitsize=2048)
    return modexp_small


@bloq_example(generalizer=(ignore_split_join, _generalize_k))
def _modexp() -> ModExp:
    modexp = ModExp.make_for_shor(big_n=13 * 17, g=9)
    return modexp


@bloq_example(generalizer=(ignore_split_join, _generalize_k))
def _modexp_window() -> ModExp:
    modexp_window = ModExp.make_for_shor(big_n=13 * 17, g=9, exp_window_size=8, mult_window_size=4)
    return modexp_window


@bloq_example
def _modexp_symb() -> ModExp:
    g, N, n_e, n_x = sympy.symbols('g N n_e, n_x')
    modexp_symb = ModExp(base=g, mod=N, exp_bitsize=n_e, x_bitsize=n_x)
    return modexp_symb

@bloq_example
def _modexp_window_symb() -> ModExp:
    g, N, n_e, n_x, w_e, w_m = sympy.symbols('g N n_e, n_x w_e w_m')
    modexp_window_symb = ModExp(base=g, mod=N, exp_bitsize=n_e, x_bitsize=n_x, exp_window_size=w_e, mult_window_size=w_m)
    return modexp_window_symb


_RSA_MODEXP_DOC = BloqDocSpec(bloq_cls=ModExp, examples=(_modexp_small, _modexp, _modexp_symb, _modexp_window, _modexp_window_symb))
