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
from functools import cached_property
from typing import Dict, Optional

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
    Signature,
    SoquetT,
)
from qualtran.bloqs.basic_gates import IntState, PlusState
from qualtran.bloqs.bookkeeping import Free
from qualtran.bloqs.cryptography._factoring_shims import MeasureQFT
from qualtran.bloqs.mod_arithmetic.mod_multiplication import CModMulK
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import is_symbolic, SymbolicInt


@frozen
class RSAPhaseEstimate(Bloq):
    """Perform a single phase estimation of the decomposition of Modular Exponentiation for the
    given base.

    The constructor requires a pre-set base, see the make_for_shor factory method for picking a
    random, valid base

    Args:
        n: The bitsize of the modulus N.
        mod: The modulus N; a part of the public key for RSA.
        base: A base for modular exponentiation.

    References:
        [Circuit for Shor's algorithm using 2n+3 qubits](https://arxiv.org/abs/quant-ph/0205095).
        Beauregard. 2003. Fig 1.
    """

    n: 'SymbolicInt'
    mod: 'SymbolicInt'
    base: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([])

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
        return cls(base=g, mod=big_n, n=little_n)

    def __attrs_post_init__(self):
        if not is_symbolic(self.n, self.mod):
            assert self.n == int(math.ceil(math.log2(self.mod)))

    def _CtrlModMul(self, k: 'SymbolicInt'):
        """Helper method to return a `CModMulK` with attributes forwarded."""
        return CModMulK(QUInt(self.n), k=k, mod=self.mod)

    def build_composite_bloq(self, bb: 'BloqBuilder') -> Dict[str, 'SoquetT']:
        if is_symbolic(self.n):
            raise DecomposeTypeError(f"Cannot decompose {self} with symbolic `n`.")
        exponent = [bb.add(PlusState()) for _ in range(2 * self.n)]
        x = bb.add(IntState(val=1, bitsize=self.n))

        base = self.base % self.mod
        for j in range((2 * self.n) - 1, 0 - 1, -1):
            exponent[j], x = bb.add(self._CtrlModMul(k=base), ctrl=exponent[j], x=x)
            base = (base * base) % self.mod

        bb.add(MeasureQFT(n=2 * self.n), x=exponent)
        bb.add(Free(QUInt(self.n), dirty=True), reg=x)
        return {}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        k = ssa.new_symbol('k')
        return {MeasureQFT(n=self.n): 1, self._CtrlModMul(k=k): 2 * self.n}


_K = sympy.Symbol('k_exp')


def _generalize_k(b: Bloq) -> Optional[Bloq]:
    if isinstance(b, CModMulK):
        return attrs.evolve(b, k=_K)

    return b


@bloq_example
def _rsa_pe() -> RSAPhaseEstimate:
    n, p, g = sympy.symbols('n p g')
    rsa_pe = RSAPhaseEstimate(n=n, mod=p, base=g)
    return rsa_pe


@bloq_example
def _rsa_pe_small() -> RSAPhaseEstimate:
    rsa_pe_small = RSAPhaseEstimate.make_for_shor(big_n=13 * 17, g=9)
    return rsa_pe_small


_RSA_PE_BLOQ_DOC = BloqDocSpec(bloq_cls=RSAPhaseEstimate, examples=[_rsa_pe_small, _rsa_pe])
