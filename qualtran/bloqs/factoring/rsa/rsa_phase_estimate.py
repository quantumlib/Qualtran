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
from qualtran.bloqs.factoring._factoring_shims import MeasureQFT
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics.types import is_symbolic, SymbolicInt

from .rsa_mod_exp import ModExp


@frozen
class RSAPhaseEstimate(Bloq):
    """Perform a single phase estimation of ModExp for the given base.

    Computes the phase estimation of a single run of Modular Exponentiation with
    an optional, pre-set base or a random, valid base.

    Args:
        n: The bitsize of the modulus N.
        mod: The modulus N; a part of the public key for RSA.
        base: An optional base for modular exponentiation.

    References:
        [Circuit for Shor's algorithm using 2n+3 qubits](https://arxiv.org/abs/quant-ph/0205095).
        Stephane Beauregard. 2003.
    """

    n: 'SymbolicInt'
    mod: 'SymbolicInt'
    base: Optional['SymbolicInt'] = None

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([])

    def __attrs_post_init__(self):
        if not is_symbolic(self.n, self.mod):
            assert self.n == int(math.ceil(math.log2(self.mod)))

    def build_composite_bloq(self, bb: 'BloqBuilder') -> Dict[str, 'SoquetT']:
        if is_symbolic(self.n):
            raise DecomposeTypeError(f"Cannot decompose {self} with symbolic `n`.")
        exponent = [bb.add(PlusState()) for _ in range(2 * self.n)]
        x = bb.add(IntState(val=1, bitsize=self.n))

        exponent, x = bb.add(
            ModExp.make_for_shor(big_n=self.mod, g=self.base), exponent=exponent, x=x
        )

        bb.add(MeasureQFT(n=2 * self.n), x=exponent)
        bb.add(Free(QUInt(self.n), dirty=True), reg=x)
        return {}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {MeasureQFT(n=self.n): 1, ModExp.make_for_shor(big_n=self.mod, g=self.base): 1}


@bloq_example
def _rsa_pe() -> RSAPhaseEstimate:
    n, p = sympy.symbols('n p')
    rsa_pe = RSAPhaseEstimate(n=n, mod=p)
    return rsa_pe


@bloq_example
def _rsa_pe_small() -> RSAPhaseEstimate:
    n, p = 6, 5 * 7
    rsa_pe_small = RSAPhaseEstimate(n=n, mod=p)
    return rsa_pe_small


_RSA_PE_BLOQ_DOC = BloqDocSpec(bloq_cls=RSAPhaseEstimate, examples=[_rsa_pe_small, _rsa_pe])
