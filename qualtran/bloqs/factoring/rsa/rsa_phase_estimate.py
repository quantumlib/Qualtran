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

from functools import cached_property
from typing import Dict

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
from qualtran.bloqs.basic_gates import PlusState
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics.types import SymbolicInt

from .._factoring_shims import MeasureQFT


@frozen
class RSAPhaseEstimate(Bloq):
    """Perform a single phase estimation of ECAddR for a given point.

    This is used as a subroutine in `FindECCPrivateKey`. First, we phase-estimate the
    addition of the base point $P$, then of the public key $Q$.

    Args:
        n: The bitsize of the elliptic curve points' x and y registers.
        point: The elliptic curve point to phase estimate against.
    """

    n: 'SymbolicInt'
    mod: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QUInt(self.n))])

    def build_composite_bloq(self, bb: 'BloqBuilder', x: Soquet, y: Soquet) -> Dict[str, 'SoquetT']:
        if isinstance(self.n, sympy.Expr):
            raise DecomposeTypeError("Cannot decompose symbolic `n`.")
        ctrl = [bb.add(PlusState()) for _ in range(2 * self.n)]
        for i in range(self.n):
            ctrl[i], x, y = bb.add(ECAddR(n=self.n, R=2**i * self.point), ctrl=ctrl[i], x=x, y=y)

        bb.add(MeasureQFT(n=2 * self.n), x=ctrl)
        return {'x': x}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {ECAddR(n=self.n, R=self.point): self.n, MeasureQFT(n=self.n): 1}


@bloq_example
def _rsa_pe() -> RSAPhaseEstimate:
    n, p = sympy.symbols('n p')
    ec_pe = RSAPhaseEstimate(n=n, p=p)
    return ec_pe


@bloq_example
def _rsa_pe_small() -> RSAPhaseEstimate:
    n, p = 3, 7
    ec_pe_small = RSAPhaseEstimate(n=n, p=p)
    return ec_pe_small


_RSA_PE_BLOQ_DOC = BloqDocSpec(bloq_cls=RSAPhaseEstimate, examples=[_rsa_pe])
