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

import functools
from functools import cached_property
from typing import Dict, Union

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
from qualtran.bloqs.basic_gates import PlusState
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics.types import SymbolicInt

from .._factoring_shims import MeasureQFT
from .ec_add_r import ECAddR, ECWindowAddR
from .ec_point import ECPoint


@frozen
class ECPhaseEstimateR(Bloq):
    """Perform a single phase estimation of ECAddR for a given point.

    This is used as a subroutine in `FindECCPrivateKey`. First, we phase-estimate the
    addition of the base point $P$, then of the public key $Q$.

    When the ellptic curve point addition window size is 1 we use the ECAddR bloq which has it's
    own bespoke circuit; when it is greater than 1 we use the windowed circuit which uses
    pre-computed classical additions loaded into the circuit.

    Args:
        n: The bitsize of the elliptic curve points' x and y registers.
        point: The elliptic curve point to phase estimate against.
        add_window_size: The number of bits in the ECAdd window.
        mul_window_size: The number of bits in the modular multiplication window.
    """

    n: 'SymbolicInt'
    point: ECPoint
    add_window_size: 'SymbolicInt' = 1
    mul_window_size: 'SymbolicInt' = 1

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QUInt(self.n)), Register('y', QUInt(self.n))])

    @property
    def ec_add(self) -> Union[functools.partial[ECAddR], functools.partial[ECWindowAddR]]:
        if self.add_window_size == 1:
            return functools.partial(ECAddR, n=self.n)
        return functools.partial(
            ECWindowAddR,
            n=self.n,
            add_window_size=self.add_window_size,
            mul_window_size=self.mul_window_size,
        )

    @property
    def num_windows(self) -> int:
        return self.n // self.add_window_size

    def build_composite_bloq(self, bb: 'BloqBuilder', x: Soquet, y: Soquet) -> Dict[str, 'SoquetT']:
        if isinstance(self.n, sympy.Expr):
            raise DecomposeTypeError("Cannot decompose symbolic `n`.")
        ctrl = [bb.add(PlusState()) for _ in range(self.n)]

        if self.add_window_size == 1:
            for i in range(self.n):
                ctrl[i], x, y = bb.add(self.ec_add(R=2**i * self.point), ctrl=ctrl[i], x=x, y=y)
        else:
            ctrls = np.split(np.array(ctrl), self.num_windows)
            for i in range(self.num_windows):
                ctrls[i], x, y = bb.add(
                    self.ec_add(R=2 ** (self.add_window_size * i) * self.point),
                    ctrl=ctrls[i],
                    x=x,
                    y=y,
                )
            ctrl = np.concatenate(ctrls, axis=None)

        bb.add(MeasureQFT(n=self.n), x=ctrl)
        return {'x': x, 'y': y}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {self.ec_add(R=self.point): self.num_windows, MeasureQFT(n=self.n): 1}

    def __str__(self) -> str:
        return f'PE${self.point}$'


@bloq_example
def _ec_pe() -> ECPhaseEstimateR:
    n, p = sympy.symbols('n p')
    Rx, Ry = sympy.symbols('R_x R_y')
    ec_pe = ECPhaseEstimateR(n=n, point=ECPoint(Rx, Ry, mod=p))
    return ec_pe


@bloq_example
def _ec_pe_small() -> ECPhaseEstimateR:
    n = 3
    Rx, Ry, p = sympy.symbols('R_x R_y p')
    ec_pe_small = ECPhaseEstimateR(n=n, point=ECPoint(Rx, Ry, mod=p))
    return ec_pe_small


_EC_PE_BLOQ_DOC = BloqDocSpec(bloq_cls=ECPhaseEstimateR, examples=[_ec_pe, _ec_pe_small])
