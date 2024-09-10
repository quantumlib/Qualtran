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

from typing import Dict

import numpy as np
from attrs import frozen

from qualtran import Bloq, bloq_example, BloqDocSpec, QFxp, Register, Signature
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.rotations.phase_gradient import _fxp
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.symbolics import is_symbolic, SymbolicInt


@frozen
class ArcSin(Bloq):
    r"""Compute the arcsine of a fixed-point number.

    Implements the unitary:
    $$
        |a\rangle|0\rangle \rightarrow |a\rangle|\arcsin(a)\rangle
    $$

    Args:
        bitsize: Number of bits used to represent the number.
        num_frac: Number of fraction bits in the number.
        num_iters: Number of Newton-Raphson iterations.
            Defaults to 4; the reference studies 3, 4, or 5 iterations.
        degree: Degree of the polynomial of the initial approximation.
            Defaults to 4; the reference studies degree-3, 4, 5, or 6 polynomials.

    Registers:
        x: `bitsize`-sized input register.
        result: `bitsize`-sized output register.

    References:
        [Optimizing Quantum Circuits for Arithmetic](https://arxiv.org/abs/1805.12445). Appendix D.
    """

    bitsize: SymbolicInt
    num_frac: SymbolicInt
    num_iters: SymbolicInt = 4
    degree: SymbolicInt = 4

    def __attrs_post_init__(self):
        if (
            not is_symbolic(self.num_frac)
            and not is_symbolic(self.bitsize)
            and self.num_frac > self.bitsize
        ):
            raise ValueError("num_frac must be < bitsize.")

    @property
    def signature(self):
        return Signature(
            [
                Register("x", QFxp(self.bitsize, self.num_frac)),
                Register("result", QFxp(self.bitsize, self.num_frac)),
            ]
        )

    def on_classical_vals(
        self, x: ClassicalValT, result: ClassicalValT
    ) -> Dict[str, ClassicalValT]:
        if is_symbolic(self.bitsize):
            raise ValueError(f"Symbolic bitsize {self.bitsize} not supported")
        x_fxp: float = _fxp(x / 2**self.bitsize, self.bitsize).astype(float)
        result ^= int(np.arcsin(x_fxp) * 2**self.bitsize)
        return {'x': x, 'result': result}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        n = self.bitsize
        p = self.bitsize - self.num_frac
        d = self.degree
        m = self.num_iters
        # directly copied from T_arcsin on page 10 of reference
        toffolis = (
            d * (3 * n**2 + n * (6 * p + 7) - 6 * (p - 1) * p - 2)
            + m * (n * (15 * n + 30 * p + 23) - 30 * p * (p - 1) - 4)
            + 9 * (n + 1) * p
            + 9 * n * (n + 1) // 2
            + 6 * n**2
            + 28 * n
            - 9 * p**2
            + 2
        )
        return {Toffoli(): toffolis}


@bloq_example
def _arcsin() -> ArcSin:
    arcsin = ArcSin(bitsize=10, num_frac=7)
    return arcsin


_ARCSIN_DOC = BloqDocSpec(bloq_cls=ArcSin, examples=[_arcsin])
