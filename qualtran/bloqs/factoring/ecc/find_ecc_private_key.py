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
from typing import Dict

import sympy
from attrs import frozen

from qualtran import Bloq, bloq_example, BloqBuilder, BloqDocSpec, QUInt, Signature, SoquetT
from qualtran.bloqs.basic_gates import IntState
from qualtran.bloqs.bookkeeping import Free
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import SymbolicInt

from .ec_phase_estimate_r import ECPhaseEstimateR
from .ec_point import ECPoint


@frozen
class FindECCPrivateKey(Bloq):
    r"""Perform two phase estimations to break elliptic curve cryptography.

    This follows the strategy in Litinski 2023. We perform two phase estimations corresponding
    to `ECCAddR(R=P)` and `ECCAddR(R=Q)` for base point $P$ and public key $Q$.

    The first phase estimation projects us into a random eigenstate of the ECCAddR(R=P) operator
    which we index by the integer $c$. Per eq. 5 in the reference, these eigenstates take the form
    $$
    |\psi_c \rangle = \sum_j^{r-1} \omega^{cj}\ | [j]P \rangle  \\
    \omega = e^{2\pi i / r} \\
    [r] P = P
    $$

    This state is a simultaneous eigenstate of the second operator, `ECCAddR(R=Q)`. By
    the definition of the operator, acting it upon $|\psi_c\rangle$ gives:
    $$
    |\psi_c \rangle \rightarrow \sum_j w^{cj} | [j]P + Q \rangle\rangle
    $$

    The private key $k$ that we wish to recover relates the public key to the base point
    $$
    Q = [k] P
    $$
    so our simultaneous eigenstate can be equivalently written as
    $$
    \sum_j^{r-1} \omega^{cj} | [j+k] P \rangle \\
    = \omega^{-ck} |\psi_c \rangle
    $$

    Therefore, the measured result of the second phase estimation is $ck$. Since we have
    already measured the random index $c$, we can divide it out to recover the private key $k$.

    Args:
        n: The bitsize of the elliptic curve points' x and y registers.
        base_point: The base point $P$ with unknown order $r$ such that $P = [r] P$.
        public_key: The public key $Q$ such that $Q = [k] P$ for private key $k$.

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585).
        Litinski. 2023. Figure 4 (a).
    """

    n: int
    base_point: ECPoint
    public_key: ECPoint

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([])

    @property
    def mod(self) -> SymbolicInt:
        if self.base_point.mod != self.public_key.mod:
            raise ValueError("Inconsistent moduli in the two points.")
        return self.base_point.mod

    @property
    def curve_a(self) -> SymbolicInt:
        if self.base_point.curve_a != self.public_key.curve_a:
            raise ValueError("Inconsistent curve parameters in the two points.")
        return self.base_point.curve_a

    def build_composite_bloq(self, bb: 'BloqBuilder') -> Dict[str, 'SoquetT']:
        x = bb.add(IntState(bitsize=self.n, val=self.base_point.x))
        y = bb.add(IntState(bitsize=self.n, val=self.base_point.y))

        x, y = bb.add(ECPhaseEstimateR(n=self.n, point=self.base_point), x=x, y=y)
        x, y = bb.add(ECPhaseEstimateR(n=self.n, point=self.public_key), x=x, y=y)

        bb.add(Free(QUInt(self.n)), reg=x)
        bb.add(Free(QUInt(self.n)), reg=y)
        return {}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        Rx = ssa.new_symbol('Rx')
        Ry = ssa.new_symbol('Ry')
        generic_point = ECPoint(Rx, Ry, mod=self.mod, curve_a=self.curve_a)

        return {ECPhaseEstimateR(n=self.n, point=generic_point): 2}

    def cost_attrs(self):
        return [('n', self.n)]


@bloq_example
def _ecc() -> FindECCPrivateKey:
    n, p = sympy.symbols('n p')
    Px, Py, Qx, Qy = sympy.symbols('P_x P_y Q_x Q_y')
    P = ECPoint(Px, Py, mod=p)
    Q = ECPoint(Qx, Qy, mod=p)
    ecc = FindECCPrivateKey(n=n, base_point=P, public_key=Q)
    return ecc


_ECC_BLOQ_DOC = BloqDocSpec(bloq_cls=FindECCPrivateKey, examples=[_ecc])
