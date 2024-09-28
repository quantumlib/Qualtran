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

from qualtran import Bloq, bloq_example, BloqBuilder, BloqDocSpec, QUInt, Signature, SoquetT
from qualtran.bloqs.basic_gates import IntState
from qualtran.bloqs.bookkeeping import Free
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import SymbolicInt

from .rsa_phase_estimate import RSAPhaseEstimate


@frozen
class FindRSAPrivateKey(Bloq):
    r"""Perform one phase estimation to break rsa cryptography.

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
        n: The bitsize of the integer we want to factor.
        mod: The integer modulus.

    References:
        [Circuit for Shor's algorithm using 2n+3 qubits](https://arxiv.org/abs/quant-ph/0205095).
        Figure 1.
    """

    n: 'SymbolicInt'
    mod: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([])

    def build_composite_bloq(self, bb: 'BloqBuilder') -> Dict[str, 'SoquetT']:
        x = bb.add(IntState(bitsize=self.n, val=1))

        x = bb.add(RSAPhaseEstimate(n=self.n, mod=self.mod), x=x)

        bb.add(Free(QUInt(self.n), dirty=True), reg=x)
        return {}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {RSAPhaseEstimate(n=self.n): 1}

    def cost_attrs(self):
        return [('n', self.n)]


@bloq_example
def _rsa() -> FindRSAPrivateKey:
    n, p = sympy.symbols('n p')
    rsa = FindRSAPrivateKey(n=n, mod=p)
    return rsa


_RSA_BLOQ_DOC = BloqDocSpec(bloq_cls=FindRSAPrivateKey, examples=[_rsa])
