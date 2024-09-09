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
r"""Bloqs for preparing the $\nu$ state for the first quantized chemistry Hamiltonian."""
from functools import cached_property
from typing import Dict, TYPE_CHECKING

from attrs import evolve, frozen

from qualtran import Bloq, bloq_example, BloqBuilder, QAny, QBit, Register, Side, Signature, SoquetT
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu import (
    FlagZeroAsFailure,
    PrepareNuSuperPositionState,
    TestNuInequality,
    TestNuLessThanMu,
)
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
    PrepareUniformSuperposition,
)

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@frozen
class PrepareMuUnaryEncodedOneHotWithProj(Bloq):
    r"""Prepare a unary encoded one-hot superposition state over the $\mu$ register for nested boxes

    Prepares the state in A20

    $$
        \frac{1}{\sqrt{2^{n_n + 2}}} \sum_{\mu=2}^{n_n+1} \sqrt{2^\mu}
        |0\dots0\underbrace{1\dots1}{\mu}\rangle
    $$

    To account for the conditional preparation of weight for the projectile we
    add additional controls to the Hadamards for $p > n_p$.

    Args:
        bitsize_n: The number of bits to represent each dimension of the
            momentum for the projectile.
        bitsize_p: The number of bits to represent each dimension of the
            momentum register for the electron.

    Registers:
        mu: the register to prepare the superposition over.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](https://arxiv.org/abs/2105.12767)
        page 21, Eq 77.
    """
    bitsize_n: int
    bitsize_p: int
    is_adjoint: bool = False

    def __attrs_post_init__(self):
        if self.bitsize_n < self.bitsize_p:
            raise ValueError(f"bitsize_n < bitsize_p : {self.bitsize_n} < {self.bitsize_p}.")

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [Register("mu", QAny(self.bitsize_n)), Register("flag", QBit(), side=Side.RIGHT)]
        )

    def adjoint(self) -> 'Bloq':
        return evolve(self, is_adjoint=not self.is_adjoint)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        if self.is_adjoint:
            return {Toffoli(): (self.bitsize_n - 1) + 1}
        else:
            return {Toffoli(): (self.bitsize_n - 1) + (self.bitsize_n - self.bitsize_p - 1) + 1}


@frozen
class PrepareNuStateWithProj(Bloq):
    r"""PREPARE for the $\nu$ state for the $U$ and $V$ components of the Hamiltonian.

    Prepares a state of the form

    $$
        \frac{1}{\sqrt{\mathcal{M}2^{n_p + 2}}}
        \sum_{\mu=2}^{n_p+1}\sum_{\nu \in B_\mu}
        \sum_{m=0}^{\lceil \mathcal M(2^{\mu-2}/\lVert\nu\rVert)^2\rceil-1}
        \frac{1}{2^\mu}|\mu\rangle|\nu_x\rangle|\nu_y\rangle|\nu_z\rangle|m\rangle|0\rangle
    $$

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        num_bits_n: The number of bits to represent each dimension of the
            momentum for the projectile.
        m_param: $\mathcal{M}$ in the reference.
        lambda_zeta: sum of nuclear charges.
        er_lambda_zeta: eq 91 of the referce. Cost of erasing qrom.

    Registers:
        mu: The state controlling the nested boxes procedure.
        nu: The momentum transfer register.
        m: an ancilla register in a uniform superposition.
        flag_nu: Flag for success of the state preparation.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](https://arxiv.org/abs/2105.12767)
        page 19, section B
    """
    num_bits_p: int
    num_bits_n: int
    m_param: int

    @cached_property
    def signature(self) -> Signature:
        n_m = (self.m_param - 1).bit_length()
        return Signature(
            [
                Register("mu", QAny(bitsize=self.num_bits_n)),
                Register("nu", QAny(bitsize=self.num_bits_n + 1), shape=(3,)),
                Register("m", QAny(bitsize=n_m)),
                Register("flag_nu", QBit()),
            ]
        )

    def build_composite_bloq(
        self, bb: BloqBuilder, mu: SoquetT, nu: SoquetT, m: SoquetT, flag_nu: SoquetT
    ) -> Dict[str, 'SoquetT']:
        mu, flag_mu = bb.add(
            PrepareMuUnaryEncodedOneHotWithProj(self.num_bits_n, self.num_bits_p), mu=mu
        )
        mu, nu = bb.add(PrepareNuSuperPositionState(self.num_bits_n), mu=mu, nu=nu)
        nu, flag_zero = bb.add(FlagZeroAsFailure(self.num_bits_n), nu=nu)
        mu, nu, flag_nu_lt_mu = bb.add(TestNuLessThanMu(self.num_bits_n), mu=mu, nu=nu)
        n_m = (self.m_param - 1).bit_length()
        m = bb.add(PrepareUniformSuperposition(self.m_param), target=m)
        mu, nu, m, flag_nu = bb.add(
            TestNuInequality(self.num_bits_n, n_m),
            mu=mu,
            nu=nu,
            m=m,
            flag_mu_prep=flag_mu,
            flag_minus_zero=flag_zero,
            flag_ineq=flag_nu_lt_mu,
            succ=flag_nu,
        )
        return {'mu': mu, 'nu': nu, 'm': m, 'flag_nu': flag_nu}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # 1. Prepare unary encoded superposition state (Eq 77)
        cost_1 = (PrepareMuUnaryEncodedOneHotWithProj(self.num_bits_n, self.num_bits_p), 1)
        n_m = (self.m_param - 1).bit_length()
        # 2. Prepare mu-nu superposition (Eq 78)
        cost_2 = (PrepareNuSuperPositionState(self.num_bits_n), 1)
        # 3. Remove minus zero
        cost_3 = (FlagZeroAsFailure(self.num_bits_n), 1)
        # 4. Test $\nu < 2^{\mu-2}$
        cost_4 = (TestNuLessThanMu(self.num_bits_n), 1)
        # 5. Prepare superposition over $m$ which is a power of two so only clifford.
        # 6. Test that $(2^{\mu-2})^2\mathcal{M} > m (\nu_x^2 + \nu_y^2 + \nu_z^2)$
        cost_6 = (TestNuInequality(self.num_bits_n, n_m), 1)
        return dict([cost_1, cost_2, cost_3, cost_4, cost_6])


@bloq_example
def _prep_mu_proj() -> PrepareMuUnaryEncodedOneHotWithProj:
    num_bits_p = 6
    num_bits_n = 8
    prep_mu_proj = PrepareMuUnaryEncodedOneHotWithProj(num_bits_n, num_bits_p)
    return prep_mu_proj


@bloq_example
def _prep_nu_proj() -> PrepareNuStateWithProj:
    num_bits_p = 6
    num_bits_n = 8
    m_param = 2 ** (2 * num_bits_n + 3)
    prep_nu_proj = PrepareNuStateWithProj(num_bits_p, num_bits_n, m_param)
    return prep_nu_proj
