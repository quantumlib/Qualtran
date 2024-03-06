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
r"""Bloqs for preparation of the U and V parts of the first quantized chemistry Hamiltonian."""
from functools import cached_property
from typing import Dict, Set, TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, BloqBuilder, QAny, QBit, Register, Side, Signature, SoquetT
from qualtran.bloqs.arithmetic import GreaterThan, Product, SumOfSquares
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
    PrepareUniformSuperposition,
)

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class PrepareMuUnaryEncodedOneHot(Bloq):
    r"""Prepare a unary encoded one-hot superposition state over the $\mu$ register for nested boxes

    Prepares the state in Eq. 77

    $$
        \frac{1}{\sqrt{2^{n_p + 2}}} \sum_{\mu=2}^{n_p+1} \sqrt{2^\mu}
        |0\dots0\underbrace{1\dots1}{\mu}\rangle
    $$

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.

    Registers:
        mu: the register to prepare the superposition over.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) page 21, Eq 77.
    """
    num_bits_p: int
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [Register("mu", QAny(self.num_bits_p)), Register("flag", QBit(), side=Side.RIGHT)]
        )

    def short_name(self) -> str:
        return r'PREP $\sqrt{2^\mu}|\mu\rangle$'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(Toffoli(), (self.num_bits_p - 1))}


@frozen
class PrepareNuSuperPositionState(Bloq):
    r"""Prepare the superposition over $\nu$ following the creation of the $|\mu\rangle$ state.

    Prepares the state in Eq. 78

    $$
        \frac{1}{\sqrt{2^{n_p + 2}}} \sum_{\mu=2}^{n_p+1}
        \sum_{\nu_{x,y,z}=-(2^{\mu-1}-1)}^{2^{\mu-1}-1}
        \frac{1}{2\mu}
        |\mu\rangle|\nu_x\rangle|\nu_y\rangle|\nu_z\rangle
    $$

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.

    Registers:
        mu: the register one-hot encoded $\mu$ register from eq 77.
        nu: the momentum transfer register.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) page 21, Eq 78.
    """
    num_bits_p: int
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("mu", QAny(self.num_bits_p)),
                Register("nu", QAny(self.num_bits_p + 1), shape=(3,)),
            ]
        )

    def short_name(self) -> str:
        return r'PREP $2^{-\mu}|\mu\rangle|\nu\rangle$'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # controlled hadamards which cannot be inverted at zero Toffoli cost.
        return {(Toffoli(), (3 * (self.num_bits_p - 1)))}


@frozen
class FlagZeroAsFailure(Bloq):
    r"""Bloq to flag if minus zero appears in the $\nu$ state.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.

    Registers:
        nu: the momentum transfer register.
        flag_minus_zero: a flag bit for failure of the state preparation.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) page 21, Eq 80.
    """
    num_bits_p: int
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("nu", QAny(self.num_bits_p + 1), shape=(3,)),
                Register("flag_minus_zero", QBit(), side=Side.RIGHT),
            ]
        )

    def short_name(self) -> str:
        return r'$\nu\ne -0$'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if self.adjoint:
            # This can be inverted with cliffords.
            return {}
        else:
            # Controlled Toffoli each having n_p + 1 controls and 2 Toffolis to
            # check the result of the Toffolis.
            return {(Toffoli(), (3 * self.num_bits_p + 2))}


@frozen
class TestNuLessThanMu(Bloq):
    r"""Bloq to flag if all components of $\nu$ are smaller in absolute value than $2^{\mu-2}$.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.

    Registers:
        nu: the momentum transfer register.
        flag_nu_lt_mu: a flag bit for failure of the state preparation.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) page 21, Eq 80.
    """
    num_bits_p: int
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("mu", QAny(self.num_bits_p)),
                Register("nu", QAny(self.num_bits_p + 1), shape=(3,)),
                Register("flag_nu_lt_mu", QBit(), side=Side.RIGHT),
            ]
        )

    def short_name(self) -> str:
        return r'$\nu < 2^{\mu-2}$'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if self.adjoint:
            # This can be inverted with cliffords.
            return {(Toffoli(), 0)}
        else:
            # n_p controlled Toffolis with four controls.
            return {(Toffoli(), 3 * self.num_bits_p)}


@frozen
class TestNuInequality(Bloq):
    r"""Bloq to flag if all components of $\nu$ are smaller in absolute value than $2^{\mu-2}$.

    Test

    $$
        (2^{\mu-2})^2 \mathcal{M} > m (\nu_x^2 + \nu_y^2 + \nu_z^2)
    $$

    where $m \in [0, \mathcal{M}-1]$ and is store in an ancilla register.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        num_bits_m: The number of bits for $\mathcal{M}$. Eq 86.
        adjoint: Whether to dagger the bloq or not.

    Registers:
        mu: the one-hot unary superposition register.
        nu: the momentum transfer register.
        m: the ancilla register in unfiform superposition.
        flag_mu_prep: A flag checking the success of the $mu$ state prep.
        flag_minus_zero: A flag from checking for negative zero.
        flag_ineq: A flag from checking $\nu \lt 2^{\mu -2}$.
        succ: a flag bit for failure of the state preparation.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) page 21, Eq 80.
    """
    num_bits_p: int
    num_bits_m: int
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("mu", QAny(self.num_bits_p)),
                Register("nu", QAny(self.num_bits_p + 1), shape=(3,)),
                Register("m", QAny(self.num_bits_m)),
                Register("flag_minus_zero", QBit(), side=Side.LEFT),
                Register("flag_mu_prep", QBit(), side=Side.LEFT),
                Register("flag_ineq", QBit(), side=Side.LEFT),
                Register("succ", QBit()),
            ]
        )

    def short_name(self) -> str:
        return r'$(2^{\mu-2})^2\mathcal{M} > m \nu^2 $'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if self.adjoint:
            return {(Toffoli(), 0)}
        else:
            # 1. Compute $\nu_x^2 + \nu_y^2 + \nu_z^2$
            cost_1 = (SumOfSquares(self.num_bits_p, k=3), 1)
            # 2. Compute $m (\nu_x^2 + \nu_y^2 + \nu_z^2)$
            cost_2 = (Product(2 * self.num_bits_p + 2, self.num_bits_m), 1)
            # 3. Inequality test
            cost_3 = (GreaterThan(self.num_bits_m, 2 * self.num_bits_p + 2), 1)
            # 4. 3 Toffoli for overall success
            cost_4 = (Toffoli(), 3)
            return {cost_1, cost_2, cost_3, cost_4}


@frozen
class PrepareNuState(Bloq):
    r"""PREPARE for the $\nu$ state for the $U$ and $V$ components of the Hamiltonian.

    Prepares a state of the form

    $$
        \frac{1}{\sqrt{\mathcal{M}2^{n_p + 2}}}
        \sum_{\mu=2}^{n_p+1}\sum_{\nu \in B_\mu}
        \sum_{m=0}^{\lceil \mathcal M(2^{\mu-2}/\lVert\nu\rVert)^2\rceil-1}
        \frac{1}{2^\mu}|\mu\rangle|\nu_x\rangle|\nu_y\rangle|\nu_z\rangle|m\rangle|0\rangle
    $$

    Note the costs differ from those listed in the reference by 5 Toffolis.

    The cost for the arithmetic is
    $$
    3 n_p^2 + n_p + 1 + 4 n_\mathcal{M}(n_p + 1) \hspace{1em} (90)
    $$
    We also need to add 3 Toffolis which can be inverted at zero Toffoli cost
    for flagging success.

    The other costs are 4(np-1) controlled hadamards (not inverted at zero cost)
    and 6np + 2 Toffolis (free inversion).  So focusing on the n_p terms and
    constants
    $n_p + 2 . 4 n_p + 6 n_p = 15 n_p$ (consistent)
    and the constants
    $ 4 - 2.4 + 2 = -2$ (not -7).

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        m_param: $\mathcal{M}$ in the reference.
        lambda_zeta: sum of nuclear charges.
        er_lambda_zeta: eq 91 of the referce. Cost of erasing qrom.

    Registers:
        mu: The state controlling the nested boxes procedure.
        nu: The momentum transfer register.
        m: an ancilla register in a uniform superposition.
        flag_nu: Flag for success of the state preparation.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) page 19, section B
    """
    num_bits_p: int
    m_param: int
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        n_m = (self.m_param - 1).bit_length()
        return Signature(
            [
                Register("mu", QAny(bitsize=self.num_bits_p)),
                Register("nu", QAny(bitsize=self.num_bits_p + 1), shape=(3,)),
                Register("m", QAny(bitsize=n_m)),
                Register("flag_nu", QBit()),
            ]
        )

    def short_name(self) -> str:
        return r"PREP $\frac{1}{\lVert \nu \rVert} |\nu\rangle $"

    def build_composite_bloq(
        self, bb: BloqBuilder, mu: SoquetT, nu: SoquetT, m: SoquetT, flag_nu: SoquetT
    ) -> Dict[str, 'SoquetT']:
        mu, flag_mu = bb.add(
            PrepareMuUnaryEncodedOneHot(self.num_bits_p, adjoint=self.adjoint), mu=mu
        )
        mu, nu = bb.add(
            PrepareNuSuperPositionState(self.num_bits_p, adjoint=self.adjoint), mu=mu, nu=nu
        )
        nu, flag_zero = bb.add(FlagZeroAsFailure(self.num_bits_p, adjoint=self.adjoint), nu=nu)
        mu, nu, flag_nu_lt_mu = bb.add(
            TestNuLessThanMu(self.num_bits_p, adjoint=self.adjoint), mu=mu, nu=nu
        )
        n_m = (self.m_param - 1).bit_length()
        m = bb.add(PrepareUniformSuperposition(self.m_param), target=m)
        mu, nu, m, flag_nu = bb.add(
            TestNuInequality(self.num_bits_p, n_m, adjoint=self.adjoint),
            mu=mu,
            nu=nu,
            m=m,
            flag_mu_prep=flag_mu,
            flag_minus_zero=flag_zero,
            flag_ineq=flag_nu_lt_mu,
            succ=flag_nu,
        )
        return {'mu': mu, 'nu': nu, 'm': m, 'flag_nu': flag_nu}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # 1. Prepare unary encoded superposition state (Eq 77)
        cost_1 = (PrepareMuUnaryEncodedOneHot(self.num_bits_p), 1)
        n_m = (self.m_param - 1).bit_length()
        # 2. Prepare mu-nu superposition (Eq 78)
        cost_2 = (PrepareNuSuperPositionState(self.num_bits_p), 1)
        # 3. Remove minus zero
        cost_3 = (FlagZeroAsFailure(self.num_bits_p, adjoint=self.adjoint), 1)
        # 4. Test $\nu < 2^{\mu-2}$
        cost_4 = (TestNuLessThanMu(self.num_bits_p, adjoint=self.adjoint), 1)
        # 5. Prepare superposition over $m$ which is a power of two so only clifford.
        # 6. Test that $(2^{\mu-2})^2\mathcal{M} > m (\nu_x^2 + \nu_y^2 + \nu_z^2)$
        cost_6 = (TestNuInequality(self.num_bits_p, n_m, adjoint=self.adjoint), 1)
        return {cost_1, cost_2, cost_3, cost_4, cost_6}
