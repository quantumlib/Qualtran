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
r"""SELECT and PREPARE for the first quantized chemistry Hamiltonian.

Here we assume the Born-Oppenheimer Hamiltonian and seek to simulate a
collection of $\eta$ electrons and $L$ static nuclei (of charge $\zeta_\ell$)
with a Hamiltonian given by:
$$
H_{BO} = T + U + V + \frac{1}{2}
\sum_{\ell\ne\kappa=1}^L\frac{\zeta_\ell\zeta_\kappa}{\lVert R_\ell - R_\kappa \rVert}
$$

In the first quantized approach we assume periodic boundary conditions and use a
plane wave Galerkin discretization.
A plane wave basis function is given by
$$
\phi_p(r) = \frac{1}{\sqrt{\Omega}} e^{-i k_p\cdot r}
$$
where $r$ is a position vector in real space, $\Omega$ is the simulation cell
volume and $k_p$ is a reciprocal lattice vector.
In three dimensions we have
$$
k_p = \frac{2\pi p }{\Omega}
$$
for $p \in G$ and
$$
G = \left[-\frac{N^{1/3} -
1}{2},\frac{N^{1/3} - 1}{2}\right]^{\otimes 3} \subset \mathbb{Z}^3,
$$
and $N$ is the total number of planewaves.

With these definitions we can write the components of the Hamiltonian as:
$$
T = \sum_{i}^\eta\sum_{p\in G}\frac{\lVert k_p\rVert^2}{2} |p\rangle\langle p|_i
$$
which defines the kinetic energy of the electrons,
$$
U = -\frac{4\pi}{\Omega}
\sum_{\ell=1}^L \sum_{i}^\eta
\sum_{p,q\in G, p\ne q}
\left(
    \zeta_{\ell}
    \frac{e^{i k_{q-p}\cdot R_\ell}}{\lVert k_{p-q}\rVert^2}
    |p\rangle\langle q|_i
\right)
$$
describes the interaction of the electrons and the nuclei, and,
$$
V = \frac{2\pi}{\Omega}
\sum_{i\ne j=1}^\eta
\sum_{p,q\in G, p\ne q}
\sum_{\nu \in G_0}
\left(
    \frac{1}{\lVert k_{\nu}\rVert^2}
    |p + \nu\rangle\langle p|_i
    |q -\nu\rangle\langle q|_i
\right)
$$
describes the electron-electron interaction. The notation $|p\rangle\langle p|_i$ is shorthand for
$I_1\otimes\cdots\otimes |p\rangle \langle p |_j \otimes \cdots \otimes I_\eta$.
The system is represented using a set of $\eta$ signed integer registers each of
size $3 n_p$ where $n_p =  \lceil \log (N^{1/3} + 1) \rceil$, with the factor of
3 accounting for the 3 spatial dimensions.

In the first quantized approach, fermion antisymmetry is encoded through initial
state preparation. Spin labels are also absent and should be accounted for
during state preparation. The cost of initial state preparation is typically
ignored.
"""
from functools import cached_property
from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
from attrs import frozen

from qualtran import Bloq, BloqBuilder, Register, Side, Signature, SoquetT
from qualtran.bloqs.and_bloq import And
from qualtran.bloqs.arithmetic import (
    Add,
    GreaterThan,
    Product,
    SignedIntegerToTwosComplement,
    SumOfSquares,
)
from qualtran.bloqs.basic_gates import Hadamard, Toffoli
from qualtran.bloqs.prepare_uniform_superposition import PrepareUniformSuperposition

if TYPE_CHECKING:
    from qualtran.resource_counting import SympySymbolAllocator


@frozen
class PrepareTUVSuperpositions(Bloq):
    r"""Prepare the equal superposition on registers selecting between T U and V.

    This will produce 3 qubits for flagging which term to apply. xx0 -> T, x0x -> U or V, 0xx -> V.

    Args:
        adjoint: whether to dagger the bloq or not.

    Registers:
        tuv: a single qubit rotated to appropriately weight T and U or V.
        uv: a single qubit rotated to appropriately weight U or V.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) page 15, section A
    """
    num_bits_t: int
    eta: int
    lambda_zeta: int
    num_bits_rot_aa: int = 8
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(tuv=1, uv=1)

    def short_name(self) -> str:
        return 'PREP TUV'

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        n_eta_zeta = (self.eta + 2 * self.lambda_zeta - 1).bit_length()
        # The cost arises from rotating a qubit, and uniform state preparation
        # over eta + 2 lambda_zeta numbers along.
        return {(self.num_bits_t + 4 * n_eta_zeta + 2 * self.num_bits_rot_aa - 12, Toffoli())}


@frozen
class PrepareIJSuperposition(Bloq):
    r"""Uniform superposition over $\eta$ values of $i$ and $j$ in unary such that $i \ne j$.

    Args:
        eta: The number of electrons.
        num_bits_rot_aa: The number of bits of precision for the single qubit
            rotation for amplitude amplification. Called $b_r$ in the reference.
        adjoint: whether to dagger the bloq or not.

    Registers:
        i: a n_eta bit register for unary encoding of eta numbers.
        j: a n_eta bit register for unary encoding of eta numbers.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) page 18, section A, around Eq 62.
    """
    eta: int
    num_bits_rot_aa: int
    adjoint: int = False

    @cached_property
    def signature(self) -> Signature:
        n_eta = (self.eta - 1).bit_length()
        return Signature([Register("ij", bitsize=n_eta, shape=(2,))])

    def short_name(self) -> str:
        return 'PREP ij'

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        n_eta = (self.eta - 1).bit_length()
        # Half of Eq. 62 which is the cost for prep and prep^\dagger
        return {((7 * n_eta + 4 * self.num_bits_rot_aa - 18), Toffoli())}


@frozen
class PreparePowerTwoState(Bloq):
    r"""Prepares the uniform superposition over $|r\rangle$ given by Eq. 69 in the reference.

    This prepares the state

    $$
        2^{(-n_p -1)/2} \sum_r=0^{n_p-2} 2^{r/2} |r\rangle
    $$

    in one-hot unary.

    Args:
        bitsize: the number of bits $n_p$ for the $|r\rangle$ register.

    Registers:
        r: The register we want to prepare the state over.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) Eq 67-69, pg 19-20
    """
    bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(r=self.bitsize)

    def short_name(self) -> str:
        return r'PREP $2^{r/2} |r\rangle$'

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        return {((self.bitsize - 2), Toffoli())}


@frozen
class PrepareTFirstQuantization(Bloq):
    r"""PREPARE for the kinetic energy operator for the first quantized chemistry Hamiltonian.

    This prepares the state

    $$
        |+\rangle\sum_{j=1}^{\eta}|j\rangle\sum_{w=0}^{2}|w\rangle
        \sum_{r=0}^{n_{p}-2}2^{r/2}|r\rangle
        \sum_{s=0}^{n_{p}-2}2^{s/2}|s\rangle
    $$

    We assume that the uniform superposition over ($i$ and) $j$ has already occured via
    PrepareIJSuperposition.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        eta: The number of electrons.
        num_bits_rot_aa: The number of bits of precision for the single qubit
            rotation for amplitude amplification. Called $b_r$ in the reference.
        adjoint: whether to dagger the bloq or not.

    Registers:
        w: a register to index one of three components of the momenta.
        r: a register encoding bits for each component of the momenta.
        s: a register encoding bits for each component of the momenta.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) page 19, section B
    """

    num_bits_p: int
    eta: int
    num_bits_rot_aa: int = 8
    adjoint: int = False

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(w=2, r=self.num_bits_p, s=self.num_bits_p)

    def short_name(self) -> str:
        return r'PREP $T$'

    def build_composite_bloq(
        self, bb: BloqBuilder, w: SoquetT, r: SoquetT, s: SoquetT
    ) -> Dict[str, 'SoquetT']:
        w = bb.add(PrepareUniformSuperposition(3), target=w)
        r = bb.add(PreparePowerTwoState(self.num_bits_p), r=r)
        s = bb.add(PreparePowerTwoState(self.num_bits_p), r=s)
        return {'w': w, 'r': r, 's': s}

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # there is a cost for the uniform state preparation for the $w$
        # register. Adding a bloq is sort of overkill, should just tag the
        # correct cost on UniformSuperPosition bloq
        # 13 is from assuming 8 bits for the rotation, and n = 2.
        uni_prep_w = (13, Toffoli())
        # Factor of two for r and s registers.
        return {uni_prep_w, (2, PreparePowerTwoState(bitsize=self.num_bits_p))}


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
            [
                Register("mu", bitsize=self.num_bits_p),
                Register("flag_mu", bitsize=1, side=Side.RIGHT),
            ]
        )

    def short_name(self) -> str:
        return r'PREP $\sqrt{2^\mu}|\mu\rangle$'

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # controlled hadamards which cannot be inverted at zero Toffoli cost.
        return {((self.num_bits_p - 1), Toffoli())}


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
    adjoint: int = False

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [Register("mu", self.num_bits_p), Register("nu", self.num_bits_p + 1, shape=(3,))]
        )

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f'PrepSupNu{dag}'

    def short_name(self) -> str:
        return r'PREP $2^{-\mu}|\mu\rangle|\nu\rangle$'

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # controlled hadamards which cannot be inverted at zero Toffoli cost.
        return {((3 * (self.num_bits_p - 1)), Toffoli())}


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
                Register("nu", self.num_bits_p + 1, shape=(3,)),
                Register("flag_minus_zero", 1, side=Side.RIGHT),
            ]
        )

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f'FlagZero{dag}'

    def short_name(self) -> str:
        return r'$\nu\ne -0$'

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        if self.adjoint:
            # This can be inverted with cliffords.
            return {(0, Toffoli())}
        else:
            # Controlled Toffoli each having n_p + 1 controls and 2 Toffolis to
            # check the result of the Toffolis.
            return {((3 * self.num_bits_p + 2), Toffoli())}


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
                Register("mu", self.num_bits_p),
                Register("nu", self.num_bits_p + 1, shape=(3,)),
                Register("flag_nu_lt_mu", 1, side=Side.RIGHT),
            ]
        )

    def short_name(self) -> str:
        return r'$\nu < 2^{\mu-2}$'

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        if self.adjoint:
            # This can be inverted with cliffords.
            return {(0, Toffoli())}
        else:
            # n_p controlled Toffolis with four controls.
            return {(3 * self.num_bits_p, Toffoli())}


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
        flag_minus_zero: A flag from checking for negative zero.
        flag_nu_lt_mu: A flag from checking $\nu \lt 2^{\mu -2}$.
        flag_ineq: A flag qubit from the inequality test.
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
                Register("mu", self.num_bits_p),
                Register("nu", self.num_bits_p + 1, shape=(3,)),
                Register("m", self.num_bits_m),
                Register("flag_mu_prep", 1, side=Side.LEFT),
                Register("flag_minus_zero", 1, side=Side.LEFT),
                Register("flag_nu_lt_mu", 1, side=Side.LEFT),
                Register("succ", 1),
            ]
        )

    def short_name(self) -> str:
        return r'$(2^{\mu-2})^2\mathcal{M} > m \nu^2 $'

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        if self.adjoint:
            return {(0, Toffoli())}
        else:
            # 1. Compute $\nu_x^2 + \nu_y^2 + \nu_z^2$
            cost_1 = (1, SumOfSquares(self.num_bits_p, k=3))
            # 2. Compute $m (\nu_x^2 + \nu_y^2 + \nu_z^2)$
            cost_2 = (1, Product(2 * self.num_bits_p + 2, self.num_bits_m))
            # 3. Inequality test
            cost_3 = (1, GreaterThan(self.num_bits_m, 2 * self.num_bits_p + 2))
            # 4. 3 Toffoli for overall success
            cost_4 = (3, Toffoli())
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
                Register("mu", bitsize=self.num_bits_p),
                Register("nu", bitsize=self.num_bits_p + 1, shape=(3,)),
                Register("m", bitsize=n_m),
                Register("succ_nu", bitsize=1),
            ]
        )

    def short_name(self) -> str:
        return r"PREP $\frac{1}{\lVert \nu \rVert} |\nu\rangle $"

    def build_composite_bloq(
        self, bb: BloqBuilder, mu: SoquetT, nu: SoquetT, m: SoquetT, succ_nu: SoquetT
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
        mu, nu, m, succ_nu = bb.add(
            TestNuInequality(self.num_bits_p, n_m, adjoint=self.adjoint),
            mu=mu,
            nu=nu,
            m=m,
            flag_mu_prep=flag_mu,
            flag_minus_zero=flag_zero,
            flag_nu_lt_mu=flag_nu_lt_mu,
            succ=succ_nu,
        )
        return {'mu': mu, 'nu': nu, 'm': m, 'succ_nu': succ_nu}

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # 1. Prepare unary encoded superposition state (Eq 77)
        cost_1 = (1, PrepareMuUnaryEncodedOneHot(self.num_bits_p))
        n_m = (self.m_param - 1).bit_length()
        # 2. Prepare mu-nu superposition (Eq 78)
        cost_2 = (1, PrepareNuSuperPositionState(self.num_bits_p))
        # 3. Remove minus zero
        cost_3 = (1, FlagZeroAsFailure(self.num_bits_p, adjoint=self.adjoint))
        # 4. Test $\nu < 2^{\mu-2}$
        cost_4 = (1, TestNuLessThanMu(self.num_bits_p, adjoint=self.adjoint))
        # 5. Prepare superposition over $m$ which is a power of two so only clifford.
        # 6. Test that $(2^{\mu-2})^2\mathcal{M} > m (\nu_x^2 + \nu_y^2 + \nu_z^2)$
        cost_6 = (1, TestNuInequality(self.num_bits_p, n_m, adjoint=self.adjoint))
        return {cost_1, cost_2, cost_3, cost_4, cost_6}


@frozen
class PrepareZetaState(Bloq):
    r"""PREPARE the superpostion over $l$ weighted by $\zeta_l$.

    This is apparently NOT just generic state preparation and there are some
    tricks I don't understand.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        eta: The number of electrons.
        m_param: $\mathcal{M}$ in the reference.
        lambda_zeta: sum of nuclear charges.
        er_lambda_zeta: eq 91 of the referce. Cost of erasing qrom.

    Registers:

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) page 23-24, last 3 paragraphs.
    """
    num_atoms: int
    lambda_zeta: int
    num_bits_nuc_pos: int
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register("l", bitsize=(self.num_atoms - 1).bit_length())])

    def short_name(self) -> str:
        return r'PREP $\zeta_\ell |\ell\rangle$'

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        if self.adjoint:
            # Really Er(x), eq 91. In practice we will reaplce this with the
            # appropriate qrom call down the line.
            return {(int(np.ceil(self.lambda_zeta**0.5)), Toffoli())}
        else:
            return {(self.lambda_zeta, Toffoli())}


@frozen
class PrepareUVFistQuantization(Bloq):
    r"""PREPARE the U and V parts of the Hamiltonian.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        eta: The number of electrons.
        num_atoms: The number of atoms. $L$ in the reference.
        m_param: $\mathcal{M}$ in the reference.
        lambda_zeta: sum of nuclear charges.
        num_bits_nuc_pos: The number of bits of precision for representing the nuclear coordinates.

    Registers:
        mu: The state controlling the nested boxes procedure.
        nu: The momentum transfer register.
        m: an ancilla register in a uniform superposition.
        l: The register for atomic species.
        flag_nu: Flag for success of the state preparation.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) page 19, section B
    """
    num_bits_p: int
    eta: int
    num_atoms: int
    m_param: int
    lambda_zeta: int
    num_bits_nuc_pos: int
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        n_nu = self.num_bits_p + 1
        n_m = (self.m_param - 1).bit_length()
        n_at = (self.num_atoms - 1).bit_length()
        return Signature(
            [
                Register("mu", bitsize=self.num_bits_p),
                Register("nu", bitsize=n_nu, shape=(3,)),
                Register("m", bitsize=n_m),
                Register("l", bitsize=n_at),
                Register("succ_nu", bitsize=1),
            ]
        )

    def short_name(self) -> str:
        return r'PREP $UV$'

    def build_composite_bloq(
        self, bb: BloqBuilder, mu: SoquetT, nu: SoquetT, m: SoquetT, l: SoquetT, succ_nu: SoquetT
    ) -> Dict[str, 'SoquetT']:
        mu, nu, m, succ_nu = bb.add(
            PrepareNuState(self.num_bits_p, self.m_param, adjoint=self.adjoint),
            mu=mu,
            nu=nu,
            m=m,
            succ_nu=succ_nu,
        )
        l = bb.add(
            PrepareZetaState(
                self.num_atoms, self.lambda_zeta, self.num_bits_nuc_pos, adjoint=self.adjoint
            ),
            l=l,
        )
        return {'mu': mu, 'nu': nu, 'm': m, 'l': l, 'succ_nu': succ_nu}

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # 1. Prepare the nu state
        # 2. Prepare the zeta_l state
        return {
            (1, PrepareNuState(self.num_bits_p, self.m_param, self.adjoint)),
            (
                1,
                PrepareZetaState(
                    self.num_atoms, self.lambda_zeta, self.num_bits_nuc_pos, self.adjoint
                ),
            ),
        }


@frozen
class SelectTFirstQuantization(Bloq):
    r"""SELECT for the kinetic energy operator for the first quantized chemistry Hamiltonian.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        eta: The number of electrons.

    Registers:
        p: The ancilla storing the ith electrons momentum.
        plus: A $|+\rangle$ state.
        flag_T: a flag to control on the success of the $T$ state preparation.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) page 20, section B
    """
    num_bits_p: int
    eta: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("p", bitsize=self.num_bits_p, shape=(3,)),
                Register("plus", bitsize=1),
                Register("flag_T", bitsize=1),
                Register("w", bitsize=2),
                Register("r", bitsize=self.num_bits_p),
                Register("s", bitsize=self.num_bits_p),
            ]
        )

    def short_name(self) -> str:
        return r'SEL $T$'

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        plus: SoquetT,
        flag_T: SoquetT,
        w: SoquetT,
        r: SoquetT,
        s: SoquetT,
        p: SoquetT,
    ) -> Dict[str, 'SoquetT']:
        # 1. Use w to control copying of component w of p into ancilla
        junk_p = bb.split(bb.allocate(self.num_bits_p))
        # 0 0 = x component
        w, out0 = bb.add(And(0, 0), ctrl=w)
        px = p[0].split()
        for ibit in range(len(px)):
            bb.add(Toffoli(), [out0, px[ibit]], junk_p[ibit])
        return {'plus': plus, 'flag_t': flag_T, 'w': w, 'r': r, 's': s, 'p': p}

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # Cost is $5(n_{p} - 1) + 2$ which comes from copying each $w$ component of $p$
        # into an ancilla register ($3(n_{p}-1)$), copying the $r$ and $s$ bit of into an
        # ancilla ($2(n_{p}-1)$), controlling on both those bit perform phase flip on an
        # ancilla $|+\rangle$ state. This requires $1$ Toffoli, Then erase which costs
        # only Cliffords. There is an additional control bit controlling the application
        # of $T$ thus we come to our total.
        # Eq 73. page
        return {((5 * (self.num_bits_p - 1) + 2), Toffoli())}


@frozen
class ApplyNuclearPhase(Bloq):
    r"""Apply the phase factor $-e^{-ik_\nu\cdot R_\ell}$ to the state.

    Args:
        num_bits_p: Number of bits for the momentum register.
        num_bits_nuc: Number of bits of precision for the nuclear positions.

    Registers:
        l: A register indexing the nuclear positions.
        rl: A register storing the value of $R_\ell$.
        nu: The momentum transfer register.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) pg 25, paragraph 2.
    """

    num_bits_p: int
    num_bits_nuc: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("l", bitsize=self.num_bits_nuc),
                Register("rl", bitsize=self.num_bits_nuc),
                Register("nu", bitsize=self.num_bits_p, shape=(3,)),
            ]
        )

    def short_name(self) -> str:
        return r'$-e^{-k_\nu\cdot R_l$'

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        n_p = self.num_bits_p
        n_n = self.num_bits_nuc
        # This is some complicated application of phase gradient gates.
        # Eq. 97.
        if n_n > n_p:
            cost = 3 * (2 * n_p * n_n - n_p * (n_p + 1) - 1)
        else:
            cost = 3 * n_n * (n_n - 1)
        return {(cost, Toffoli())}


@frozen
class SelectUVFirstQuantization(Bloq):
    r"""SELECT for the U and V operators for the first quantized chemistry Hamiltonian.

    This does not include the controlled swaps from p_i and q_j system registers
    into ancilla registers and back again. Hence there is no system register.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        eta: The number of electrons.
        num_bits_nuc_pos: The number of bits to store each component of the
            nuclear positions. $n_R$ in the reference.

    Registers:

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767)
    """

    num_bits_p: int
    eta: int
    num_bits_nuc_pos: int

    @cached_property
    def signature(self) -> Signature:
        n_nu = self.num_bits_p + 1
        return Signature(
            [
                Register("flag_tuv", bitsize=1),
                Register("flag_uv", bitsize=1),
                Register("l", bitsize=self.num_bits_nuc_pos),
                Register("rl", bitsize=self.num_bits_nuc_pos),
                Register("nu", bitsize=n_nu, shape=(3,)),
                Register("p", bitsize=n_nu, shape=(3,)),
                Register("q", bitsize=n_nu, shape=(3,)),
            ]
        )

    def short_name(self) -> str:
        return r'SEL $UV$'

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        flag_tuv: SoquetT,
        flag_uv: SoquetT,
        l: SoquetT,
        rl: SoquetT,
        nu: SoquetT,
        p: SoquetT,
        q: SoquetT,
    ) -> Dict[str, 'SoquetT']:
        num_bits_nu = self.num_bits_p + 1
        # bb.allocate()
        for i in range(3):
            p[i] = bb.add(SignedIntegerToTwosComplement(num_bits_nu), x=p[i])
            # should be controlled on V only
            p[i] = bb.add(Add(num_bits_nu), a=nu[i], b=p[i])
            p[i] = bb.add(SignedIntegerToTwosComplement(num_bits_nu), x=p[i])
            q[i] = bb.add(SignedIntegerToTwosComplement(num_bits_nu), x=q[i])
            # should be controlled on U or V only
            # flip bits and add one.
            q[i] = bb.add(Add(num_bits_nu), a=nu[i], b=q[i])
            q[i] = bb.add(SignedIntegerToTwosComplement(num_bits_nu), x=q[i])
        l, rl, nu = bb.add(
            ApplyNuclearPhase(self.num_bits_p, self.num_bits_nuc_pos), l=l, rl=rl, nu=nu
        )
        return {
            'flag_tuv': flag_tuv,
            'flag_uv': flag_uv,
            'l': l,
            'rl': rl,
            'nu': nu,
            'p': p,
            'q': q,
        }

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        cost_tc = (6, SignedIntegerToTwosComplement(self.num_bits_p))
        cost_add = (6, Add(self.num_bits_p + 1))  # + 2?
        cost_ctrl_add = (6 * (self.num_bits_p + 1), Toffoli())
        # + 2 as these numbers are larger from addition of $\nu$
        cost_inv_tc = (6, SignedIntegerToTwosComplement(self.num_bits_p + 2))
        # 2. Phase by $e^{ik\cdot R}$ in the case of $U$ only.
        cost_phase = (1, ApplyNuclearPhase(self.num_bits_p, self.num_bits_nuc_pos))
        return {cost_tc, cost_add, cost_ctrl_add, cost_inv_tc, cost_phase}


@frozen
class PrepareFirstQuantization(Bloq):
    """State preparation for the first quantized chemistry Hamiltonian."""

    num_bits_p: int
    eta: int
    num_atoms: int
    lambda_zeta: int
    m_param: int = 2**8
    num_bits_nuc_pos: int = 16
    num_bits_t: int = 16
    num_bits_rot_aa: int = 8
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        n_nu = self.num_bits_p + 1
        n_eta = (self.eta - 1).bit_length()
        n_at = (self.num_atoms - 1).bit_length()
        n_m = (self.m_param - 1).bit_length()
        return Signature(
            [
                Register("tuv", bitsize=1),
                Register("uv", bitsize=1),
                Register("plus_t", bitsize=1),
                Register("ij", bitsize=n_eta, shape=(2,)),
                Register("w", bitsize=3),
                Register("r", bitsize=self.num_bits_p),
                Register("s", bitsize=self.num_bits_p),
                Register("mu", bitsize=self.num_bits_p),
                Register("nu", bitsize=n_nu, shape=(3,)),
                Register("m", bitsize=n_m),
                Register("succ_nu", bitsize=1),
                Register("l", bitsize=n_at),
            ]
        )

    def short_name(self) -> str:
        return r'PREP'

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        tuv: SoquetT,
        uv: SoquetT,
        plus_t: SoquetT,
        ij: SoquetT,
        w: SoquetT,
        r: SoquetT,
        s: SoquetT,
        mu: SoquetT,
        nu: SoquetT,
        m: SoquetT,
        succ_nu: SoquetT,
        l: SoquetT,
    ) -> Dict[str, 'SoquetT']:
        tuv, uv = bb.add(
            PrepareTUVSuperpositions(
                self.num_bits_t,
                self.eta,
                self.lambda_zeta,
                self.num_bits_rot_aa,
                adjoint=self.adjoint,
            ),
            tuv=tuv,
            uv=uv,
        )
        ij = bb.add(PrepareIJSuperposition(self.eta, self.num_bits_rot_aa, self.adjoint), ij=ij)
        # |+>
        plus_t = bb.add(Hadamard(), q=plus_t)
        w, r, s = bb.add(
            PrepareTFirstQuantization(
                self.num_bits_p, self.eta, self.num_bits_rot_aa, adjoint=self.adjoint
            ),
            w=w,
            r=r,
            s=s,
        )
        mu, nu, m, l, succ_nu = bb.add(
            PrepareUVFistQuantization(
                self.num_bits_p,
                self.eta,
                self.num_atoms,
                self.m_param,
                self.lambda_zeta,
                self.num_bits_nuc_pos,
                adjoint=self.adjoint,
            ),
            mu=mu,
            nu=nu,
            m=m,
            l=l,
            succ_nu=succ_nu,
        )
        return {
            'tuv': tuv,
            'uv': uv,
            'plus_t': plus_t,
            'ij': ij,
            'w': w,
            'r': r,
            's': s,
            'mu': mu,
            'nu': nu,
            'm': m,
            'l': l,
            'succ_nu': succ_nu,
        }


@frozen
class SWAPIJ(Bloq):
    """Placeholder for swap combined with unary iteration over i/j."""

    eta: int
    trg_bitsize: int

    @cached_property
    def signature(self) -> Signature:
        sel_bitsize = (self.eta - 1).bit_length()
        return Signature(
            [
                Register("sel", bitsize=sel_bitsize),
                Register("sys", bitsize=self.trg_bitsize, shape=(self.eta, 3)),
                Register("anc", bitsize=self.trg_bitsize, shape=(3)),
            ]
        )

    def short_name(self) -> str:
        return r'SWAP'

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        return {(3 * self.eta * self.trg_bitsize + self.eta - 2, Toffoli())}


@frozen
class SelectFirstQuantization(Bloq):
    """State preparation for the first quantized chemistry Hamiltonian."""

    num_bits_p: int
    eta: int
    num_atoms: int
    lambda_zeta: int
    m_param: int = 2**8
    num_bits_nuc_pos: int = 16
    num_bits_t: int = 16
    num_bits_rot_aa: int = 8
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        n_nu = self.num_bits_p + 1
        n_eta = (self.eta - 1).bit_length()
        n_at = (self.num_atoms - 1).bit_length()
        n_m = (self.m_param - 1).bit_length()
        return Signature(
            [
                Register("tuv", bitsize=1),
                Register("uv", bitsize=1),
                Register("plus_t", bitsize=1),
                Register("ij", bitsize=n_eta, shape=(2,)),
                Register("w", bitsize=3),
                Register("r", bitsize=self.num_bits_p),
                Register("s", bitsize=self.num_bits_p),
                Register("mu", bitsize=self.num_bits_p),
                Register("nu", bitsize=n_nu, shape=(3,)),
                Register("m", bitsize=n_m),
                Register("succ_nu", bitsize=1),
                Register("l", bitsize=n_at),
                Register("rl", bitsize=self.num_bits_nuc_pos),
                Register("sys", bitsize=self.num_bits_p, shape=(self.eta, 3)),
                Register("pq", bitsize=self.num_bits_p, shape=(2, 3)),
            ]
        )

    def short_name(self) -> str:
        return r'SELECT'

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        tuv: SoquetT,
        uv: SoquetT,
        plus_t: SoquetT,
        ij: SoquetT,
        w: SoquetT,
        r: SoquetT,
        s: SoquetT,
        mu: SoquetT,
        nu: SoquetT,
        m: SoquetT,
        succ_nu: SoquetT,
        l: SoquetT,
        rl: SoquetT,
        sys: SoquetT,
        pq: SoquetT,
    ) -> Dict[str, 'SoquetT']:
        p, q = pq
        i, j = ij
        i, sys, p = bb.add(SWAPIJ(self.eta, self.num_bits_p + 1), sel=i, sys=sys, anc=p)
        j, sys, q = bb.add(SWAPIJ(self.eta, self.num_bits_p + 1), sel=j, sys=sys, anc=q)
        p, plus_t, tuv, w, r, s = bb.add(
            SelectTFirstQuantization(self.num_bits_p, self.eta),
            p=p,
            plus=plus_t,
            flag_T=tuv,
            w=w,
            r=r,
            s=s,
        )
        tuv, uv, l, rl, nu, p, q = bb.add(
            SelectUVFirstQuantization(self.num_bits_nuc_pos, self.eta, self.num_bits_nuc_pos),
            flag_tuv=tuv,
            flag_uv=uv,
            l=l,
            rl=rl,
            nu=nu,
            p=p,
            q=q,
        )
        i, sys, p = bb.add(SWAPIJ(self.eta, self.num_bits_p + 1), sel=i, sys=sys, anc=p)
        j, sys, q = bb.add(SWAPIJ(self.eta, self.num_bits_p + 1), sel=j, sys=sys, anc=q)
        pq = [p, q]
        ij = [i, j]
        return {
            'tuv': tuv,
            'uv': uv,
            'plus_t': plus_t,
            'ij': ij,
            'w': w,
            'r': r,
            's': s,
            'mu': mu,
            'nu': nu,
            'm': m,
            'l': l,
            'rl': rl,
            'succ_nu': succ_nu,
            'sys': sys,
            'pq': pq,
        }
