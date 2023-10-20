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
"""SELECT and PREPARE for the first quantized chemistry Hamiltonian."""
from functools import cached_property
from typing import Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
from attrs import frozen

from qualtran import Bloq, Register, Signature
from qualtran.bloqs.arithmetic import GreaterThan, Product, SumOfSquares
from qualtran.bloqs.basic_gates import TGate, Toffoli

if TYPE_CHECKING:
    from qualtran.resource_counting import SympySymbolAllocator


@frozen
class UniformSuperPostionIJFirstQuantization(Bloq):
    r"""Uniform superposition over $\eta$ values of $i$ and $j$ in unary such that $i \ne j$.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        eta: The number of electrons.
        num_bits_rot_aa: The number of bits of precision for the single qubit
            rotation for amplitude amplification. Called $b_r$ in the reference.

    Registers:
        i: a n_eta bit register for unary encoding of eta numbers.
        j: a n_eta bit register for unary encoding of eta numbers.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) page 19, section B
    """
    num_plane_wave: int
    eta: int
    num_bits_rot_aa: int

    @cached_property
    def signature(self) -> Signature:
        n_eta = (self.eta - 1).bit_length()
        return Signature.build(i=n_eta, j=n_eta)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        n_eta = (self.eta - 1).bit_length()
        # Half of Eq. 62 which is the cost for prep and prep^\dagger
        return {(4 * (7 * n_eta + 4 * self.num_bits_rot_aa - 18), TGate())}


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
        (https://arxiv.org/abs/2105.12767) Eq 66-69, pg 19-20
    """
    bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(r=self.bitsize)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        return {(4 * (self.bitsize - 2), TGate())}


@frozen
class PrepareTFirstQuantization(Bloq):
    r"""PREPARE for the kinetic energy operator for the first quantized chemistry Hamiltonian.

    This prepares the state

    $$
        |+\rangle\sum_{j=1}^{\eta}|j\rangle\sum_{w=0}^{2}|w\rangle
        \sum_{r=0}^{n_{p}-2}2^{r/2}|r\rangle
        \sum_{s=0}^{n_{p}-2}2^{s/2}|s\rangle
    $$

    We assume that the uniform superposition over $j$ has already occured via
    UniformSuperPositionIJFirstQuantization.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        eta: The number of electrons.
        num_bits_rot_aa: The number of bits of precision for the single qubit
            rotation for amplitude amplification. Called $b_r$ in the reference.

    Registers:

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) page 19, section B
    """

    num_bits_p: int
    eta: int
    num_bits_rot_aa: int = 8
    adjoint: int = False

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(plus=1, w=2, r=self.num_bits_p - 2, s=self.num_bits_p - 2)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # there is a cost for the uniform state preparation for the $w$
        # register. Adding a bloq is sort of overkill, should just tag the
        # correct cost on UniformSuperPosition bloq
        # 13 is from assuming 8 bits for the rotation, and n = 2.
        uni_prep_w = (4 * 13, TGate())
        # Factor of two for r and s registers.
        return {uni_prep_w, (2, PreparePowerTwoState(bitsize=self.num_bits_p))}


@frozen
class SelectTFirstQuantization(Bloq):
    r"""SELECT for the kinetic energy operator for the first quantized chemistry Hamiltonian.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        eta: The number of electrons.

    Registers:
        sys: The system register.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) page 20, section B
    """
    num_bits_p: int
    eta: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("sys", bitsize=self.num_bits_p, shape=(self.eta, 3)),
                Register("plus", bitsize=1),
                Register("flag_T", bitsize=1),
            ]
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # Cost is $5(n_{p} - 1) + 2$ which comes from copying each $w$ component of $p$
        # into an ancilla register ($3(n_{p}-1)$), copying the $r$ and $s$ bit of into an
        # ancilla ($2(n_{p}-1)$), controlling on both those bit perform phase flip on an
        # ancilla $|+\rangle$ state. This requires $1$ Toffoli, Then erase which costs
        # only Cliffords. There is an additional control bit controlling the application
        # of $T$ thus we come to our total.
        # Eq 73. page
        return {(4 * (5 * (self.num_bits_p - 1) + 2), TGate())}


@frozen
class PrepareMuUnaryEncodedOneHot(Bloq):
    r"""Prepare a unary encoded one-hot superposition state.

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
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) page 21, Eq 77.
    """
    num_bits_p: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(mu=self.num_bits_p + 1)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # controlled hadamards which cannot be inverted at zero Toffoli cost.
        return {(4 * (self.num_bits_p - 1), TGate())}


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
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) page 21, Eq 78.
    """
    num_bits_p: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [Register("mu", self.num_bits_p + 1), Register("nu", self.num_bits_p + 1, shape=(3,))]
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # controlled hadamards which cannot be inverted at zero Toffoli cost.
        return {(4 * (3 * (self.num_bits_p - 1)), TGate())}


@frozen
class FlagZeroAsFailure(Bloq):
    r"""Bloq to flag if minus zero appears in the $\nu$ state.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.

    Registers:
        nu: the momentum transfer register.
        flag_minus_zero: a flag bit for failure of the state preparation.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) page 21, Eq 80.
    """
    num_bits_p: int
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [Register("nu", self.num_bits_p + 1, shape=(3,)), Register("flag_minus_zero", 1)]
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        if self.adjoint:
            # This can be inverted with cliffords.
            return {(0, TGate())}
        else:
            # Controlled Toffoli each having n_p + 1 controls and 2 Toffolis to
            # check the result of the Toffolis.
            return {(4 * (3 * self.num_bits_p + 2), TGate())}


@frozen
class TestNuLessThanMu(Bloq):
    r"""Bloq to flag if all components of $\nu$ are smaller in absolute value than $2^{\mu-2}$.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.

    Registers:
        nu: the momentum transfer register.
        flag_nu_lt_mu: a flag bit for failure of the state preparation.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) page 21, Eq 80.
    """
    num_bits_p: int
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [Register("mu", self.num_bits_p + 1), Register("nu", self.num_bits_p + 1, shape=(3,))]
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        if self.adjoint:
            # This can be inverted with cliffords.
            return {(0, TGate())}
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
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) page 21, Eq 80.
    """
    num_bits_p: int
    num_bits_m: int
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("mu", self.num_bits_p + 1),
                Register("nu", self.num_bits_p + 1, shape=(3,)),
                Register("m", self.num_bits_m, shape=(3,)),
                Register("flag_minus_zero", 1),
                Register("flag_nu_lt_mu", 1),
                Register("flag_ineq", 1),
                Register("succ", 1),
            ]
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        if self.adjoint:
            return {(0, TGate())}
        else:
            # 1. Compute $\nu_x^2 + \nu_y^2 + \nu_z^2$
            cost_1 = (1, SumOfSquares(self.num_bits_p, k=3))
            # 2. Compute $m (\nu_x^2 + \nu_y^2 + \nu_z^2)$
            cost_2 = (1, Product(self.num_bits_p, self.num_bits_m))
            # 3. Inequality test
            cost_3 = (1, GreaterThan(2 * self.num_bits_p + self.num_bits_m + 2))
            # 4. 3 Toffoli for overall success
            cost_4 = (3, Toffoli())


@frozen
class PrepareNuState(Bloq):
    r"""PREPARE for the $\nu$ state for the $U$ and $V$ components of the Hamiltonian.

    Prepares a state of the form

    $$
        \frac{1}{2\mathcal{M}2^{self.num_bits_p +2 }}
        \sum_{\mu=2}^{self.num_bits_p+1}\sum_{\nu B_\mu}
        \sum_{m=0}^{\lceil \mathcal M(2^{\mu-2}/|\nu|)^2\rceil-1}
        \frac{1}{2^\mu}|\mu\rangle|\nu_x|rangle|\nu_y\rangle|\nu_z\rangle|m\rangle|0\rangle
    $$

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        eta: The number of electrons.
        m_param: $\mathcal{M}$ in the reference.
        lambda_zeta: sum of nuclear charges.
        er_lambda_zeta: eq 91 of the referce. Cost of erasing qrom.

    Registers:
        mu: The state controlling the nested boxes procedure.
        nu: The momentum transfer register.
        m: an ancilla register in a uniform superposition.
        flag_nu: Flag for success of the state preparation.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) page 19, section B
    """
    num_bits_p: int
    eta: int
    m_param: int
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        # this is for the nu register which lives on a grid of twice the size
        # the nu grid is twice as large, so one more bit is needed
        n_nu = (self.num_bits_p - 1).bit_length() + 2
        n_mu = self.num_bits_p.bit_length()  # is this correct?
        n_m = (self.m_param - 1).bit_length()
        return Signature(
            [
                Register("mu", bitsize=n_mu),
                Register("nu", bitsize=n_nu, shape=(3,)),
                Register("m", bitsize=n_m),
                Register("l", bitsize=self.num_bits_nuc_pos),
                Register("flag_nu", bitsize=1),
            ]
        )

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
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) page 23-24, last 3 paragraphs.
    """
    num_atoms: int
    lambda_zeta: int
    num_bits_nuc_pos: int
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register("l", bitsize=(self.num_atoms - 1).bitsize())])

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        if self.adjoint:
            # Really Er(x), eq 91. In practice we will reaplce this with the
            # appropriate qrom call down the line.
            return np.int(np.ceil(self.lambda_zeta**0.5))
        else:
            return self.lambda_zeta


@frozen
class PepareUVFirstQuantization(Bloq):
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
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) page 19, section B
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
        # this is for the nu register which lives on a grid of twice the size
        # the nu grid is twice as large, so one more bit is needed
        n_nu = (self.num_bits_p - 1).bit_length() + 2
        n_mu = self.num_bits_p.bit_length()
        #
        n_m = (self.m_param - 1).bit_length()
        return Signature(
            [
                Register("mu", bitsize=n_mu),
                Register("nu", bitsize=n_nu, shape=(3,)),
                Register("m", bitsize=n_m),
                Register("l", bitsize=self.num_bits_nuc_pos),
                Register("flag_nu", bitsize=1),
            ]
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # 1. Prepare the nu state
        # 2. Prepare the zeta_l state
        return {
            (1, PrepareNuState(self.num_bits_p, self.eta, self.m_param, self.adjoint)),
            (
                1,
                PrepareZetaState(
                    self.num_atoms, self.lambda_zeta, self.num_bits_nuc_pos, self.adjoint
                ),
            ),
        }


@frozen
class SelectUVFirstQuantization(Bloq):
    r"""SELECT for the kinetic energy operator for the first quantized chemistry Hamiltonian.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        eta: The number of electrons.
        num_bits_nuc_pos: The number of bits to store each component of the
            nuclear positions. $n_R$ in the reference.

    Registers:

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767)
    """

    num_bits_p: int
    eta: int
    num_bits_nuc_pos: int

    @cached_property
    def signature(self) -> Signature:
        n_nu = self.num_bits_p + 1
        n_eta = (self.eta - 1).bit_length()
        return Signature(
            [
                Register("flag_U", bitsize=1),
                Register("flag_UorV", bitsize=1),
                Register("plus", bitsize=1),
                Register("ij", bitsize=n_eta, shape=(2,)),
                Register("l", bitsize=self.num_bits_nuc_pos),
                Register("nu", bitsize=n_nu, shape=(3,)),
                Register("sys", bitsize=self.num_bits_p, shape=(self.eta, 3)),
                # + some ancilla for the controlled swaps of system registers.
            ]
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # Addition and subtraction of $\nu$ to p and q.
        cost_add = 24 * self.num_bits_p  # Eq 93.
        # phasing by the structure factor $-e^{i k_{\nu} \cdot R_{\ell}}$ costs $6 n_{p} n_{R}$
        cost_phase = 6 * self.num_bits_p * self.num_bits_nuc_pos  # Eq 97.
        return {(4 * (cost_add + cost_phase), TGate())}
