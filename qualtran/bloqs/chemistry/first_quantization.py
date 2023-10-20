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

from attrs import frozen

from qualtran import Bloq, Register, Signature
from qualtran.bloqs.basic_gates import TGate

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
        # correct cost on cirq_ft bloq, 13 is from assuming 8 bits for the rotation, and n = 2.
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
class PrepareUnaryEncodedOneHot(Bloq):
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
    adjoint: False

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
            return {(4 * 3 * self.num_bits_p, TGate())}

@frozen
class PrepareUVFistQuantization(Bloq):
    r"""PREPARE for the kinetic energy operator for the first quantized chemistry Hamiltonian.

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

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) page 19, section B
    """
    num_bits_p: int
    eta: int
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
        # 1. Prepare unary encoded superposition state
        n_m = (self.m_param - 1).bit_length()
        if self.adjoint:
            cost = 14 * self.num_bits_p - 8  # eq 90 - 89 page 23
            cost += cost_qrom_erasure(self.lambda_zeta)
        else:
            cost = (
                3 * self.num_bits_p**2 + self.num_bits_p + 1 + 4 * n_m * (self.num_bits_p + 1)
            )  # eq 89, page 23
            cost += self.lambda_zeta  # Eq 92.


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
        (https://arxiv.org/abs/2105.12767) page 19, section B
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
