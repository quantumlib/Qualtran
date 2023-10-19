# ## System Register
# The system Register is represented as signed integers for each Euler direction.
# for $\eta$ electrons we use $n_{p}$ bits for each of the $xyz$-directions which means the system register is $3 \eta n_{p}$ bits wide.  We will need additional ancilla registers for PREPARE and SELECT.
# We will be moving $i$ and $j$ electron registers into a working register which for $T$ $U$ and $V$ costs
# $12 \eta n_{p} + 4\eta - 8$. This comes from the core $12 \eta n_{p}$ plus 4 factors of $\eta - 2$ unary iteration.
# [1] $n_{\eta\zeta} + 2 n_{\eta} + 6 n_{p} + n_{\mathcal{M}} + 16$
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

from attrs import field, frozen

from qualtran import Bloq, BloqBuilder, Register, Signature, SoquetT
from qualtran.bloqs.basic_gates import TGate

if TYPE_CHECKING:
    from qualtran.resource_counting import SympySymbolAllocator


@frozen
class UniformSuperPostionIJFirstQuantization(Bloq):
    r"""Uniform superposition over $\eta$ values of $i$ and $j$ in unary.

    Args:
        num_pw_each_dim: The number of planewaves in each of the x, y and z
            directions. In total, for a cubic box, there are N = num_pw_each_dim**3
            planewaves. The number of bits required (in each dimension)
            is thus $\log N^1/3 + 1$, where the + 1 is for the sign bit.
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
        return {4 * (7 * n_eta + 4 * self.num_bits_rot_aa - 18, TGate())}


@frozen
class PrepareTFirstQuantization(Bloq):
    r"""PREPARE for the kinetic energy operator for the first quantized chemistry Hamiltonian.

    This prepares the state

    $$
        |+\rangle\sum_{j=1}^{\eta}|j\rangle\sum_{w=0}^{2}|w\rangle
        \sum_{r=0}^{n_{p}-2}2^{r/2}|r\rangle
        \sum_{s=0}^{n_{p}-2}2^{s/2}|s\rangle
    $$

    Args:
        num_pw_each_dim: The number of planewaves in each of the x, y and z
            directions. In total, for a cubic box, there are N = num_pw_each_dim**3
            planewaves. The number of bits required (in each dimension)
            is thus $\log N^1/3 + 1$, where the + 1 is for the sign bit.
        eta: The number of electrons.
        num_bits_rot_aa: The number of bits of precision for the single qubit
            rotation for amplitude amplification. Called $b_r$ in the reference.

    Registers:

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) page 19, section B
    """

    num_pw_each_dim: int
    eta: int
    num_bits_rot_aa: int = 8

    @cached_property
    def signature(self) -> Signature:
        n_p = (self.num_pw_each_dim - 1).bit_length() + 1
        return Signature.build(plus=1, w=3, r=n_p - 2, s=n_p - 2)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # The cost arises from an equal superposition  over $\eta$ values of $i$
        # and $j$, followed by the uniform preparation over $w$, $r$ and $s$.
        n_p = (self.num_pw_each_dim - 1).bit_length() + 1
        uni_prep = UniformSuperPostionIJFirstQuantization(
            self.num_pw_each_dim, self.eta, self.num_bits_rot_aa
        )
        return {(1, uni_prep), (4 * (2 * n_p + 9), TGate())}


@frozen
class SelectTFirstQuantization(Bloq):
    r"""SELECT for the kinetic energy operator for the first quantized chemistry Hamiltonian.

    Args:
        num_pw_each_dim: The number of planewaves in each of the x, y and z
            directions. In total, for a cubic box, there are N = num_pw_each_dim**3
            planewaves. The number of bits required (in each dimension)
            is thus $\log N^1/3 + 1$, where the + 1 is for the sign bit.
        eta: The number of electrons.

    Registers:
        sys:

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) page 19, section B
    """

    num_pw_each_dim: int
    eta: int

    @cached_property
    def signature(self) -> Signature:
        n_p = (self.num_pw_each_dim - 1).bit_length() + 1
        return Signature(
            [
                Register("sys", bitsize=n_p, shape=(self.eta,)),
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
        n_p = (self.num_pw_each_dim - 1).bit_length() + 1
        return {(4 * (5 * (n_p - 1) + 2), TGate())}


@frozen
class PrepareUVFistQuantization(Bloq):
    r"""PREPARE for the kinetic energy operator for the first quantized chemistry Hamiltonian.

    Prepares a state of the form

    $$
        \frac{1}{2\mathcal{M}2^{n_p +2 }}
        \sum_{\mu=2}^{n_p+1}\sum_{\nu B_\mu}
        \sum_{m=0}^{\lceil \mathcal M(2^{\mu-2}/|\nu|)^2\rceil-1}
        \frac{1}{2^\mu}|\mu\rangle|\nu_x|rangle|\nu_y\rangle|\nu_z\rangle|m\rangle|0\rangle
    $$

    Args:
        num_pw_each_dim: The number of planewaves in each of the x, y and z
            directions. In total, for a cubic box, there are N = num_pw_each_dim**3
            planewaves. The number of bits required (in each dimension)
            is thus $\log N^1/3 + 1$, where the + 1 is for the sign bit.
        eta: The number of electrons.
        m_param: $\mathcal{M}$ in the reference.
        num_bits_rot_aa: The number of bits of precision for the single qubit
            rotation for amplitude amplification. Called $b_r$ in the reference.

    Registers:

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) page 19, section B
    """
    num_pw_each_dim: int
    eta: int
    m_param: int
    lambda_zeta: int
    er_lambda_zeta: int
    num_bits_rot_aa: int = 8
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        # this is for the nu register which lives on a grid of twice the size
        n_p = (self.num_pw_each_dim - 1).bit_length() + 1
        # the nu grid is twice as large, so one more bit is needed
        n_nu = (self.num_pw_each_dim - 1).bit_length() + 2
        n_mu = n_p.bit_length()
        #
        n_m = (self.m_param - 1).bit_length()
        return Signature(
            [
                Register("mu", bitsize=n_mu),
                Register("nu", bitsize=n_nu, shape=(3,)),
                Register("m", bitsize=n_m),
                Register("flag_nu", bitsize=1),
            ]
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # for $1/||\nu||$ preparation and inversion we have $3n_{p}^{2} + 15n_{p} − 7 +
        # 4n_{\mathcal{M}}(n_{p}+1)$
        n_p = (self.num_pw_each_dim - 1).bit_length() + 1
        n_m = (self.m_param - 1).bit_length()
        if self.adjoint:
            cost = 14 * n_p - 8  # eq 90 - 89 page 23
        else:
            cost = 3 * n_p**2 + n_p + 1 + 4 * n_m * (n_p + 1)  # eq 89, page 23
            cost += self.lambda_zeta * self.er_lambda_zeta  # Eq 92.


# ## SELECT U + V
# [1] Addition and subtraction of $\nu$ costs $24 n_{p}$
# [2] phasing by the structure factor $-e^{i k_{\nu} \cdot R_{\ell}}$ costs $6 n_{p} n_{R}$
@frozen
class SelectUVFirstQuantization(Bloq):
    r"""SELECT for the kinetic energy operator for the first quantized chemistry Hamiltonian.

    Args:
        num_pw_each_dim: The number of planewaves in each of the x, y and z
            directions. In total, for a cubic box, there are N = num_pw_each_dim**3
            planewaves. The number of bits required (in each dimension)
            is thus $\log N^1/3 + 1$, where the + 1 is for the sign bit.
        eta: The number of electrons.
        num_bits_nuc_pos: The number of bits to store each component of the
            nuclear positions. $n_R$ in the reference.

    Registers:
        sys:

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) page 19, section B
    """

    num_pw_each_dim: int
    eta: int
    num_bits_nuc_pos: int

    @cached_property
    def signature(self) -> Signature:
        n_p = (self.num_pw_each_dim - 1).bit_length() + 1
        n_nu = (self.num_pw_each_dim - 1).bit_length() + 2
        n_eta = (self.eta - 1).bit_length()
        return Signature(
            [
                Register("flag_U", bitsize=1),
                Register("flag_UorV", bitsize=1),
                Register("plus", bitsize=1),
                Register("ij", bitsize=n_eta, shape=(2,)),
                Register("nu", bitsize=n_nu, shape=(3,)),
                Register("sys", bitsize=n_p, shape=(self.eta,)),
                # + some ancilla for the controlled swaps of system registers.
            ]
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # Addition and subtraction of $\nu$ to p and q.
        n_p = (self.num_pw_each_dim - 1).bit_length() + 1
        cost_add = 24 * n_p  # Eq 93.
        # phasing by the structure factor $-e^{i k_{\nu} \cdot R_{\ell}}$ costs $6 n_{p} n_{R}$
        cost_phase = 6 * n_p * self.num_bits_nuc_pos  # Eq 97.
        return {(4 * (cost_add + cost_phase), TGate())}
