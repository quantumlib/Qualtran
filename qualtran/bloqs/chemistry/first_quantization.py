# # Cubic Unit Cell First-Quantized Bloqs

# ## System Register
# The system Register is represented as signed integers for each Euler direction.
# for $\eta$ electrons we use $n_{p}$ bits for each of the $xyz$-directions which means the system register is $3 \eta n_{p}$ bits wide.  We will need additional ancilla registers for PREPARE and SELECT.
# We will be moving $i$ and $j$ electron registers into a working register which for $T$ $U$ and $V$ costs
# $12 \eta n_{p} + 4\eta - 8$. This comes from the core $12 \eta n_{p}$ plus 4 factors of $\eta - 2$ unary iteration.

# ## PREPARE Kinetic

# The kinetic energy PREPARE prepare's a state

# $|+\rangle\sum_{j=1}^{\eta}|j\rangle\sum_{w=0}^{2}|w\rangle\sum_{r=0}^{n_{p}-2}2^{r/2}|r\rangle \sum_{s=0}^{n_{p}-2}2^{s/2}|s\rangle$

# [1] The cost of preparing equal superposition over $\eta$ values of $i$ and $j$ is  $14 n_{\eta} + 8 b_{r} - 36$

# [2] State prep costs for $w$ $r$ and $s$ are $2 ( 2 n_{p} + 9)$

# ## SELECT Kinetic

# [1] Cost is $5(n_{p} - 1) + 2$ which comes from copying each $w$ component of $p$ into an ancilla register ($3(n_{p}-1)$), copying the $r$ and $s$ bit of into an ancilla ($2(n_{p}-1)$), controlling on both those bit perform phase flip on an ancilla $|+\rangle$ state. This requires $1$ Toffoli, Then erase which costs only Cliffords. There is an additional control bit controlling the application of $T$ thus we come to our total.


# ## PREPARE U + V

# [1] for $1/||\nu||$ preparation and inversion we have $3n_{p}^{2} + 15n_{p} − 7 + 4n_{\mathcal{M}}(n_{p}+1)$

# [2] QROM for $R_{\ell}$ and for state prep with amplitudes $\sqrt{\lambda_{\ell}}$ we have cost $\lambda_{\zeta} + \mathrm{Er}(\lambda_{\zeta})$

# ## SELECT U + V

# [1] Addition and subtraction of $\nu$ costs $24 n_{p}$

# [2] phasing by the structure factor $-e^{i k_{\nu} \cdot R_{\ell}}$ costs $6 n_{p} n_{R}$


# ## REFLECTION Cost

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
class UniformSuperPostionKineticFirstQuantization(Bloq):
    r"""Uniform superposition over $\eta$ values of $i$ and $j$.

    Args:

    Registers:

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) page 19, section B
    """
    num_plane_wave: int
    eta: int
    num_bits_rot_aa: int

    @cached_property
    def signature(self) -> Signature:
        return Signature([])

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        n_eta = (self.eta - 1).bit_length()
        return {(14 * n_eta + 8 * self.num_bits_rot_aa - 36, TGate())}


@frozen
class PrepareKineticFirstQuantization(Bloq):
    r"""PREPARE for the kinetic energy operator for the first quantized chemistry Hamiltonian.

    This prepares the state

    $$
        |+\rangle\sum_{j=1}^{\eta}|j\rangle\sum_{w=0}^{2}|w\rangle\sum_{r=0}^{n_{p}-2}2^{r/2}|r\rangle
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
        uni_prep = UniformSuperPostionKineticFirstQuantization(
            self.num_pw_each_dim, self.eta, self.num_bits_rot_aa
        )
        return {(1, uni_prep), (2 * (2 * n_p + 9), TGate())}
