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
r"""Bloqs for PREPARE T for the first quantized chemistry Hamiltonian with a quantum projectile."""
from functools import cached_property
from typing import Set, TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, Signature
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_t import PreparePowerTwoState

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class PrepareTProjFirstQuantization(Bloq):
    r"""PREPARE for the kinetic energy operator for the first quantized chemistry Hamiltonian.

    This prepares the state

    $$
        |+\rangle\sum_{j=1}^{\eta}|j\rangle\sum_{w=0}^{2}|w\rangle
        \sum_{r=0}^{n_{p}-2}2^{r/2}|r\rangle
        \sum_{s=0}^{n_{p}-2}2^{s/2}|s\rangle
    $$

    The case assumes a quantum projectile whose state is descirbed by num_bits_n bits.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        num_bits_n: The number of bits to represent each dimension of the
            momentum register for the projectile. This is called $n_n$ in the reference.
        eta: The number of electrons.
        num_bits_rot_aa: The number of bits of precision for the single qubit
            rotation for amplitude amplification. Called $b_r$ in the reference.
        adjoint: whether to dagger the bloq or not.

    Registers:
        w: a register to index one of three components of the momenta.
        w_mean: a register to index one of three components of the momenta for the
            projectile (used for the kmean part of the Hamiltonian)
        r: a register encoding bits for each component of the momenta.
        s: a register encoding bits for each component of the momenta.

    References:
        [Quantum computation of stopping power for inertial fusion target design](
            https://arxiv.org/abs/2308.12352) page 11, C3 also page 31 App A. Sec 2 b.
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) page 19, section B
    """

    num_bits_p: int
    num_bits_n: int
    eta: int
    num_bits_rot_aa: int = 8
    adjoint: int = False

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(w=2, w_mean=2, r=self.num_bits_p, s=self.num_bits_p)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # there is a cost for the uniform state preparation for the $w$
        # register. Adding a bloq is sort of overkill, should just tag the
        # correct cost on UniformSuperPosition bloq
        # 13 is from assuming 8 bits for the rotation, and n = 2.
        uni_prep_w = (Toffoli(), 13)
        # Factor of two for r and s registers.
        ctrl_mom = (PreparePowerTwoState(bitsize=self.num_bits_n), 2)
        # Inequality test can be inverted at zero cost
        if self.adjoint:
            # pg 31 (Appendix A. Sec 2 c)
            k_k_proj = (Toffoli(), 0)
            ctrl_had = (Toffoli(), 0)
        else:
            # Cost for preparing a state for selecting the components of k_p^w k_proj^w
            # Prepare a uniform superposition over 8 states and do 2 inequality
            # tests to select between x, y and z.
            # built on w_proj above
            k_k_proj = (Toffoli(), 16)
            # controlled Hadmards for preparing T_proj or T_elec, factor of 2 for r and s registers.
            ctrl_had = (Toffoli(), 2 * (self.num_bits_n - self.num_bits_p))
        # pg 31 (Appendix A. Sec 2 c)
        ctrl_swap = (Toffoli(), 2)
        return {uni_prep_w, ctrl_had, ctrl_mom, k_k_proj, ctrl_swap}
