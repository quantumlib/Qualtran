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
from typing import TYPE_CHECKING

from attrs import evolve, frozen

from qualtran import Bloq, bloq_example, Signature
from qualtran.bloqs.basic_gates import Toffoli

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@frozen
class PreparePowerTwoStateWithProj(Bloq):
    r"""Prepares the uniform superposition over $|r\rangle$ given by Eq. 69 in the reference.

    This prepares the state

    $$
        2^{(-n_p -1)/2} \sum_r=0^{n_p-2} 2^{r/2} |r\rangle
    $$

    in one-hot unary.

    To account for the conditional preparation of weight for the projectile we
    add additional controls to the Hadamards for $p > n_p$.

    Args:
        bitsize_n: the number of bits for the projectiles momentum $n_n$.
        bitsize_p: the number of bits for the electron's momentum $n_p$

    Registers:
        r: The register we want to prepare the state over.

    References:
        [Quantum computation of stopping power for inertial fusion target design](https://arxiv.org/abs/2308.12352)
        page 11, C3 also page 31 App A. Sec 2 b.

        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](https://arxiv.org/abs/2105.12767)
        page 19, section B
    """
    bitsize_n: int
    bitsize_p: int
    is_adjoint: bool = False

    def __attrs_post_init__(self):
        if self.bitsize_n < self.bitsize_p:
            raise ValueError(f"bitsize_n < bitsize_p : {self.bitsize_n} < {self.bitsize_p}.")

    def adjoint(self) -> 'Bloq':
        return evolve(self, is_adjoint=not self.is_adjoint)

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(r=self.bitsize_n)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        if self.is_adjoint:
            return {Toffoli(): (self.bitsize_n - 2)}
        else:
            # The doubly controlled hadamard can be converted to and And and a
            # controlled Hadamard, with the And gate being inverted at zero
            # Toffoli cost.
            return {Toffoli(): (self.bitsize_n - 2) + self.bitsize_n - self.bitsize_p}


@frozen
class PrepareTFirstQuantizationWithProj(Bloq):
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
        is_adjoint: whether to dagger the bloq or not.

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
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(w=2, w_mean=2, r=self.num_bits_n, s=self.num_bits_n)

    def adjoint(self) -> 'Bloq':
        return evolve(self, is_adjoint=not self.is_adjoint)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # there is a cost for the uniform state preparation for the $w$
        # register. Adding a bloq is sort of overkill, should just tag the
        # correct cost on UniformSuperPosition bloq
        # 13 is from assuming 8 bits for the rotation, and n = 2.
        uni_prep_w = 13
        # Factor of two for r and s registers.
        ctrl_mom_gate = PreparePowerTwoStateWithProj(
            bitsize_n=self.num_bits_n, bitsize_p=self.num_bits_p, is_adjoint=self.is_adjoint
        )
        # Inequality test can be inverted at zero cost
        if self.is_adjoint:
            # pg 31 (Appendix A. Sec 2 c)
            k_k_proj = 0
        else:
            # Cost for preparing a state for selecting the components of k_p^w k_proj^w
            # Prepare a uniform superposition over 8 states and do 2 inequality
            # tests to select between x, y and z.
            # built on w_proj above
            k_k_proj = 16
        # pg 31 (Appendix A. Sec 2 c)
        ctrl_swap = 2
        return {Toffoli(): uni_prep_w + k_k_proj + ctrl_swap, ctrl_mom_gate: 2}


@bloq_example
def _prep_power_two_proj() -> PreparePowerTwoStateWithProj:
    num_bits_p = 6
    num_bits_n = 8
    prep_power_two_proj = PreparePowerTwoStateWithProj(num_bits_n, num_bits_p)
    return prep_power_two_proj


@bloq_example
def _prep_t_proj() -> PrepareTFirstQuantizationWithProj:
    num_bits_p = 6
    num_bits_n = 8
    eta = 32
    prep_t_proj = PrepareTFirstQuantizationWithProj(num_bits_p, num_bits_n, eta)
    return prep_t_proj
