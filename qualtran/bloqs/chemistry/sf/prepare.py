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

from functools import cached_property
from typing import Set, TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, bloq_example, Signature
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.chemistry.black_boxes import PrepareUniformSuperposition, QROAM, QROAMTwoRegs

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class InnerPrepareSingleFactorization(Bloq):
    """Inner prepare for THC block encoding.

    Prepares the state in Eq. B11 of Ref [1].

    Currently we only provide costs as listed in Ref[1] without their corresponding decompositions.

    Args:
        num_aux: Dimension of auxiliary index for single factorized Hamiltonian. Call L in Ref[1].
        num_spin_orb: The number of spin orbitals. Typically called N.
        num_bits_state_prep: The number of bits of precision for coherent alias
            sampling. Called aleph (fancy N) in Ref[1].
        num_bits_rot_aa: Number of bits of precision for rotations for amplitude
            amplification in uniform state preparation. Called b_r in Ref[1].
        adjoint: Whether this bloq is daggered or not. This affects the QROM cost.

    Registers:
        l: register to store L values for auxiliary index.
        p: spatial orbital index. range(0, num_spin_orb // 2)
        q: spatial orbital index. range(0, num_spin_orb // 2)
        succ_pq: flag for success of this state preparation.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://arxiv.org/abs/2011.03494) Appendix B. Listing 2, page 44.
    """

    num_aux: int
    num_spin_orb: int
    num_bits_state_prep: int
    num_bits_rot_aa: int = 8
    adjoint: bool = False
    kp1: int = 1
    kp2: int = 1

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f"In-Prep{dag}"

    @cached_property
    def signature(self) -> Signature:
        n = (self.num_spin_orb // 2 - 1).bit_length()
        return Signature.build(l=self.num_aux.bit_length(), p=n, q=n, succ_pq=1)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        n = (self.num_spin_orb // 2 - 1).bit_length()
        # prep uniform upper triangular.
        cost_a = (Toffoli(), 6 * n + 2 * self.num_bits_rot_aa - 7)
        # contiguous index
        cost_b = (Toffoli(), n**2 + n - 1)
        # QROAM
        cost_c = (
            QROAMTwoRegs(
                self.num_aux + 1,
                self.num_spin_orb**2 // 8 + self.num_spin_orb // 2,
                self.kp1,
                self.kp2,
                2 * n + self.num_bits_state_prep + 2,
                adjoint=self.adjoint,
            ),
            1,
        )
        # inequality test
        cost_d = (Toffoli(), self.num_bits_state_prep)
        # controlled swap
        cost_e = (Toffoli(), 2 * n)
        return {cost_a, cost_b, cost_c, cost_d, cost_e}


@frozen
class OuterPrepareSingleFactorization(Bloq):
    r"""Outer state preparation.

    Implements Eq. B8 from the Reference.

    Args:
        num_aux: Dimension of auxiliary index for single factorized Hamiltonian.
            Called $L$ in the reference.
        num_bits_state_prep: The number of bits of precision for coherent alias
            sampling. Called $\aleph$ in the reference.
        num_bits_rot_aa: Number of bits of precision for rotations for amplitude
            amplification in uniform state preparation. Called $b_r$ in the reference.
        adjoint: Whether to dagger the bloq or not.

    Registers:
        l: register to store L values for auxiliary index.
        succ_l: flag for success of this state preparation.
        l_ne_zero: flag for preparation of one-body term too.
        rot_aa: The qubit that is rotated for amplitude amplification.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://arxiv.org/abs/2011.03494)
            Appendix B, page 43. Step 1.
    """

    num_aux: int
    num_bits_state_prep: int
    num_bits_rot_aa: int = 8
    adjoint: bool = False

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f"OuterPrep{dag}"

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(l=self.num_aux.bit_length(), succ_l=1, l_ne_zero=1, rot_aa=1)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        cost_a = (PrepareUniformSuperposition(self.num_aux + 1, self.num_bits_rot_aa), 1)
        num_bits_l = (self.num_aux + 1).bit_length()
        output_size = num_bits_l + self.num_bits_state_prep + 2
        cost_b = (QROAM(self.num_aux + 1, output_size, adjoint=self.adjoint), 1)
        cost_c = (Toffoli(), self.num_bits_state_prep)
        cost_d = (Toffoli(), num_bits_l + 1)
        return {cost_a, cost_b, cost_c, cost_d}


@bloq_example
def _prep_inner() -> InnerPrepareSingleFactorization:
    num_aux = 50
    num_bits_state_prep = 12
    num_bits_rot = 7
    num_spin_orb = 10
    num_aux = 50
    prep_inner = InnerPrepareSingleFactorization(
        num_aux=num_aux,
        num_spin_orb=num_spin_orb,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa=num_bits_rot,
        kp1=2**2,
        kp2=2**3,
    )
    return prep_inner


@bloq_example
def _prep_outer() -> OuterPrepareSingleFactorization:
    num_aux = 50
    num_bits_state_prep = 12
    num_bits_rot_aa = 8
    prep_outer = OuterPrepareSingleFactorization(
        num_aux, num_bits_state_prep=num_bits_state_prep, num_bits_rot_aa=num_bits_rot_aa
    )
    return prep_outer
