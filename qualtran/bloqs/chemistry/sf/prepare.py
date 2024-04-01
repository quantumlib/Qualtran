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
from qualtran.bloqs.arithmetic.comparison import LessThanEqual
from qualtran.bloqs.arithmetic.conversions import ToContiguousIndex
from qualtran.bloqs.basic_gates import CSwap, Toffoli
from qualtran.bloqs.chemistry.black_boxes import QROAM, QROAMTwoRegs
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
    PrepareUniformSuperposition,
)

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
        kp1: qroam blocking factor for data indexed by the first register. ($l$ in this case)
        kp2: qroam blocking factor for data indexed by the second register. ($p,
            q$ in this case (these are made into a contiguous register.))

    Registers:
        l: register to store L values for auxiliary index.
        p: spatial orbital index. range(0, num_spin_orb // 2)
        q: spatial orbital index. range(0, num_spin_orb // 2)
        succ_pq: flag for success of this state preparation.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://arxiv.org/abs/2011.03494) Appendix B. Listing 2, page 44.
        [Qubitization of Arbitrary Basis Quantum Chemistry Leveraging Sparsity
        and Low Rank Factorization](https://quantum-journal.org/papers/q-2019-12-02-208/) Sec. 3.2
    """

    num_aux: int
    num_spin_orb: int
    num_bits_state_prep: int
    num_bits_rot_aa: int = 8
    kp1: int = 1
    kp2: int = 1

    def pretty_name(self) -> str:
        return "In-Prep"

    @cached_property
    def signature(self) -> Signature:
        n = (self.num_spin_orb // 2 - 1).bit_length()
        return Signature.build(l=self.num_aux.bit_length(), p=n, q=n, succ_pq=1)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        n = (self.num_spin_orb // 2 - 1).bit_length()
        # 2. a prep uniform upper triangular.
        cost_up_tri = (Toffoli(), 6 * n + 2 * self.num_bits_rot_aa - 7)
        # 2.b contiguous index from p and q
        cost_contg_indx = (ToContiguousIndex(n, 2 * n), 1)
        # Note the data size is for storing the upper triangular matrices of
        # size N/2, so N/2 (N/2 + 1)/2. There is an error in the reference
        # equation B13 of Reg[1] (it is correct in Ref[2], and openfermion).
        # 2.c QROAM
        cost_qroam = (
            QROAMTwoRegs(
                self.num_aux + 1,
                self.num_spin_orb**2 // 8 + self.num_spin_orb // 4,
                self.kp1,
                self.kp2,
                2 * n + self.num_bits_state_prep + 2,
            ),
            1,
        )
        # inequality test
        cost_ineq = (LessThanEqual(self.num_bits_state_prep, self.num_bits_state_prep), 1)
        cost_swap = (CSwap(2 * n), 1)
        return {cost_up_tri, cost_contg_indx, cost_qroam, cost_ineq, cost_swap}


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

    def pretty_name(self) -> str:
        return "OuterPrep"

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(l=self.num_aux.bit_length(), succ_l=1, l_ne_zero=1, rot_aa=1)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # 1.a
        cost_uni = (PrepareUniformSuperposition(self.num_aux + 1), 1)
        num_bits_l = (self.num_aux + 1).bit_length()
        output_size = num_bits_l + self.num_bits_state_prep + 2
        # 1.b
        cost_qroam = (QROAM(self.num_aux + 1, output_size), 1)
        # 1.c inequality test for alias sampling
        cost_ineq = (LessThanEqual(self.num_bits_state_prep, self.num_bits_state_prep), 1)
        # 1.d swap alt/keep values
        cost_swap = (CSwap(num_bits_l + 1), 1)
        return {cost_uni, cost_qroam, cost_ineq, cost_swap}


@bloq_example
def _prep_inner() -> InnerPrepareSingleFactorization:
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
