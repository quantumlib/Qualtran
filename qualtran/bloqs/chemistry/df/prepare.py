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
r"""Bloqs for the state preparation of the DF Hamiltonian."""

from functools import cached_property
from typing import Dict, Iterable, Optional, Set, Tuple, TYPE_CHECKING

from attrs import frozen
from sympy import factorint

from qualtran import Bloq, BloqBuilder, Register, Signature, SoquetT
from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.chemistry.black_boxes import QROAM
from qualtran.bloqs.chemistry.df.common_bitsize import get_num_bits_lxi
from qualtran.bloqs.swap_network import CSwapApprox
from qualtran.bloqs.util_bloqs import ArbitraryClifford

if TYPE_CHECKING:
    from qualtran.resource_counting import SympySymbolAllocator


@frozen
class InnerPrepareDoubleFactorization(Bloq):
    r"""Inner prepare for DF block encoding.

    Prepare state over $p$ register controlled on outer $l$ register.

    Currently we only provide costs as listed in Ref[1] without their corresponding decompositions.

    Args:
        num_aux: Dimension of auxiliary index for double factorized Hamiltonian. Call L in Ref[1].
        num_spin_orb: The number of spin orbitals. Typically called N.
        num_xi: Rank of second factorization. Full rank implies $Xi = num_spin_orb // 2$.
        num_bits_rot_aa: Number of bits of precision for single qubit
            rotation for amplitude amplification. Called $b_r$ in the reference.
        num_bits_state_prep: The number of bits of precision for coherent alias
            sampling. Called $\aleph$ in Ref[1].
        adjoint: Whether this bloq is daggered or not. This affects the QROM cost.

    Registers:
        xi: data register for number storing $\Xi^{(l)}$.
        rot: qubit for amplitude amplification.
        succ_p: control to flag success for inner state preparation.
        p: Register for inner state preparation. This is of size $\ceil \log (L \Xi + N / 2)$.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://arxiv.org/abs/2011.03494).
            Step 3. Page 52.
    """

    num_aux: int
    num_spin_orb: int
    num_xi: int
    num_bits_rot_aa: int
    num_bits_state_prep: int
    adjoint: bool = False

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f"In-Prep{dag}"

    @cached_property
    def signature(self) -> Signature:
        lxi = get_num_bits_lxi(self.num_aux, self.num_xi, self.num_spin_orb)
        return Signature.build(
            xi=(self.num_xi - 1).bit_length(), offset=lxi, rot=self.num_bits_rot_aa, succ_p=1, p=lxi
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # Step 3.
        num_bits_xi = (self.num_xi - 1).bit_length()
        # uniform superposition over xi^l
        cost_a = 7 * num_bits_xi + 2 * self.num_bits_rot_aa - 6
        num_bits_lxi = get_num_bits_lxi(self.num_aux, self.num_xi, self.num_spin_orb)
        # add offset to get correct bit of QROM from [l + offset^l, l+offset^l+Xi^l]
        cost_b = num_bits_lxi - 1
        # QROAM for alt/keep values
        bp = num_bits_xi + self.num_bits_state_prep + 2  # C31
        cost_c = (1, QROAM(self.num_aux * self.num_xi + self.num_spin_orb // 2, bp, self.adjoint))
        # inequality tests + CSWAP
        cost_d = self.num_bits_state_prep + num_bits_xi
        return {(4 * (cost_a + cost_b + cost_d), TGate()), cost_c}


@frozen
class OuterPrepareDoubleFactorization(Bloq):
    r"""Outer state preparation.

    Implements "the appropriate superposition state" on the $l$ register. i.e.

    $$
    |0\rangle = \sum_l a_l |l\rangle,
    $$

    where $a_l = \sqrt{\frac{c^{l}}{\sum_{l} c^{(l)}}$, and $c^{l} = \sum_{k} f_k^{(l)}^2$.

    Args:
        num_aux: Dimension of auxiliary index for double factorized Hamiltonian. Call L in Ref[1].
        num_bits_state_prep: The number of bits of precision for coherent alias
            sampling. Called $\aleph$ in Ref[1].
        num_bits_rot_aa: Number of bits of precision for single qubit
            rotation for amplitude amplification in inner state preparation.
            Called $b_r$ in the reference.
        adjoint: Whether to dagger the bloq or not.

    Registers:
        l: register to store L values for auxiliary index.
        succ_l: flag for success of this state preparation.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://arxiv.org/abs/2011.03494)
            Appendix C, page 51 and 52
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
        return Signature.build(l=self.num_aux.bit_length(), succ_l=1)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # https://arxiv.org/pdf/2011.03494.pdf page 51
        # accounts for tiny factor if L is not divisible by factor of 2^eta.
        factors = factorint(self.num_aux)
        eta = factors[min(list(sorted(factors.keys())))]
        if (self.num_aux + 1) % 2 == 1:
            eta = 0
        num_bits_l = self.num_aux.bit_length()
        cost_a = 3 * num_bits_l - 3 * eta + 2 * self.num_bits_rot_aa - 9
        # QROM for alt/keep
        output_size = num_bits_l + self.num_bits_state_prep
        cost_b = (1, QROAM(self.num_aux + 1, output_size, adjoint=self.adjoint))
        # inequality test
        cost_c = self.num_bits_state_prep
        # controlled swaps
        cost_d = num_bits_l
        return {(4 * (cost_a + cost_c + cost_d), TGate()), cost_b}


@frozen
class OutputIndexedData(Bloq):
    r"""Output data indexed by outer index $l$ needed for inner preparation.

    We need to output $\Xi^{(l)}$ with $n_{\Xi}$ bits and the amplitude
    amplification angles with $b_r$ bits for the inner state preparation. We
    also output an offset with $n_{L\Xi}$ bits. The offset will be used to help
    index a contiguous register formed from $l$ and $k$.

    Args:
        num_aux: Dimension of auxiliary index for double factorized Hamiltonian. Called L in Ref[1].
        num_spin_orb: The number of spin orbitals. Typically called N.
        num_xi: Rank of second factorization. Full rank implies $Xi = num_spin_orb // 2$.
        num_bits_rot_aa: Number of bits of precision for single qubit
            rotation for amplitude amplification.  Called $b_r$ in the reference.
        ajoint: Whether to dagger the bloq or note. Affects bloq counts.

    Registers:
        l: register to store L values for auxiliary index.
        l_ne_zero: flag for one-body term.
        xi: rank for each l
        rot: rotation for amplitude amplification.
        offset: offset for each DF factor.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://arxiv.org/abs/2011.03494)
            Appendix C, page 52. Step 2.
    """

    num_aux: int
    num_spin_orb: int
    num_xi: int
    num_bits_rot_aa: int = 8
    adjoint: bool = False

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f"In_l-data_l{dag}"

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(
            l=self.num_aux.bit_length(),
            l_ne_zero=1,
            xi=(self.num_xi - 1).bit_length(),
            rot_data=self.num_bits_rot_aa,
            offset=get_num_bits_lxi(self.num_aux, self.num_xi, self.num_spin_orb),
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # listing 2 C29/30 page 52
        num_bits_xi = (self.num_xi - 1).bit_length()
        num_bits_offset = get_num_bits_lxi(self.num_aux, self.num_xi, self.num_spin_orb)
        bo = num_bits_xi + num_bits_offset + self.num_bits_rot_aa + 1
        return {(1, QROAM(self.num_aux + 1, bo, adjoint=self.adjoint))}
