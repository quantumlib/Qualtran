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
from typing import Set, TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, bloq_example, QUInt, Signature
from qualtran.bloqs.arithmetic import Add
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.chemistry.black_boxes import QROAM
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
    PrepareUniformSuperposition,
)

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class InnerPrepareDoubleFactorization(Bloq):
    r"""Inner prepare for DF block encoding.

    Prepare state over $p$ register controlled on outer $l$ register.

    Args:
        num_aux: Dimension of auxiliary index for double factorized Hamiltonian. Call L in Ref[1].
        num_spin_orb: The number of spin orbitals. Typically called N.
        num_eig: Total number of eigenvalues.
        num_bits_rot_aa: Number of bits of precision for single qubit
            rotation for amplitude amplification. Called $b_r$ in the reference.
        num_bits_state_prep: The number of bits of precision for coherent alias
            sampling. Called $\aleph$ in Ref[1].

    Registers:
        xi: data register for number storing $\Xi^{(l)}$.
        rot: qubit for amplitude amplification.
        succ_p: control to flag success for inner state preparation.
        p: Register for inner state preparation.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://arxiv.org/abs/2011.03494).  Step 3. Page 52.
    """

    num_aux: int
    num_spin_orb: int
    num_eig: int
    num_bits_rot_aa: int
    num_bits_state_prep: int

    def pretty_name(self) -> str:
        return "In-Prep"

    @cached_property
    def signature(self) -> Signature:
        nlxi = (self.num_eig + self.num_spin_orb // 2 - 1).bit_length()
        nxi = (self.num_spin_orb // 2 - 1).bit_length()
        return Signature.build(xi=nxi, offset=nlxi, rot=self.num_bits_rot_aa, succ_p=1, p=nxi)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Step 3.
        num_bits_xi = (self.num_spin_orb // 2 - 1).bit_length()
        # uniform superposition state, requires controlled hadamards
        # https://github.com/quantumlib/Qualtran/issues/237
        cost_a = (Toffoli(), 7 * num_bits_xi + 2 * self.num_bits_rot_aa - 6)
        # add offset to get correct bit of QROM from [l + offset^l, l+offset^l+Xi^l]
        num_bits_lxi = (self.num_eig + self.num_spin_orb // 2 - 1).bit_length()
        cost_b = (Add(QUInt(num_bits_lxi)), 1)
        # QROAM for alt/keep values
        bp = num_bits_xi + self.num_bits_state_prep + 2  # C31
        cost_c = (QROAM(self.num_eig + self.num_spin_orb // 2, bp), 1)
        # inequality tests + CSWAP
        cost_d = (Toffoli(), self.num_bits_state_prep + num_bits_xi)
        return {cost_a, cost_b, cost_c, cost_d}


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

    def pretty_name(self) -> str:
        return "OuterPrep"

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(l=self.num_aux.bit_length(), succ_l=1)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Listing 1 steps a-d
        num_bits_l = self.num_aux.bit_length()
        cost_a = (PrepareUniformSuperposition(self.num_aux + 1), 1)
        # QROAM for alt/keep
        output_size = num_bits_l + self.num_bits_state_prep
        cost_b = (QROAM(self.num_aux + 1, output_size), 1)
        # inequality test
        cost_c = (Toffoli(), self.num_bits_state_prep)
        # controlled swaps
        cost_d = (Toffoli(), num_bits_l)
        return {cost_a, cost_b, cost_c, cost_d}


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
        num_eig: Total number of eigenvalues.
        num_bits_rot_aa: Number of bits of precision for single qubit
            rotation for amplitude amplification.  Called $b_r$ in the reference.

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
    num_eig: int
    num_bits_rot_aa: int = 8

    def pretty_name(self) -> str:
        return "In_l-data_l"

    @cached_property
    def signature(self) -> Signature:
        nlxi = (self.num_eig + self.num_spin_orb // 2 - 1).bit_length()
        nxi = (self.num_spin_orb // 2 - 1).bit_length()
        return Signature.build(
            l=self.num_aux.bit_length(),
            l_ne_zero=1,
            xi=nxi,
            rot_data=self.num_bits_rot_aa,
            offset=nlxi,
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # listing 2 C29/30 page 52
        num_bits_lxi = (self.num_eig + self.num_spin_orb // 2 - 1).bit_length()
        num_bits_xi = (self.num_spin_orb // 2 - 1).bit_length()
        bo = num_bits_xi + num_bits_lxi + self.num_bits_rot_aa + 1
        return {(QROAM(self.num_aux + 1, bo), 1)}


@bloq_example
def _prep_inner() -> InnerPrepareDoubleFactorization:
    num_bits_state_prep = 12
    num_bits_rot = 7
    num_spin_orb = 10
    num_aux = 50
    num_eig = num_aux * (num_spin_orb // 2)
    prep_inner = InnerPrepareDoubleFactorization(
        num_aux=num_aux,
        num_spin_orb=num_spin_orb,
        num_eig=num_eig,
        num_bits_rot_aa=num_bits_rot,
        num_bits_state_prep=num_bits_state_prep,
    )
    return prep_inner


@bloq_example
def _prep_outer() -> OuterPrepareDoubleFactorization:
    num_aux = 50
    num_bits_state_prep = 12
    num_bits_rot = 18
    prep_outer = OuterPrepareDoubleFactorization(
        num_aux, num_bits_state_prep=num_bits_state_prep, num_bits_rot_aa=num_bits_rot
    )
    return prep_outer


@bloq_example
def _indexed_data() -> OutputIndexedData:
    num_aux = 50
    num_bits_rot = 18
    num_spin_orb = 10
    num_aux = 50
    num_eig = num_aux * num_spin_orb // 2
    indexed_data = OutputIndexedData(
        num_aux=num_aux, num_spin_orb=num_spin_orb, num_eig=num_eig, num_bits_rot_aa=num_bits_rot
    )
    return indexed_data
