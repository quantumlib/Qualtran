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
r"""Bloqs for the double-factorized chemistry Hamiltonian in second quantization.

Recall that for the single factorized Hamiltonian we have
$$
    H = \sum_{pq}T'_{pq} a_p^\dagger a_q + \frac{1}{2} \sum_l \left(\sum_{pq}
W_{pq}^{(l)} a_p^\dagger a_q\right)^2.
$$
One arrives at the double factorized (DF) Hamiltonian by further factorizing the
$W_{pq}^{(l)}$ terms as
$$
    W^{(l)}_{pq} = \sum_{k} U^{(l)}_{pk} f_k^{(l)} U^{(l)*}_{qk},
$$
so that
$$
    H = \sum_{pq}T'_{pq} a_p^\dagger a_q + \frac{1}{2} \sum_l U^{(l)}\left(\sum_{k}^{\Xi^{(l)}}
        f_k^{(l)} n_k\right)^2 U^{(l)\dagger}
$$
where $\Xi^{(l)} $ is the rank of second factorization. In principle one can
truncate the second factorization to reduce the amount of information required
to specify the Hamiltonian. However this somewhat complicates the implementation
and it is more convenient to fix the rank of the second factorization for all
$l$ terms. Nevertheless the authors of Ref[1] chose to do it the hard way.
"""

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
class DoubleFactorizationOneBody(BlockEncoding):
    r"""Block encoding of double factorization one-body Hamiltonian.

    Implements inner "half" of Fig. 15 in the reference. This block encoding is
    applied twice (with a reflection around the inner state preparation
    registers) to implement a (roughly) the square of this one-body operator.

    Note succ_pq will be allocated as an ancilla during decomposition and it is not relected on.

    Args:
        num_aux: Dimension of auxiliary index for double factorized Hamiltonian. Call L in Ref[1].
        num_spin_orb: The number of spin orbitals. Typically called N.
        num_xi: Rank of second factorization. Full rank implies $Xi = num_spin_orb // 2$.
        num_bits_state_prep: The number of bits of precision for coherent alias
            sampling. Called $\aleph$ in Ref[1].
        num_bits_rot_aa_outer: Number of bits of precision for single qubit
            rotation for amplitude amplification. Called $b_r$ in the reference.
        num_bits_rot: Number of bits of precision for rotations for amplitude
            amplification in uniform state preparation. Called $\beth$ in Ref[1].
        adjoint: Whether this bloq is daggered or not. This affects the QROM cost.

    Registers:
        succ_l: control for success for outer state preparation.
        l_ne_zero: control for one-body part of Hamiltonian.
        l: Register for outer state preparation. This is of size $\ceil \log (L + 1)$.
        p: Register for inner state preparation. This is of size $\ceil \log (L \Xi + N / 2)$.
        spin: A single qubit register for spin.
        rot: qubit for amplitude amplification.
        state_prep: ancilla for state preparation.
        xi: data register for number storing $\Xi^{(l)}$.
        offset: Offset for p register.
        sys: The system register.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://arxiv.org/abs/2011.03494)
    """
    num_aux: int
    num_spin_orb: int
    num_xi: int
    num_bits_state_prep: int
    num_bits_rot_aa: int = 8
    num_bits_rot: int = 24
    adjoint: bool = False

    @property
    def control_registers(self) -> Iterable[Register]:
        return [Register("succ_l", bitsize=1), Register("l_ne_zero", bitsize=1)]

    @property
    def junk_registers(self) -> Iterable[Register]:
        return [
            Register("p", bitsize=(self.num_xi - 1).bit_length()),
            Register("spin", bitsize=1),
            Register("rot", bitsize=1),
            Register("state_prep", bitsize=self.num_bits_state_prep),
        ]

    @property
    def target_registers(self) -> Iterable[Register]:
        return [Register("sys", bitsize=self.num_spin_orb)]

    @property
    def selection_registers(self) -> Iterable[Register]:
        # really of size L + 1, hence just bit_length()
        nlxi = get_num_bits_lxi(self.num_aux, self.num_xi, self.num_spin_orb)
        nxi = (self.num_xi - 1).bit_length()  # C14
        return [
            Register("l", bitsize=self.num_aux.bit_length()),
            # slight abuse of selection registers here, these are really data
            # registers which are l dependent
            Register("xi", bitsize=nxi),
            Register("offset", bitsize=nlxi),
        ]

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        num_bits_lxi = get_num_bits_lxi(self.num_aux, self.num_xi, self.num_spin_orb)
        in_prep = InnerPrepare(
            num_aux=self.num_aux,
            num_spin_orb=self.num_spin_orb,
            num_xi=self.num_xi,
            num_bits_rot_aa=self.num_bits_rot_aa,
            num_bits_state_prep=self.num_bits_state_prep,
            adjoint=False,
        )
        in_prep_dag = InnerPrepare(
            num_aux=self.num_aux,
            num_spin_orb=self.num_spin_orb,
            num_xi=self.num_xi,
            num_bits_rot_aa=self.num_bits_rot_aa,
            num_bits_state_prep=self.num_bits_state_prep,
            adjoint=True,
        )
        n = self.num_spin_orb // 2
        rot = ProgRotGateArray(
            num_aux=self.num_aux,
            num_xi=self.num_xi,
            num_spin_orb=self.num_spin_orb,
            num_bits_rot=self.num_bits_rot,
            adjoint=False,
        )
        rot_dag = ProgRotGateArray(
            num_aux=self.num_aux,
            num_xi=self.num_xi,
            num_spin_orb=self.num_spin_orb,
            num_bits_rot=self.num_bits_rot,
            adjoint=True,
        )
        # 2*In-prep_l, addition, Rotations, 2*H, 2*SWAPS, subtraction
        return {
            (1, in_prep),  # in-prep_l listing 3 page 52/53
            (1, in_prep_dag),  # in_prep_l^dag
            (1, rot),  # rotate into system basis  listing 4 pg 54
            (4, TGate()),  # apply CCZ first then CCCZ, the cost is 1 + 2 Toffolis (step 4e, and 7)
            (1, rot_dag),  # Undo rotations
            (2, CSwapApprox(self.num_spin_orb // 2)),  # Swaps for spins
            (2, ArbitraryClifford(n=1)),  # 2 Hadamards for spin superposition
        }


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


@frozen
class DoubleFactorization(BlockEncoding):
    r"""Block encoding of double factorization Hamiltonian.

    Implements Fig. 15 in the reference.

    Args:
        num_spin_orb: The number of spin orbitals. Typically called N.
        num_aux: Dimension of auxiliary index for double factorized Hamiltonian. Called L in Ref[1].
        num_xi: Rank of second factorization. Full rank implies $Xi = num_spin_orb // 2$.
        num_bits_state_prep: The number of bits of precision for coherent alias
            sampling. Called $\aleph$ in Ref[1]. We assume this is the same for
            both outer and inner state preparations.
        num_bits_rot: Number of bits of precision for rotations
            amplification in uniform state preparation. Called $\beth$ in Ref[1].
        num_bits_rot_aa_outer: Number of bits of precision for single qubit
            rotation for amplitude amplification in outer state preparation.
            Called $b_r$ in the reference.
        num_bits_rot_aa_inner: Number of bits of precision for single qubit
            rotation for amplitude amplification in inner state preparation.
            Called $b_r$ in the reference.

    Registers:
        ctrl: An optional control register. This bloq should just be controlled.
        l: Register for outer state preparation. This is of size $\ceil \log (L + 1)$.
        p: Register for inner state preparation. This is of size $\ceil \log (L \Xi + N / 2)$.
        spin: A single qubit register for spin.
        sys: The system register.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            hypercontraction](https://arxiv.org/abs/2011.03494)
    """
    num_spin_orb: int
    num_aux: int
    num_xi: int
    num_bits_state_prep: int = 8
    num_bits_rot: int = 24
    num_bits_rot_aa_outer: int = 8
    num_bits_rot_aa_inner: int = 8

    @classmethod
    def build_from_coeffs(cls, one_body_ham, factorized_two_body_ham) -> 'DoubleFactorization':
        """Factory method to build double factorization block encoding given Hamiltonian inputs.

        Args:
            one_body_ham: One body hamiltonian ($T_{pq}$') matrix elements. (includes exchange terms).
            factorized_two_body_ham: One body hamiltonian ($W^{(l)}_{pq}$).

        Returns:
            Double factorized bloq with alt/keep values appropriately constructed.

        Refererences:
            [Even More Efficient Quantum Computations of Chemistry Through Tensor
                hypercontraction]
                (https://arxiv.org/abs/2011.03494). Eq. B7 pg 43.
        """
        assert len(one_body_ham.shape) == 2
        assert len(factorized_two_body_ham.shape) == 3
        raise NotImplementedError("Factory method not implemented yet.")

    @property
    def control_registers(self) -> Iterable[Register]:
        return [Register('ctrl', bitsize=1)]

    @property
    def target_registers(self) -> Iterable[Register]:
        return [Register("sys", bitsize=self.num_spin_orb)]

    @property
    def selection_registers(self) -> Iterable[Register]:
        return [Register("l", bitsize=self.num_aux.bit_length())]

    @property
    def junk_registers(self) -> Iterable[Register]:
        return [Register("p", bitsize=(self.num_xi - 1).bit_length()), Register("spin", bitsize=1)]

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrl: SoquetT, p: SoquetT, spin: SoquetT, sys: SoquetT, l: SoquetT
    ) -> Dict[str, 'SoquetT']:
        succ_l, l_ne_zero, theta, succ_p = bb.split(bb.allocate(4))
        num_bits_lxi = get_num_bits_lxi(self.num_aux, self.num_xi, self.num_spin_orb)
        num_bits_xi = (self.num_xi - 1).bit_length()
        xi = bb.allocate(num_bits_xi)
        offset = bb.allocate(num_bits_lxi)
        rot_data = bb.allocate(self.num_bits_rot_aa_inner)

        outer_prep = OuterPrepare(
            self.num_aux,
            num_bits_state_prep=self.num_bits_state_prep,
            num_bits_rot_aa=self.num_bits_rot_aa_outer,
        )
        l, succ_l = bb.add(outer_prep, l=l, succ_l=succ_l)
        in_l_data_l = OutputIndexedData(
            num_aux=self.num_aux,
            num_spin_orb=self.num_spin_orb,
            num_xi=self.num_xi,
            num_bits_rot_aa=self.num_bits_rot_aa_inner,
        )
        l, l_ne_zero, xi, rot_data, offset = bb.add(
            in_l_data_l, l=l, l_ne_zero=l_ne_zero, xi=xi, rot_data=rot_data, offset=offset
        )
        one_body = DoubleFactorizationOneBody(
            self.num_aux,
            self.num_spin_orb,
            self.num_xi,
            self.num_bits_state_prep,
            num_bits_rot_aa=self.num_bits_rot_aa_inner,
            num_bits_rot=self.num_bits_rot,
        )
        state_prep_anc = bb.allocate(self.num_bits_state_prep)
        one_body_sq = BlockEncodeChebyshevPolynomial(one_body, order=2)
        # ancilla for amplitude amplifcation
        rot = bb.allocate(1)
        succ_l, l_ne_zero, p, spin, rot, state_prep_anc, sys, l, xi, offset = bb.add(
            one_body_sq,
            succ_l=succ_l,
            l_ne_zero=l_ne_zero,
            p=p,
            spin=spin,
            rot=rot,
            state_prep=state_prep_anc,
            sys=sys,
            l=l,
            xi=xi,
            offset=offset,
        )
        bb.free(rot)
        bb.free(state_prep_anc)
        in_l_data_l = OutputIndexedData(
            num_aux=self.num_aux,
            num_spin_orb=self.num_spin_orb,
            num_xi=self.num_xi,
            num_bits_rot_aa=self.num_bits_rot_aa_inner,
            adjoint=True,
        )
        l, l_ne_zero, xi, rot_data, offset = bb.add(
            in_l_data_l, l=l, l_ne_zero=l_ne_zero, xi=xi, rot_data=rot_data, offset=offset
        )
        # prepare_l^dag
        outer_prep = OuterPrepare(
            self.num_aux,
            num_bits_state_prep=self.num_bits_state_prep,
            num_bits_rot_aa=self.num_bits_rot_aa_outer,
            adjoint=True,
        )
        l, succ_l = bb.add(outer_prep, l=l, succ_l=succ_l)
        bb.free(xi)
        bb.free(offset)
        bb.free(rot_data)
        bb.free(succ_l)
        bb.free(succ_p)
        bb.free(l_ne_zero)
        bb.free(theta)
        out_regs = {'l': l, 'sys': sys, 'p': p, 'spin': spin, 'ctrl': ctrl}
        return out_regs
