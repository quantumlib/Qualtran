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

import numpy as np
from attrs import frozen
from sympy import factorint

from qualtran import Bloq, BloqBuilder, Register, Signature, SoquetT
from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.block_encoding import BlockEncodeChebyshevPolynomial, BlockEncoding
from qualtran.bloqs.swap_network import CSwapApprox
from qualtran.bloqs.util_bloqs import ArbitraryClifford

if TYPE_CHECKING:
    from qualtran.resource_counting import SympySymbolAllocator


def get_num_bits_lxi(num_aux, num_xi, num_spin_orb) -> int:
    num_bits_lxi = (num_aux * num_xi + num_spin_orb // 2 - 1).bit_length()
    return num_bits_lxi


def get_qroam_cost(data_size: int, bitsize: int, adjoint=False) -> Tuple[int, int]:
    """This gives the optimal k and minimum cost for a QROM over L values of
        size M.

    Adapted from openfermion.

    Args:
        data_size: Amount of data we want to load.
        bitsize: the amount of bits of output we need.

    Returns:
       val_opt: minimal (optimal) cost of QROM
    """
    if adjoint:
        k = 0.5 * np.log2(data_size)
        value = lambda k: data_size / 2**k + 2**k
    else:
        k = 0.5 * np.log2(data_size / bitsize)
        assert k >= 0
        value = lambda k: data_size / 2**k + bitsize * (2**k - 1)
    k_int = np.array([np.floor(k), np.ceil(k)])
    k_opt = k_int[np.argmin(value(k_int))]
    val_opt = np.ceil(value(k_opt))
    return int(val_opt)


@frozen
class QROAM(Bloq):
    """Placeholder bloq for QROAM.

    Helpful for comparing costs to those quoted in literature and those from cirq_ft.
    """

    data_size: int
    target_bitsize: int
    adjoint: bool = False

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f"QROAM{dag}"

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(
            sel=(int(np.ceil((self.data_size / self.k)) - 1).bit_length()), trg=self.target_bitsize
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        cost = get_qroam_cost(self.data_size, self.target_bitsize, adjoint=self.adjoint)
        # print("QROM ", self.adjoint, cost)
        return {(4 * cost, TGate())}


@frozen
class QROAMTwoRegs(Bloq):
    """Placeholder bloq for QROAM on two registers.

    Helpful for comparing costs to those quoted in literature and those from cirq_ft.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://arxiv.org/abs/2011.03494)
            Appendix B, page 44
    """

    data_size_a: int
    data_size_b: int
    target_bitsize_a: int
    target_bitsize_b: int
    ka: int
    kb: int
    adjoint = False

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f"QROAM{dag}"

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(
            sel_a=(int(np.ceil((self.data_size_a / self.ka)) - 1).bit_length()),
            sel_b=(int(np.ceil((self.data_size_b / self.kb)) - 1).bit_length()),
            trg=self.target_bitsize_a + self.target_bitsize_b,  # ?
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        if self.adjoint:
            fac_1 = int(np.ceil(self.data_size_a / self.ka))
            fac_2 = int(np.ceil(self.data_size_b / self.kb))
            fac_3 = (self.target_bitsize_a + self.target_bitsize_b) * (self.ka * self.kb - 1)
        else:
            fac_1 = int(np.ceil(self.data_size_a / self.ka))
            fac_2 = int(np.ceil(self.data_size_b / self.kb))
            fac_3 = self.ka * self.kb
        return {(4 * (fac_1 + fac_2 + fac_3), TGate())}


@frozen
class ProgRotGateArray(Bloq):
    """Rotate to to/from MO basis so-as-to apply number operators in DF basis.

    This is really a subclass of cirq_ft's ProgrammableRotationGateArray

    Args:

    Registers:

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.030305).
            Step 4. Page 53.
    """

    num_aux: int
    num_xi: int
    num_spin_orb: int
    num_bits_rot: int
    num_bits_offset: int
    adjoint: bool = False

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f"Rotations{dag}"

    @cached_property
    def signature(self) -> Signature:
        num_bits_lxi = get_num_bits_lxi(self.num_aux, self.num_xi, self.num_spin_orb)
        return Signature.build(
            offset=self.num_bits_offset,
            p=num_bits_lxi,
            rotations=(self.num_spin_orb // 2)
            * self.num_bits_rot,  # This assumes kr = 1 so we're dumping the N rotations in memory.
            spin_sel=1,
            sys_a=self.num_spin_orb // 2,
            sys_b=self.num_spin_orb // 2,
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        num_bits_lxi = get_num_bits_lxi(self.num_aux, self.num_xi, self.num_spin_orb)
        cost_a = num_bits_lxi - 1  # contiguous register
        data_size = self.num_aux * self.num_xi + self.num_spin_orb // 2
        cost_c = self.num_spin_orb * (self.num_bits_rot - 2)  # apply rotations
        return {
            (4 * (cost_a + cost_c), TGate()),
            (1, QROAM(data_size, self.num_spin_orb * self.num_bits_rot // 2, adjoint=self.adjoint)),
        }


@frozen
class InnerPrepare(Bloq):
    """Inner prepare for DF block encoding.

    Prepare state over $p$ register controlled on outer $l$ register.

    Currently we only provide costs as listed in Ref[1] without their corresponding decompositions.

    Args:
        num_aux: Dimension of auxiliary index for double factorized Hamiltonian. Call L in Ref[1].

    Registers:

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.030305).
            Step 3. Page 52.
    """

    num_aux: int
    num_spin_orb: int
    num_xi: int
    num_bits_rot_aa: int
    num_bits_offset: int
    num_bits_state_prep: int
    adjoint: bool = False

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f"In-Prep{dag}"

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(
            xi=(self.num_xi - 1).bit_length(),
            offset=self.num_bits_offset,
            rot=self.num_bits_rot_aa,
            succ_p=1,
            p=get_num_bits_lxi(self.num_aux, self.num_xi, self.num_spin_orb),
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
class DoubleFactorizationOneBody(BlockEncoding):
    r"""Block encoding of double factorization one-body Hamiltonian.

    Implements inner "half" of Fig. 15 in the reference. This block encoding is
    applied twice (with a reflection around the inner state preparation
    registers) to implement a (roughly) the square of this one-body operator.

    Note succ_pq will be allocated as an ancilla during decomposition and it is not relected on.

    Args:
        num_aux: Dimension of auxiliary index for double factorized Hamiltonian. Call L in Ref[1].
        num_spin_orb: The number of spin orbitals. Typically called N.
        num_bits_state_prep: The number of bits of precision for coherent alias
            sampling. Called aleph (fancy N) in Ref[1].
        num_bits_rot: Number of bits of precision for rotations for amplitude
            amplification in uniform state preparation. Called b_r in Ref[1].
        adjoint: Whether this bloq is daggered or not. This affects the QROM cost.

    Registers:

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.030305)
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
        return [
            Register("succ_l", bitsize=1),
            Register("succ_p", bitsize=1),
            Register("l_ne_zero", bitsize=1),
        ]

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
            num_bits_offset=num_bits_lxi,
            num_bits_state_prep=self.num_bits_state_prep,
            adjoint=False,
        )
        in_prep_dag = InnerPrepare(
            num_aux=self.num_aux,
            num_spin_orb=self.num_spin_orb,
            num_xi=self.num_xi,
            num_bits_rot_aa=self.num_bits_rot_aa,
            num_bits_offset=num_bits_lxi,
            num_bits_state_prep=self.num_bits_state_prep,
            adjoint=True,
        )
        n = self.num_spin_orb // 2
        rot = ProgRotGateArray(
            num_aux=self.num_aux,
            num_xi=self.num_xi,
            num_spin_orb=self.num_spin_orb,
            num_bits_rot=self.num_bits_rot,
            num_bits_offset=num_bits_lxi,
            adjoint=False,
        )
        rot_dag = ProgRotGateArray(
            num_aux=self.num_aux,
            num_xi=self.num_xi,
            num_spin_orb=self.num_spin_orb,
            num_bits_rot=self.num_bits_rot,
            num_bits_offset=num_bits_lxi,
            adjoint=True,
        )
        # 2*In-prep_l, addition, Rotations, 2*H, 2*SWAPS, subtraction
        return {
            (1, in_prep),  # in-prep_l listing 3 page 52/53
            (1, in_prep_dag),  # in_prep_l^dag
            (1, rot),  # rotate into system basis  listing 4 pg 54
            (4, TGate()),  # apply CCZ / CCCZ (this should be accounted for but is not currently)
            (1, rot_dag),  # Undo rotations
            (2, CSwapApprox(self.num_spin_orb // 2)),  # Swaps for spins
            (2, ArbitraryClifford(n=1)),  # 2 Hadamards for spin superposition
        }


@frozen
class OuterPrepare(Bloq):
    r"""Outer state preparation.

    Implements "the appropriate superposition state" on the $l$ register. i.e.

    $$
    |0\rangle = \sum_l a_l |l\rangle,
    $$

    where $a_l = \sqrt{\frac{c^{l}}{\sum_{l} c^{(l)}}$, and $c^{l} = \sum_{k} f_k^{(l)}^2$.

    Args:
        num_aux: Dimension of auxiliary index for double factorized Hamiltonian. Call L in Ref[1].
        adjoint: Whether to dagger the bloq or not.

    Registers:
        l: register to store L values for auxiliary index.
        p: spatial orbital index. range(0, num_spin_orb // 2)
        q: spatial orbital index. range(0, num_spin_orb // 2)
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

    Registers:
        l: register to store L values for auxiliary index.
        l_ne_zero: flag for one-body term.
        xi: rank for each l
        offset: offset for each DF factor.
        rot: rotation for amplitude amplification.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://arxiv.org/abs/2011.03494)
            Appendix C, page 52. Step 2.
    """

    num_aux: int
    num_spin_orb: int
    num_xi: int
    num_bits_rot_aa: int = 8
    ko: int = 1
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
            rot=self.num_bits_rot_aa,
            offset=get_num_bits_lxi(self.num_aux, self.num_xi, self.num_spin_orb),
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # listing 2 C29/30 page 52
        num_bits_xi = (self.num_xi - 1).bit_length()
        num_bits_offset = get_num_bits_lxi(self.num_aux, self.num_xi, self.num_spin_orb)
        bo = num_bits_xi + num_bits_offset + self.num_bits_rot_aa + 1
        print("this: ", bo, num_bits_xi, num_bits_offset, self.num_bits_rot_aa)
        return {(1, QROAM(self.num_aux + 1, bo, adjoint=self.adjoint))}


@frozen
class DoubleFactorization(BlockEncoding):
    r"""Block encoding of double factorization Hamiltonian.

    Implements Fig. 15 in the reference.

    Args:
        num_spin_orb: The number of spin orbitals. Typically called N.
        num_aux: Dimension of auxiliary index for double factorized Hamiltonian. Called L in Ref[1].
        num_xi: Rank of second factorization. Full rank implies xi = num_spin_orb // 2.
        num_bits_state_prep: The number of bits of precision for coherent alias
            sampling. Called aleph (fancy N) in Ref[1].
        num_bits_rot: Number of bits of precision for rotations for amplitude
            amplification in uniform state preparation. Called b_r in Ref[1].
        qroam_k_factor: QROAM blocking factor for data prepared over l (auxiliary) index.
            Defaults to 1 (i.e. use QROM).

    Registers:
        l: register to store L values for auxiliary index.
        succ_pq: flag for success of this state preparation.
        succ_l: flag for success of this state preparation.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            hypercontraction](https://journals.aps.org/prxquantum/pdf/10.1103/prxquantum.2.030305)
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
            SingleFactorization bloq with alt/keep values appropriately constructed.

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

    def build_composite_bloq(self, bb: 'BloqBuilder', **regs: SoquetT) -> Dict[str, 'SoquetT']:
        l = regs['l']
        sys = regs['sys']
        p = regs['p']
        spin = regs['spin']
        succ_l, l_ne_zero, theta, succ_p = bb.split(bb.allocate(4))
        num_bits_lxi = get_num_bits_lxi(self.num_aux, self.num_xi, self.num_spin_orb)
        num_bits_xi = (self.num_xi - 1).bit_length()
        xi = bb.allocate(num_bits_xi)
        offset = bb.allocate(num_bits_lxi)
        rot = bb.allocate(self.num_bits_rot_aa_inner)

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
        l, l_ne_zero, xi, rot, offset = bb.add(
            in_l_data_l, l=l, l_ne_zero=l_ne_zero, xi=xi, rot=rot, offset=offset
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
        succ_l, succ_p, l_ne_zero, p, spin, rot, state_prep_anc, sys, l, xi, offset = bb.add(
            one_body_sq,
            succ_l=succ_l,
            succ_p=succ_p,
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
        bb.free(state_prep_anc)
        in_l_data_l = OutputIndexedData(
            num_aux=self.num_aux,
            num_spin_orb=self.num_spin_orb,
            num_xi=self.num_xi,
            num_bits_rot_aa=self.num_bits_rot_aa_inner,
            adjoint=True,
        )
        l, l_ne_zero, xi, rot, offset = bb.add(
            in_l_data_l, l=l, l_ne_zero=l_ne_zero, xi=xi, rot=rot, offset=offset
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
        bb.free(rot)
        bb.free(succ_l)
        bb.free(succ_p)
        bb.free(l_ne_zero)
        bb.free(theta)
        out_regs = {'l': l, 'sys': sys, 'p': p, 'spin': spin, 'ctrl': regs['ctrl']}
        return out_regs
