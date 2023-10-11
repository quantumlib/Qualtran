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
from typing import Dict, Iterable, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
from attrs import frozen
from sympy import factorint

from qualtran import Bloq, BloqBuilder, Register, Signature, SoquetT
from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.block_encoding import BlockEncodeChebyshevPolynomial, BlockEncoding
from qualtran.bloqs.swap_network import CSwapApprox

if TYPE_CHECKING:
    from qualtran.resource_counting import SympySymbolAllocator


@frozen
class InnerPrepare(Bloq):
    """Inner prepare for THC block encoding.

    Prepares the state in Eq. B11 of Ref [1].

    Currently we only provide costs as listed in Ref[1] without their corresponding decompositions.

    Args:
        num_aux: Dimension of auxiliary index for double factorized Hamiltonian. Call L in Ref[1].
        num_spin_orb: The number of spin orbitals. Typically called N.
        num_bits_state_prep: The number of bits of precision for coherent alias
            sampling. Called aleph (fancy N) in Ref[1].
        num_bits_rot: Number of bits of precision for rotations for amplitude
            amplification in uniform state preparation. Called b_r in Ref[1].
        adjoint: Whether this bloq is daggered or not. This affects the QROM cost.
        kp1: QROAM blocking factor for data prepared over l (auxiliary) index.
            Defaults to 1 (i.e. QROM).
        kp1: QROAM blocking factor for data prepared over pq indicies. Defaults to 1 (i.e.) QROM.

    Registers:
        l: register to store L values for auxiliary index.
        p: spatial orbital index. range(0, num_spin_orb // 2)
        q: spatial orbital index. range(0, num_spin_orb // 2)
        succ_pq: flag for success of this state preparation.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.030305)
    """

    num_aux: int
    num_spin_orb: int
    num_bits_state_prep: int
    num_bits_rot: int = 8
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

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        n = (self.num_spin_orb // 2 - 1).bit_length()
        # prep uniform upper triangular.
        cost_a = 6 * n + 2 * self.num_bits_rot - 7
        # contiguous index
        cost_b = n**2 + n - 1
        # QROAM
        # Strictly speaking kp1 and kp2 can be different for the adjoint
        cost_c = int(np.ceil((self.num_aux + 1) / self.kp1))
        cost_c *= int(np.ceil((self.num_spin_orb**2 / 8 + self.num_spin_orb / 2) / self.kp2))
        bp = 2 * n + self.num_bits_state_prep + 2
        if self.adjoint:
            cost_c += self.kp1 * self.kp2
        else:
            cost_c += bp * (self.kp1 * self.kp2 - 1)
        # inequality test
        cost_d = self.num_bits_state_prep
        # controlled swap
        cost_e = 2 * n
        return {(4 * (cost_a + cost_b + cost_c + cost_d + cost_e), TGate())}


@frozen
class SELECT(Bloq):
    r"""Single Factorization SELECT bloq.

    Implements selected Majorana Fermion operation. Placeholder for the moment.

    Args:
        num_spin_orb: The number of spin orbitals. Typically called N.
        additional_control: whether to control on $l \ne zero$ or not.

    Registers:
        l: register to store L values for auxiliary index.
        p: spatial orbital index. range(0, num_spin_orb // 2)
        q: spatial orbital index. range(0, num_spin_orb // 2)
        succ_pq: flag for success of this state preparation.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.030305)
    """

    num_spin_orb: int
    additional_control: bool = False

    @cached_property
    def signature(self) -> Signature:
        n = (self.num_spin_orb // 2 - 1).bit_length()
        if self.additional_control:
            return Signature.build(p=n, q=n, alpha=1, succ_pq=1, succ_l=1, l_ne_zero=1)
        return Signature.build(p=n, q=n, alpha=1, succ_pq=1, succ_l=1)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        return {(2 * (self.num_spin_orb - 2), TGate())}


@frozen
class OuterPrepare(Bloq):
    r"""Outer state preparation.

    Implements "the appropriate superposition state" on the $l$ register.

    Args:
        num_aux: Dimension of auxiliary index for double factorized Hamiltonian. Call L in Ref[1].
        kl: QROAM blocking factor for data prepared over l (auxiliary) index.
            Defaults to 1 (i.e. QROM).
        adjoint: Whether to dagger the bloq or not.

    Registers:
        l: register to store L values for auxiliary index.
        p: spatial orbital index. range(0, num_spin_orb // 2)
        q: spatial orbital index. range(0, num_spin_orb // 2)
        succ_l: flag for success of this state preparation.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://arxiv.org/pdf/2011.03494.pdf)
            Appendix C, page 51 and 52
    """

    num_aux: int
    kl: int
    num_bits_state_prep: int
    num_bits_rot: int = 8
    adjoint: bool = False

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f"OuterPrep{dag}"

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(l=self.num_aux.bit_length(), succ_l=1)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # https://arxiv.org/pdf/2011.03494.pdf page 44
        factors = factorint(self.num_aux)
        eta = factors[min(list(sorted(factors.keys())))]
        if self.num_aux % 2 == 1:
            eta = 0
        num_bits_l = self.num_aux.bit_length()
        cost_a = 3 * num_bits_l - 3 * eta + 2 * self.num_bits_rot - 9
        output_size = num_bits_l + self.num_bits_state_prep
        if self.adjoint:
            cost_b = int(np.ceil((self.num_aux + 1) / self.kl)) + self.kl
        else:
            cost_b = int(np.ceil((self.num_aux + 1) / self.kl)) + output_size * (self.kl - 1)
        cost_c = self.num_bits_state_prep
        cost_d = num_bits_l
        return {(4 * (cost_a + cost_b + cost_c + cost_d), TGate())}


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
        kp1: QROAM blocking factor for data prepared over l (auxiliary) index.
            Defaults to 1 (i.e. QROM).
        kp1: QROAM blocking factor for data prepared over pq indicies. Defaults to 1 (i.e.) QROM.

    Registers:
        l: register to store L values for auxiliary index.
        p: spatial orbital index. range(0, num_spin_orb // 2)
        q: spatial orbital index. range(0, num_spin_orb // 2)
        succ_l: flag for success of this state preparation.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.030305)
    """
    num_aux: int
    num_spin_orb: int
    num_bits_state_prep: int
    num_bits_rot: int = 8
    adjoint: bool = False
    kp1: int = 1
    kp2: int = 1

    @property
    def control_registers(self) -> Iterable[Register]:
        return [Register("succ_l", bitsize=1), Register("l_ne_zero", bitsize=1)]

    @property
    def selection_registers(self) -> Iterable[Register]:
        return [Register("l", bitsize=self.num_aux.bit_length())]

    @property
    def target_registers(self) -> Iterable[Register]:
        return [Register("sys", bitsize=self.num_spin_orb)]

    @property
    def junk_registers(self) -> Iterable[Register]:
        return [
            Register("p", bitsize=self.num_spin_orb // 2),
            Register("q", bitsize=self.num_spin_orb // 2),
            Register("swap_pq", bitsize=1),
            Register("alpha", bitsize=1),
        ]

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        iprep = InnerPrepare(
            self.num_aux,
            self.num_spin_orb,
            self.num_bits_state_prep,
            self.num_bits_rot,
            adjoint=False,
            kp1=self.kp1,
            kp2=self.kp2,
        )
        iprep_dag = InnerPrepare(
            self.num_aux,
            self.num_spin_orb,
            self.num_bits_state_prep,
            self.num_bits_rot,
            adjoint=True,
            kp1=self.kp1,
            kp2=self.kp2,
        )
        n = (self.num_spin_orb // 2 - 1).bit_length()
        # prepare + prepare^dag, 2 SWAPS, SELECT
        return {
            (1, iprep),
            (1, iprep_dag),
            (2, CSwapApprox(n)),
            (1, SELECT(num_spin_orb=self.num_spin_orb)),
        }


@frozen
class DoubleFactorization(BlockEncoding):
    r"""Block encoding of double factorization Hamiltonian.

    Implements Fig. 15 in the reference.

    Args:
        num_spin_orb: The number of spin orbitals. Typically called N.
        num_aux: Dimension of auxiliary index for double factorized Hamiltonian. Called L in Ref[1].
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
    num_bits_state_prep: int
    num_bits_rot: int = 8
    qroam_k_factor: int = 1

    @classmethod
    def build(cls, one_body_ham, factorized_two_body_ham) -> 'DoubleFactorization':
        """Factory method to build double factorization block encoding given Hamiltonian inputs.

        Args:
            one_body_ham: One body hamiltonian ($T_{pq}$') matrix elements. (includes exchange terms).
            factorized_two_body_ham: One body hamiltonian ($W^{(l)}_{pq}$).

        Returns:
            SingleFactorization bloq with alt/keep values appropriately constructed.

        Refererences:
            [Even More Efficient Quantum Computations of Chemistry Through Tensor
                hypercontraction]
                (https://journals.aps.org/prxquantum/pdf/10.1103/prxquantum.2.030305). Eq. B7 pg 43.
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
        return [
            Register("succ_l", bitsize=1),
            Register("theta", bitsize=1),
            Register("l_ne_zero", bitsize=1),
            Register("p", bitsize=self.num_spin_orb // 2),
            Register("q", bitsize=self.num_spin_orb // 2),
            Register("swap_pq", bitsize=1),
            Register("alpha", bitsize=1),
        ]

    def build_composite_bloq(self, bb: 'BloqBuilder', **regs: SoquetT) -> Dict[str, 'SoquetT']:
        """ """
        succ_l = regs['succ_l']
        l_ne_zero = regs['l_ne_zero']
        l = regs['l']
        sys = regs['sys']
        p = regs['p']
        q = regs['q']
        swap_pq = regs['swap_pq']
        alpha = regs['alpha']
        ctrl = regs['ctrl']
        theta = regs['theta']
        # prepare_l
        # epsilon = 2**-self.num_bits_state_prep / len(self.out_prep_probs)
        outer_prep = OuterPrepare(
            self.num_aux,
            kl=self.qroam_k_factor,
            num_bits_state_prep=self.num_bits_state_prep,
            num_bits_rot=self.num_bits_rot,
        )
        l, succ_l, l_ne_zero = bb.add(outer_prep, l=l, succ_l=succ_l, l_ne_zero=l_ne_zero)
        one_body = DoubleFactorizationOneBody(
            self.num_aux, self.num_spin_orb, self.num_bits_state_prep, self.num_bits_rot
        )
        one_body_sq = BlockEncodeChebyshevPolynomial(one_body, order=2)
        succ_l, l_ne_zero, l, p, q, swap_pq, alpha, sys = bb.add(
            one_body_sq,
            succ_l=succ_l,
            l_ne_zero=l_ne_zero,
            l=l,
            p=p,
            q=q,
            swap_pq=swap_pq,
            alpha=alpha,
            sys=sys,
        )
        # prepare_l^dag
        outer_prep = OuterPrepare(
            self.num_aux,
            kl=self.qroam_k_factor,
            num_bits_state_prep=self.num_bits_state_prep,
            num_bits_rot=self.num_bits_rot,
            adjoint=True,
        )
        l, succ_l, l_ne_zero = bb.add(outer_prep, l=l, succ_l=succ_l, l_ne_zero=l_ne_zero)
        out_regs = {
            'l': l,
            'succ_l': succ_l,
            'l_ne_zero': l_ne_zero,
            'sys': sys,
            'p': p,
            'q': q,
            'swap_pq': swap_pq,
            'alpha': alpha,
            'theta': theta,
            'ctrl': ctrl,
        }
        return out_regs
