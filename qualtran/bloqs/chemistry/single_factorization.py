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

import abc
from functools import cached_property
from typing import Dict, Iterable, Optional, Set, Tuple, TYPE_CHECKING

import cirq
import numpy as np
from attrs import frozen
from cirq_ft.algos.state_preparation import StatePreparationAliasSampling
from sympy import factorint

from qualtran import Bloq, BloqBuilder, Register, Signature, SoquetT
from qualtran.bloqs.and_bloq import MultiAnd
from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.block_encoding import BlockEncodeChebyshevPolynomial, BlockEncoding
from qualtran.bloqs.util_bloqs import ArbitraryClifford
from qualtran.cirq_interop import CirqGateAsBloq

if TYPE_CHECKING:
    from qualtran.resource_counting import SympySymbolAllocator


@frozen
class SingleFactorizationOneBody(BlockEncoding):
    r"""Block encoding of single factorization one-body Hamiltonian."""
    num_spin_orb: int

    @property
    def control_registers(self) -> Iterable[Register]:
        return [Register("succ_l", bitsize=1), Register("l_ne_zero", bitsize=1)]

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

    def build_composite_bloq(self, bb: 'BloqBuilder', **regs: SoquetT) -> Dict[str, 'SoquetT']:
        return regs

    def bloq_counts()


@frozen
class OuterPrepare(Bloq):
    """Prepare over l"""

    num_aux: int
    qroam_factor: int
    num_bits_state_prep: int
    num_bits_rot: int = 8
    adjoint: bool = False

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f"Prep{dag}"

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(l=self.num_aux.bit_length(), succ_l=1, l_ne_zero=1)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # prepaing on L + 1 basis states and an inequality test to flag l_ne_zero to on
        # eta is a number such that L + 1 is divisible by 2^\eta
        # if eta is a power of 2 then we see a reduction in the cost
        # https://arxiv.org/pdf/2011.03494.pdf page 44
        factors = factorint(self.num_aux)
        eta = factors[min(list(sorted(factors.keys())))]
        if self.num_aux % 2 == 1:
            eta = 0
        num_bits_l = self.num_aux.bit_length()
        cost_1 = 3 * num_bits_l - 3 * eta + 2 * self.num_bits_rot - 9
        output_size = num_bits_l + self.num_bits_state_prep + 2
        cost_2 = int(np.ceil((self.num_aux + 1) / self.qroam_factor)) + output_size * (
            self.qroam_factor - 1
        )
        cost_3 = self.num_bits_state_prep
        cost_4 = num_bits_l + 1
        return {(4 * (cost_1 + cost_2 + cost_3 + cost_4), TGate())}


@frozen
class InnerPrepare(Bloq):
    """Prepare over l"""

    num_aux: int
    qroam_factor: int
    num_bits_state_prep: int
    num_bits_rot: int = 8
    adjoint: bool = False

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f"Prep{dag}"

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(l=self.num_aux.bit_length(), succ_l=1, l_ne_zero=1)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # prepaing on L + 1 basis states and an inequality test to flag l_ne_zero to on
        # eta is a number such that L + 1 is divisible by 2^\eta
        # if eta is a power of 2 then we see a reduction in the cost
        # https://arxiv.org/pdf/2011.03494.pdf page 44
        factors = factorint(self.num_aux)
        eta = factors[min(list(sorted(factors.keys())))]
        if self.num_aux % 2 == 1:
            eta = 0
        num_bits_l = self.num_aux.bit_length()
        cost_1 = 3 * num_bits_l - 3 * eta + 2 * self.num_bits_rot - 9
        output_size = num_bits_l + self.num_bits_state_prep + 2
        cost_2 = int(np.ceil((self.num_aux + 1) / self.qroam_factor)) + output_size * (
            self.qroam_factor - 1
        )
        cost_3 = self.num_bits_state_prep
        cost_4 = num_bits_l + 1
        return {(4 * (cost_1 + cost_2 + cost_3 + cost_4), TGate())}


@frozen
class SingleFactorization(BlockEncoding):
    r"""Block encoding of single factorization one-body Hamiltonian."""
    num_spin_orb: int
    num_aux: int
    num_bits_state_prep: int
    num_bits_rot: int = 8
    qroam_k_factor: int = 1

    @property
    def control_registers(self) -> Iterable[Register]:
        return [Register('ctrl', bitsize=1)]

    @property
    def target_registers(self) -> Iterable[Register]:
        return [Register("sys", bitsize=self.num_spin_orb)]

    @property
    def junk_registers(self) -> Iterable[Register]:
        return [
            Register("succ_l", bitsize=1),
            Register("l_ne_zero", bitsize=1),
            Register("l", bitsize=self.num_aux.bit_length()),
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
        # prepare_l
        # epsilon = 2**-self.num_bits_state_prep / len(self.out_prep_probs)
        outer_prep = OuterPrepare(
            self.num_aux,
            qroam_factor=self.qroam_k_factor,
            num_bits_state_prep=self.num_bits_state_prep,
            num_bits_rot=self.num_bits_rot,
        )
        l, succ_l, l_ne_zero = bb.add(outer_prep, l=l, succ_l=succ_l, l_ne_zero=l_ne_zero)
        one_body = SingleFactorizationOneBody(self.num_spin_orb)
        one_body_sq = BlockEncodeChebyshevPolynomial(one_body, order=2)
        sys, p, q, swap_pq, alpha, succ_l, l_ne_zero = bb.add(
            one_body_sq,
            sys=sys,
            p=p,
            q=q,
            swap_pq=swap_pq,
            alpha=alpha,
            succ_l=succ_l,
            l_ne_zero=l_ne_zero,
        )
        # prepare_l^dag
        outer_prep = OuterPrepare(
            self.num_aux,
            qroam_factor=self.qroam_k_factor,
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
            'ctrl': ctrl,
        }
        return out_regs
