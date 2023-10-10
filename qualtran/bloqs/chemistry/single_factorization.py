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

from qualtran import Bloq, BloqBuilder, Register, Signature, SoquetT
from qualtran.bloqs.and_bloq import MultiAnd
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
    @abc.abstractmethod
    def control_registers(self) -> Iterable[Register]:
        return [Register("succ_l", bitsize=1), Register("l_ne_zero", bitsize=1)]

    @property
    @abc.abstractmethod
    def target_registers(self) -> Iterable[Register]:
        return [Register("sys", bitsize=self.num_spin_orb)]

    @property
    @abc.abstractmethod
    def junk_registers(self) -> Iterable[Register]:
        return [
            Register("p", bitsize=self.num_spin_orb // 2),
            Register("q", bitsize=self.num_spin_orb // 2),
            Register("swap_pq", bitsize=1),
            Register("alpha", bitsize=1),
        ]


@frozen
class SingleFactorization(BlockEncoding):
    r"""Block encoding of single factorization one-body Hamiltonian."""
    num_spin_orb: int
    num_aux: int
    num_bits_state_prep: int

    @property
    @abc.abstractmethod
    def control_registers(self) -> Iterable[Register]:
        return [Register('ctrl', bitsize=1)]

    @property
    @abc.abstractmethod
    def target_registers(self) -> Iterable[Register]:
        return [Register("sys", bitsize=self.num_spin_orb)]

    @property
    @abc.abstractmethod
    def junk_registers(self) -> Iterable[Register]:
        return [
            Register("succ_l", bitsize=1),
            Register("l_ne_zero", bitsize=1)
            Register("p", bitsize=self.num_spin_orb // 2),
            Register("q", bitsize=self.num_spin_orb // 2),
            Register("swap_pq", bitsize=1),
            Register("alpha", bitsize=1),
            ]

    def build_composite_bloq(self, bb: 'BloqBuilder', **regs: SoquetT) -> Dict[str, 'SoquetT']:
        """
        """
        out_regs = {}
        # prepare_l
        epsilon = 2**-self.num_bits_state_prep / len(self.out_prep_probs)
        outer_prep = CirqGateAsBloq(StatePreparationAliasSampling.from_lcu_probs(self.outer_prep_probs, epsilon=epsilon))
        l, succ_l, l_ne_zero = bb.add(
            outer_prep, l=l, succ_l=succ_l, l_ne_zero=l_ne_zero
        )
        one_body = SingleFactorizationOneBody(self.num_spin_orb)
        one_body_sq = BlockEncodeChebyshevPolynomial(one_body, order=2)
        # prepare_l^dag
        # Is this the correct adjoint?
        l, succ_l, l_ne_zero = bb.add(
            outer_prep, l=l, succ_l=succ_l, l_ne_zero=l_ne_zero
        )
        return out_regs
