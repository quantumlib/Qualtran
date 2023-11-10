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


@frozen
class ProgRotGateArray(Bloq):
    r"""Rotate to to/from MO basis so-as-to apply number operators in DF basis.

    This is really a subclass of qualtran's ProgrammableRotationGateArray

    Args:
        num_aux: Dimension of auxiliary index for double factorized Hamiltonian. Called L in Ref[1].
        num_xi: Rank of second factorization. Full rank implies $Xi = num_spin_orb // 2$.
        num_spin_orb: The number of spin orbitals. Typically called N.
        num_bits_rot: Number of bits of precision for rotations
            amplification in uniform state preparation. Called $\beth$ in Ref[1].

    Registers:
        offset: Offset for p register.
        p: Register for inner state preparation. This is of size $\ceil \log (L \Xi + N / 2)$.
        rotatations: Data register storing rotations.
        spin_sel: A single qubit register for spin.
        sys_a: The system register for alpha electrons.
        sys_b: The system register for beta electons.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://arxiv.org/abs/2011.03494).
            Step 4. Page 53.
    """

    num_aux: int
    num_xi: int
    num_spin_orb: int
    num_bits_rot: int
    adjoint: bool = False

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f"Rotations{dag}"

    @cached_property
    def signature(self) -> Signature:
        num_bits_lxi = get_num_bits_lxi(self.num_aux, self.num_xi, self.num_spin_orb)
        return Signature.build(
            offset=num_bits_lxi,
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
