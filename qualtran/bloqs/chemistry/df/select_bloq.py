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
r"""Bloqs for the applying number operators to system for the DF Hamiltonian."""

from functools import cached_property
from typing import TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, QAny, QBit, Register, Signature
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.chemistry.black_boxes import QROAM

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@frozen
class ProgRotGateArray(Bloq):
    r"""Rotate to/from MO basis so-as-to apply number operators in DF basis.

    An actual implementation of should derive from ProgrammableRotationGateArray.

    Args:
        num_aux: Dimension of auxiliary index for double factorized Hamiltonian. Called L in Ref[1].
        num_spin_orb: The number of spin orbitals. Typically called N.
        num_eig: The total number of eigenvalues.
        num_bits_rot: Number of bits of precision for rotations
            amplification in uniform state preparation. Called $\beth$ in Ref[1].

    Registers:
        offset: Offset for p register.
        p: Register for inner state preparation. This is of size $\ceil \log (L \Xi + N / 2)$.
        rotatations: Data register storing rotations.
        spin_sel: A single qubit register for spin.
        sys: The system register.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://arxiv.org/abs/2011.03494). Step 4. Page 53.
    """

    num_aux: int
    num_spin_orb: int
    num_eig: int
    num_bits_rot: int

    @cached_property
    def signature(self) -> Signature:
        nlxi = (self.num_eig + self.num_spin_orb // 2 - 1).bit_length()
        nxi = (self.num_spin_orb // 2 - 1).bit_length()
        return Signature(
            [
                Register('offset', QAny(nlxi)),
                Register('p', QAny(nxi)),
                Register('rotations', QAny(bitsize=(self.num_spin_orb // 2) * self.num_bits_rot)),
                Register('spin', QBit()),
                Register('sys', QAny(bitsize=self.num_spin_orb // 2), shape=(2,)),
            ]
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # Step 4 in the reference.
        nlxi = (self.num_eig + self.num_spin_orb // 2 - 1).bit_length()
        nxi = (self.num_spin_orb // 2 - 1).bit_length()
        cost_a = nlxi - 1  # contiguous register
        data_size = self.num_eig + self.num_spin_orb // 2
        cost_c = self.num_spin_orb * (self.num_bits_rot - 2)  # apply rotations
        return {
            Toffoli(): (cost_a + cost_c),
            QROAM(data_size, self.num_spin_orb * self.num_bits_rot // 2): 1,
        }
