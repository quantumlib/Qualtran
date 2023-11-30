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
"""SELECT for the sparse chemistry Hamiltonian in second quantization."""

from functools import cached_property
from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING

from attrs import frozen

from qualtran import bloq_example, BloqBuilder, BloqDocSpec, Register, SelectionRegister, SoquetT
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.select_and_prepare import SelectOracle

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class SelectSparse(SelectOracle):
    r"""SELECT oracle for the sparse Hamiltonian.

    Implements the two applications of Fig. 13.

    Args:
        num_spin_orb: The number of spin orbitals. Typically called N.

    Registers:
        flag_1b: a single qubit to flag whether the one-body Hamiltonian is to
            be applied or not during SELECT.
        swap_pq: a |+> state to restore the symmetries of the p and q indices.
        swap_rs: a |+> state to restore the symmetries of the r and s indices.
        swap_pqrs: a |+> state to restore the symmetries of between (pq) and (rs).
        theta: sign qubit.
        pqrs: the register to store the spatial orbital index.
        alpha: spin for (pq) indicies.
        beta: spin for (rs) indicies.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            hypercontraction](https://arxiv.org/abs/2011.03494) Fig 13.
    """
    num_spin_orb: int
    control_val: Optional[int] = None

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return () if self.control_val is None else (Register('control', 1),)

    @cached_property
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        return (
            SelectionRegister(
                "p",
                bitsize=(self.num_spin_orb // 2 - 1).bit_length(),
                iteration_length=self.num_spin_orb // 2,
            ),
            SelectionRegister(
                "q",
                bitsize=(self.num_spin_orb // 2 - 1).bit_length(),
                iteration_length=self.num_spin_orb // 2,
            ),
            SelectionRegister(
                "r",
                bitsize=(self.num_spin_orb // 2 - 1).bit_length(),
                iteration_length=self.num_spin_orb // 2,
            ),
            SelectionRegister(
                "s",
                bitsize=(self.num_spin_orb // 2 - 1).bit_length(),
                iteration_length=self.num_spin_orb // 2,
            ),
            SelectionRegister("alpha", bitsize=1),
            SelectionRegister("beta", bitsize=1),
            SelectionRegister("flag_1b", bitsize=1),
        )

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register("sys", bitsize=self.num_spin_orb),)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Pg 30, enumeration 1: 2 applications of SELECT in Fig. 13, one of
        # which is not controlled (for the two body part of the Ham). The figure
        # is a bit misleading as applying that circuit twice would square the
        # value in the sign. In reality the Z to pick up the sign could be done
        # after prepare (but only once).
        # In practice we would apply the selected majoranas to (p, q, alpha) and then (r, s, beta).
        return {(Toffoli(), (4 * self.num_spin_orb - 6))}
