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
"""PREPARE for the sparse chemistry Hamiltonian in second quantization."""

from functools import cached_property
from typing import Dict, Set, Tuple, TYPE_CHECKING

import numpy as np
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    BloqExample,
    Register,
    SelectionRegister,
    Signature,
    SoquetT,
)
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.chemistry.black_boxes import PrepareUniformSuperposition
from qualtran.bloqs.select_and_prepare import PrepareOracle
from qualtran.bloqs.swap_network import CSwapApprox

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class PrepareSparse(PrepareOracle):
    r"""Prepare oracle for the sparse chemistry Hamiltonian

    Prepare the state:

    $$
        |0\rangle|+\rangle|0\rangle|0\rangle
        \sum_{\sigma}\sum_{pq}
        \sqrt{\frac{T_{pq}'}{2\lambda}}
        |\theta_{pq}^T\rangle|pq\sigma\rangle|000\rangle
        +|1\rangle|+\rangle|+\rangle|+\rangle
        \sum_{\alpha\beta}
        \sum_{pqrs}
        \sqrt{\frac{\tilde{V}_{pqrs}'}{2\lambda}}
        |\theta_{pqrs}^V\rangle|pq\alpha\rangle|rs\beta\rangle
    $$

    Args:
        num_spin_orb: The number of spin orbitals.
        num_non_zero: The number of non-zero matrix elements.
        num_bits_state_prep: the number of bits of precision for state
            preparation. This will control the size of the keep register.
        num_bits_rot_aa: The number of bits of precision for the single-qubit
            rotation for amplitude amplification during the uniform state
            preparataion. Default 8.
        adjoint: Whether we are apply PREPARE or PREPARE^dag
        k: qroam blocking factor.

    Registers:
        pqrs: the register to store the spatial orbital index.
        theta: sign qubit.
        alpha: spin for (pq) indicies.
        beta: spin for (rs) indicies.
        swap_pq: a |+> state to restore the symmetries of the p and q indices.
        swap_rs: a |+> state to restore the symmetries of the r and s indices.
        swap_pqrs: a |+> state to restore the symmetries of between (pq) and (rs).
        flag_1b: a single qubit to flag whether the one-body Hamiltonian is to
            be applied or not during SELECT.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            hypercontraction](https://arxiv.org/abs/2011.03494) Eq. A11.
    """
    num_spin_orb: int
    num_non_zero: int
    num_bits_state_prep: int
    num_bits_rot_aa: int = 8
    adjoint: bool = False
    k: int = 1

    @cached_property
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        # issue here in that pqrs should not be reflected on.
        return [
            SelectionRegister(
                "d",
                bitsize=(self.num_non_zero - 1).bit_length(),
                iteration_length=self.num_non_zero,
            ),
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
            SelectionRegister("sym_swaps", bitsize=1),
            SelectionRegister("swap_pq", bitsize=1),
            SelectionRegister("swap_rs", bitsize=1),
            SelectionRegister("swap_pqrs", bitsize=1),
            SelectionRegister("flag_1b", bitsize=1),
        ]

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        num_bits_spat = (self.num_spin_orb // 2 - 1).bit_length()
        if self.adjoint:
            num_toff_qrom = int(np.ceil(self.num_non_zero / self.k)) + self.k  # A15
        else:
            output_size = self.num_bits_state_prep + 8 * num_bits_spat + 4
            num_toff_qrom = int(np.ceil(self.num_non_zero / self.k)) + output_size * (
                self.k - 1
            )  # A14
        qrom_cost = (Toffoli(), num_toff_qrom)
        if self.adjoint:
            return {
                (PrepareUniformSuperposition(self.num_non_zero, self.num_bits_rot_aa), 1),
                qrom_cost,
            }
        swap_cost_state_prep = (CSwapApprox(num_bits_spat), 4 + 4)  # 2. pg 39
        ineq_cost_state_prep = (Toffoli(), (self.num_bits_state_prep + 1))  # 2. pg 39

        return {
            (PrepareUniformSuperposition(self.num_non_zero, self.num_bits_rot_aa), 1),
            qrom_cost,
            swap_cost_state_prep,
            ineq_cost_state_prep,
        }


@bloq_example
def _prepare_sparse() -> PrepareSparse:
    num_non_zero = 1011
    num_spin_orb = 18
    num_bits_state_prep = 11
    prep = PrepareSparse(num_spin_orb, num_non_zero, num_bits_state_prep)
    return prep


_SPARSE_PREPARE = BloqDocSpec(
    bloq_cls=PrepareSparse,
    import_line='from qualtran.bloqs.chemistry.sparse.prepare import PrepareSparse',
    examples=(_prepare_sparse,),
)
