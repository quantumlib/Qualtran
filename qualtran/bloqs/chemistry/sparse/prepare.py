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
from typing import Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
from attrs import frozen
from sympy import factorint

from qualtran import Bloq, Register, Signature
from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.swap_network import CSwapApprox

if TYPE_CHECKING:
    from qualtran.resource_counting import SympySymbolAllocator


@frozen
class PrepareUniformSuperpositionSparse(Bloq):
    r"""Prepare a uniform superposition over $d$ basis states.

    Before preparing the state in Eq. A11 of the reference we need to prepare a
    uniform superposition over $d$ basis states, where $d$ is the number of
    non-zero entries in our Hamiltonian. We will use this state to ultimately
    output the non-zero (p,q)_j and (p,q,r,s)_j tuples of non-zero symmetry
    inequivalent matrix element indicies.

    Args:
        num_non_zero: The number of non-zero matrix elements.
        num_bits_rot_aa: The number of bits of precision for the single-qubit
            rotation for amplitude amplification during the uniform state
            preparataion. Default 8.

    Registers:
        d: the register to prepare the uniform superposition on.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            hypercontraction](https://arxiv.org/abs/2011.03494) Page 39.
    """
    num_non_zero: int
    num_bits_rot_aa: int = 8

    @cached_property
    def signature(self) -> Signature:
        regs = [Register('d', (self.num_non_zero - 1).bit_length())]
        return Signature(regs)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        factors = factorint(self.num_non_zero)
        eta = factors[min(list(sorted(factors.keys())))]
        if self.num_non_zero % 2 == 1:
            eta = 0
        uniform_prep = (
            3 * (self.num_non_zero - 1).bit_length() - 3 * eta + 2 * self.num_bits_rot_aa - 9
        )
        return {(4 * uniform_prep, TGate())}


@frozen
class PrepareSparse(Bloq):
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
    def signature(self) -> Signature:
        return Signature(
            [
                Register("pqrs", (self.num_spin_orb // 2 - 1).bit_length(), shape=(4,)),
                Register("theta", 1),
                Register("alpha", 1),
                Register("beta", 1),
                Register("swap_pq", 1),
                Register("swap_rs", 1),
                Register("swap_pqrs", 1),
                Register("flag_1b", 1),
            ]
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        num_bits_spat = (self.num_spin_orb // 2 - 1).bit_length()
        if self.adjoint:
            num_toff_qrom = int(np.ceil(self.num_non_zero / self.k)) + self.k  # A15
        else:
            output_size = self.num_bits_state_prep + 8 * num_bits_spat + 4
            num_toff_qrom = int(np.ceil(self.num_non_zero / self.k)) + output_size * (
                self.k - 1
            )  # A14
        qrom_cost = (4 * num_toff_qrom, TGate())
        if self.adjoint:
            return {
                (1, PrepareUniformSuperpositionSparse(self.num_non_zero, self.num_bits_rot_aa)),
                qrom_cost,
            }
        swap_cost_state_prep = (4 + 4, CSwapApprox(num_bits_spat))  # 2. pg 39
        ineq_cost_state_prep = (4 * (self.num_bits_state_prep + 1), TGate())  # 2. pg 39
        return {
            (1, PrepareUniformSuperpositionSparse(self.num_non_zero, self.num_bits_rot_aa)),
            qrom_cost,
            swap_cost_state_prep,
            ineq_cost_state_prep,
        }
