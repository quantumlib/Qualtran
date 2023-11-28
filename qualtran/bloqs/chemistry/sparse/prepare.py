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
from attrs import field, frozen
from numpy.typing import NDArray

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
from qualtran.bloqs.arithmetic.comparison import LessThanEqual
from qualtran.bloqs.basic_gates import CSwap, Hadamard, Toffoli
from qualtran.bloqs.basic_gates.z_basis import ZGate
from qualtran.bloqs.chemistry.black_boxes import (
    PrepareUniformSuperposition as BBPrepareUniformSuperposition,
)
from qualtran.bloqs.controlled_bloq import ControlledBloq
from qualtran.bloqs.on_each import OnEach
from qualtran.bloqs.prepare_uniform_superposition import PrepareUniformSuperposition
from qualtran.bloqs.select_and_prepare import PrepareOracle
from qualtran.bloqs.select_swap_qrom import find_optimal_log_block_size, SelectSwapQROM
from qualtran.bloqs.swap_network import CSwapApprox
from qualtran.linalg.lcu_util import preprocess_lcu_coefficients_for_reversible_sampling

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
    ind: Tuple[int, ...] = field(repr=False)
    alt: Tuple[int, ...] = field(repr=False)
    keep: Tuple[int, ...] = field(repr=False)
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
            SelectionRegister("sigma", bitsize=self.num_bits_state_prep),
            SelectionRegister("alpha", bitsize=1),
            SelectionRegister("beta", bitsize=1),
            SelectionRegister("rot_aa", bitsize=1),
            SelectionRegister("swap_pq", bitsize=1),
            SelectionRegister("swap_rs", bitsize=1),
            SelectionRegister("swap_pqrs", bitsize=1),
        ]

    @cached_property
    def junk_registers(self) -> Tuple[SelectionRegister, ...]:
        return (
            Register('alt_pqrs', bitsize=(self.num_spin_orb // 2 - 1).bit_length(), shape=(4,)),
            Register('theta', bitsize=1, shape=(2,)),
            Register('keep', bitsize=self.num_bits_state_prep),
            Register("less_than", bitsize=1),
            Register("flag_1b", bitsize=(2,)),
        )

    @classmethod
    def from_hamiltonian_coeffs(
        cls,
        num_spin_orb: int,
        integrals: NDArray[np.float64],
        indicies: NDArray[np.int_],
        num_bits_state_prep: int = 8,
        num_bits_rot_aa: int = 8,
    ) -> 'PrepareSparse':
        r"""Factory method to build PrepareSparse from Hamiltonian coefficients.

        Args:
            num_spin_orb: The number of spin orbitals.
            integrals: sparsified, spin-free, symmetry inequivalent Hamiltonian integrals.
                We assume that the first $N\times (N+1)/2$ terms account for the
                upper triangular part of the one-body Hamiltonian $T'_{pq}$, and
                that the rest of the entries are the non-zero (by sparsity), symmetry
                inequivalent ($p \geq q$, $r\geq s$, $pq \geq rs$) chemist's integrals (pq|rs).
            indicies: An Array of length-4 tuples corresponding indicies
                (p,q,r,s) of the non-zero matrix elements in the Hamiltonian. For
                the one-body term the r and s terms are not accessed.
            num_bits_state_prep: The number of bits for the state prepared during alias sampling.
            num_bits_rot_aa: The number of bits of precision for the single-qubit
                rotation for amplitude amplification during the uniform state
                preparataion. Default 8.

        Returns:
            Constructed PrepareSparse object.

        Refererences:
            [Even More Efficient Quantum Computations of Chemistry Through Tensor
                hypercontraction](https://arxiv.org/abs/2011.03494) Eq. A11.
            [Qubitization of Arbitrary Basis Quantum Chemistry Leveraging
            Sparsity and Low Rank
            Factorization](https://quantum-journal.org/papers/q-2019-12-02-208/)
            Sec 5 page 15
        """
        num_non_zero = len(integrals)
        alt, keep, mu = preprocess_lcu_coefficients_for_reversible_sampling(
            integrals, epsilon=2**-num_bits_state_prep / num_non_zero
        )
        assert mu == num_bits_rot_aa
        return PrepareSparse(
            num_spin_orb,
            num_non_zero,
            num_bits_state_prep,
            tuple(indicies),
            tuple(alt),
            tuple(keep),
            num_bits_rot_aa=num_bits_rot_aa,
        )

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        d: 'SoquetT',
        p: 'SoquetT',
        q: 'SoquetT',
        r: 'SoquetT',
        s: 'SoquetT',
        sigma: 'SoquetT',
        alpha: 'SoquetT',
        beta: 'SoquetT',
        rot_aa: 'SoquetT',
        swap_pq: 'SoquetT',
        swap_rs: 'SoquetT',
        swap_pqrs: 'SoquetT',
        alt_pqrs: 'SoquetT',
        theta: 'SoquetT',
        keep: 'SoquetT',
        less_than: 'SoquetT',
        flag_1b: 'SoquetT',
    ) -> Dict[str, 'SoquetT']:
        # 1. Prepare \sum_d |d\rangle
        d = bb.add(PrepareUniformSuperposition(self.num_non_zero), target=d)
        # 2. QROM the ind_d alt_d values
        n_n = (self.num_spin_orb // 2 - 1).bit_length()
        target_bitsizes = (n_n,) * 8 + (self.num_bits_state_prep) + (1,) * 4
        block_size = 2 ** find_optimal_log_block_size(self.num_non_zero, sum(target_bitsizes))
        qrom = SelectSwapQROM(
            self.ind,
            self.alt,
            self.theta,
            target_bitsizes,
            block_size=find_optimal_log_block_size(),
        )
        bb.add(qrom)
        lte_bloq = LessThanEqual(self.num_bits_state_prep, self.num_bits_state_prep)
        # prepare uniform superposition over sigma
        sigma = bb.add(OnEach(Hadamard(), self.num_bits_state_prep), q=sigma)
        keep, sigma, less_than = bb.add(lte_bloq, x=keep, y=sigma, target=less_than)
        # TODO: Missing the CZ's for the sign bits
        keep, theta[1] = bb.add(ControlledBloq(ZGate), control=keep, q=theta[1])
        # swap the ind and alt_pqrs values
        # TODO: These swaps are inverted at zero Toffoli cost in the reference.
        # The method is to copy all values being swapped before they are swapped. Then
        # to invert the controlled swap, perform measurements on the swapped
        # values in the X basis. We can perform phase fixups using
        # controlled-phase operations, where the control is the control qubit
        # for the controlled swaps, and the targets are the copies of the
        # registers.
        less_than, alt_pqrs[0], p = bb.add(CSwap(n_n), ctrl=less_than, x=alt_pqrs[0], y=p)
        less_than, alt_pqrs[1], q = bb.add(CSwap(n_n), ctrl=less_than, x=alt_pqrs[0], y=q)
        less_than, alt_pqrs[2], r = bb.add(CSwap(n_n), ctrl=less_than, x=alt_pqrs[0], y=r)
        less_than, alt_pqrs[3], s = bb.add(CSwap(n_n), ctrl=less_than, x=alt_pqrs[0], y=s)
        # swap the 1b/2b alt values
        less_than, flag_1b[0], flag_1b[1] = bb.add(
            CSwap(1), ctrl=less_than, x=flag_1b[0], y=flag_1b[1]
        )
        # invert the comparator
        keep, sigma, less_than = bb.add(lte_bloq, x=keep, y=sigma, target=less_than)
        # prepare |+> states for symmetry swaps
        swap_pq = bb.add(Hadamard(), q=swap_pq)
        swap_rs = bb.add(Hadamard(), q=swap_rs)
        swap_pqrs = bb.add(Hadamard(), q=swap_pqrs)
        # Perform symmetry swaps
        swap_pq, p, q = bb.add(CSwap(n_n), xtrl=swap_pq, x=p, y=q)
        swap_rs, r, s = bb.add(CSwap(n_n), xtrl=swap_pq, x=r, y=s)
        swap_pqrs, p, r = bb.add(CSwap(n_n), xtrl=swap_pq, x=p, y=r)
        swap_pqrs, q, s = bb.add(CSwap(n_n), xtrl=swap_pq, x=q, y=s)
        return {
            'd': d,
            'p': p,
            'q': q,
            'r': r,
            's': s,
            'alpha': alpha,
            'beta': beta,
            'sigma': sigma,
            'rot_aa': rot_aa,
            'swap_pq': swap_pq,
            'swap_rs': swap_rs,
            'swap_pqrs': swap_pqrs,
            'alt_pqrs': alt_pqrs,
            'theta': theta,
            'keep': keep,
            'less_than': less_than,
            'flag_1b': flag_1b,
        }

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
                (BBPrepareUniformSuperposition(self.num_non_zero, self.num_bits_rot_aa), 1),
                qrom_cost,
            }
        swap_cost_state_prep = (CSwapApprox(num_bits_spat), 4 + 4)  # 2. pg 39
        ineq_cost_state_prep = (Toffoli(), (self.num_bits_state_prep + 1))  # 2. pg 39

        return {
            (BBPrepareUniformSuperposition(self.num_non_zero, self.num_bits_rot_aa), 1),
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
