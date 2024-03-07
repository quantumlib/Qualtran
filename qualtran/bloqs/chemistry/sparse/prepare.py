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

import itertools
from functools import cached_property
from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import (
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    BoundedQUInt,
    QAny,
    QBit,
    Register,
    SoquetT,
)
from qualtran.bloqs.arithmetic.comparison import LessThanEqual
from qualtran.bloqs.basic_gates import CSwap, Hadamard, Toffoli
from qualtran.bloqs.basic_gates.on_each import OnEach
from qualtran.bloqs.basic_gates.z_basis import ZGate
from qualtran.bloqs.select_and_prepare import PrepareOracle
from qualtran.bloqs.select_swap_qrom import find_optimal_log_block_size, SelectSwapQROM
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
    PrepareUniformSuperposition,
)
from qualtran.linalg.lcu_util import preprocess_lcu_coefficients_for_reversible_sampling

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


def get_sparse_inputs_from_integrals(
    tpq_prime: NDArray[np.float64], eris: NDArray[np.float64], drop_element_thresh: float = 0.0
):
    r"""Simple way to build sparse integrals from usual chemistry integrals.

    Extract permutational-unique elements, and then truncate based on drop_element_thresh.

    Args:
        tpq_prime: The modified one-body matrix elements.
        eris: The 4-index electron repulsion integral (ERI) tensor in chemists notation (pq|rs).
        drop_element_thresh: Threshold for considering an ERI element as zero.
            Default 0, i.e. don't threshold the elements.

    Returns:
        integrals: Sparsified, symmetry inequivalent integrals.
        indicies: corresponding indices of the non-zero matrix elements.

    References:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            hypercontraction](https://arxiv.org/abs/2011.03494) Eq. A19 page 40.
    """
    assert len(tpq_prime.shape) == 2, "hcore should be a matrix"
    assert len(eris.shape) == 4, "eris should be 4-index tensor"
    num_spat = tpq_prime.shape[-1]
    tril = np.tril_indices(num_spat)
    # we don't sparsify one-body, but just take the lower triangular part
    tpq_sparse = tpq_prime[tril]
    tpq_indx = np.array([(ix[0], ix[1], 0, 0) for ix in zip(*tril)])
    eris_eight = []
    pqrs_indx = []

    # "_indx" to avoid positional arguments out of order pylint error.
    def _add(p_indx: int, q_indx: int, r_indx: int, s_indx: int):
        nonlocal eris_eight, pqrs_indx
        eris_eight.append(eris[p_indx, q_indx, r_indx, s_indx])
        pqrs_indx.append((p_indx, q_indx, r_indx, s_indx))

    # ignoring scaling factors for the moment.
    for p, q, r, s in itertools.combinations(range(num_spat), 4):
        _add(p, q, r, s)
        _add(p, r, q, s)
        _add(p, s, r, q)
    for p, q, r in itertools.combinations(range(num_spat), 3):
        _add(p, p, q, r)
        _add(p, q, p, r)
        _add(q, q, p, r)
        _add(q, r, p, q)
        _add(r, r, p, q)
        _add(r, p, q, r)
    for p, q in itertools.combinations(range(num_spat), 2):
        _add(p, p, q, q)
        _add(p, q, p, q)
        _add(p, p, p, q)
        _add(q, q, q, p)
    for p in range(num_spat):
        _add(p, p, p, p)
    eris_eight = np.array(eris_eight)
    pqrs_indx = np.array(pqrs_indx)
    keep_indx = np.where(np.abs(eris_eight) > drop_element_thresh)
    eris_eight = eris_eight[keep_indx]
    pqrs_indx = pqrs_indx[keep_indx[0]]
    return np.concatenate((tpq_indx, pqrs_indx)), np.concatenate((tpq_sparse, eris_eight))


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
            preparation. Default 8.
        adjoint: Whether we are apply PREPARE or PREPARE^dag
        qroam_block_size: qroam blocking factor.

    Registers:
        d: the register indexing non-zero matrix elements.
        pqrs: the register to store the spatial orbital index.
        sigma: the register prepared for alias sampling.
        alpha: spin for (pq) indicies.
        beta: spin for (rs) indicies.
        rot_aa: the qubit rotated for amplitude amplification.
        swap_pq: a |+> state to restore the symmetries of the p and q indices.
        swap_rs: a |+> state to restore the symmetries of the r and s indices.
        swap_pqrs: a |+> state to restore the symmetries of between (pq) and (rs).
        theta: sign qubit.
        alt_pqrs: the register to store the alternate values for the spatial orbital indices.
        theta: A two qubit register for the sign bit and it's alternate value.
        keep: The register containing the keep values for alias sampling.
        less_than: A single qubit for the result of the inequality test during alias sampling.
        flag_1b: a single qubit register indicating whether to apply only the one-body SELECT.
        alt_flag_1b: alternate value for flag_1b

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            hypercontraction](https://arxiv.org/abs/2011.03494) Eq. A11.
    """
    num_spin_orb: int
    num_non_zero: int
    num_bits_state_prep: int
    alt_pqrs: Tuple[int, ...] = field(repr=False)
    alt_theta: Tuple[int, ...] = field(repr=False)
    alt_one_body: Tuple[int, ...] = field(repr=False)
    ind_pqrs: Tuple[int, ...] = field(repr=False)
    theta: Tuple[int, ...] = field(repr=False)
    one_body: Tuple[int, ...] = field(repr=False)
    keep: Tuple[int, ...] = field(repr=False)
    num_bits_rot_aa: int = 8
    adjoint: bool = False
    qroam_block_size: Optional[int] = None

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        # issue here in that pqrs should not be reflected on.
        # See: https://github.com/quantumlib/Qualtran/issues/549
        return (
            Register(
                "d",
                BoundedQUInt(
                    bitsize=(self.num_non_zero - 1).bit_length(), iteration_length=self.num_non_zero
                ),
            ),
            Register(
                "p",
                BoundedQUInt(
                    bitsize=(self.num_spin_orb // 2 - 1).bit_length(),
                    iteration_length=self.num_spin_orb // 2,
                ),
            ),
            Register(
                "q",
                BoundedQUInt(
                    bitsize=(self.num_spin_orb // 2 - 1).bit_length(),
                    iteration_length=self.num_spin_orb // 2,
                ),
            ),
            Register(
                "r",
                BoundedQUInt(
                    bitsize=(self.num_spin_orb // 2 - 1).bit_length(),
                    iteration_length=self.num_spin_orb // 2,
                ),
            ),
            Register(
                "s",
                BoundedQUInt(
                    bitsize=(self.num_spin_orb // 2 - 1).bit_length(),
                    iteration_length=self.num_spin_orb // 2,
                ),
            ),
            Register("sigma", BoundedQUInt(self.num_bits_state_prep)),
            Register("alpha", BoundedQUInt(1)),
            Register("beta", BoundedQUInt(1)),
            Register("rot_aa", BoundedQUInt(1)),
            Register("swap_pq", BoundedQUInt(1)),
            Register("swap_rs", BoundedQUInt(1)),
            Register("swap_pqrs", BoundedQUInt(1)),
            Register("flag_1b", BoundedQUInt(1)),
        )

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return (
            Register(
                'alt_pqrs', QAny(bitsize=(self.num_spin_orb // 2 - 1).bit_length()), shape=(4,)
            ),
            Register('theta', QBit(), shape=(2,)),
            Register('keep', QAny(bitsize=self.num_bits_state_prep)),
            Register("less_than", QBit()),
            Register("alt_flag_1b", QBit()),
        )

    @classmethod
    def from_hamiltonian_coeffs(
        cls,
        num_spin_orb: int,
        tpq_prime: NDArray[np.float64],
        eris: NDArray[np.float64],
        num_bits_state_prep: int = 8,
        num_bits_rot_aa: int = 8,
        drop_element_thresh: float = 0.0,
        qroam_block_size: Optional[int] = None,
    ) -> 'PrepareSparse':
        r"""Factory method to build PrepareSparse from Hamiltonian coefficients.

        Args:
            num_spin_orb: The number of spin orbitals.
            tpq_prime: The modified one-body integrals.
            eris: Two electron integrals in chemist's notation i.e. (pq|rs).
            num_bits_state_prep: The number of bits for the state prepared during alias sampling.
            num_bits_rot_aa: The number of bits of precision for the single-qubit
                rotation for amplitude amplification during the uniform state
                preparation. Default 8.
            drop_element_thresh: Threshold for considering an ERI element as zero.
                Default 0, i.e. don't threshold the elements.
            qroam_block_size: Block size for qroam (called $k$ in the reference).

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
        indicies, integrals = get_sparse_inputs_from_integrals(
            tpq_prime, eris, drop_element_thresh=drop_element_thresh
        )
        num_non_zero = len(integrals)
        alt, keep, mu = preprocess_lcu_coefficients_for_reversible_sampling(
            np.abs(integrals), epsilon=2**-num_bits_state_prep / num_non_zero
        )
        theta = (1 - np.sign(integrals)) // 2
        num_lt = num_spin_orb // 2 * (num_spin_orb // 2 + 1)
        one_body = np.array([0] * num_lt + [1] * len(integrals[num_lt:]))
        alt_pqrs = indicies[alt]
        alt_theta = theta[alt]
        alt_one_body = one_body[alt]
        return PrepareSparse(
            num_spin_orb,
            num_non_zero,
            num_bits_state_prep,
            tuple(tuple([int(_) for _ in x]) for x in alt_pqrs.T),
            tuple([int(_) for _ in alt_theta]),
            tuple([int(_) for _ in alt_one_body]),
            tuple(tuple([int(_) for _ in x]) for x in indicies.T),
            tuple([int(_) for _ in theta]),
            tuple([int(_) for _ in one_body]),
            tuple(keep),
            num_bits_rot_aa=num_bits_rot_aa,
            qroam_block_size=qroam_block_size,
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
        flag_1b: 'SoquetT',
        alt_pqrs: 'SoquetT',
        theta: 'SoquetT',
        keep: 'SoquetT',
        less_than: 'SoquetT',
        alt_flag_1b: 'SoquetT',
    ) -> Dict[str, 'SoquetT']:
        # 1. Prepare \sum_d |d\rangle
        d = bb.add(PrepareUniformSuperposition(self.num_non_zero), target=d)
        # 2. QROM the ind_d alt_d values
        n_n = (self.num_spin_orb // 2 - 1).bit_length()
        target_bitsizes = (
            (n_n,) * 4 + (1,) * 2 + (n_n,) * 4 + (1,) * 2 + (self.num_bits_state_prep,)
        )
        if self.qroam_block_size is None:
            block_size = 2 ** find_optimal_log_block_size(self.num_non_zero, sum(target_bitsizes))
        else:
            block_size = self.qroam_block_size
        qrom = SelectSwapQROM(
            self.ind_pqrs[0],
            self.ind_pqrs[1],
            self.ind_pqrs[2],
            self.ind_pqrs[3],
            self.theta,
            self.one_body,
            self.alt_pqrs[0],
            self.alt_pqrs[1],
            self.alt_pqrs[2],
            self.alt_pqrs[3],
            self.alt_theta,
            self.alt_one_body,
            self.keep,
            target_bitsizes=target_bitsizes,
            block_size=block_size,
        )
        (
            d,
            p,
            q,
            r,
            s,
            theta[0],
            flag_1b,
            alt_pqrs[0],
            alt_pqrs[1],
            alt_pqrs[2],
            alt_pqrs[3],
            theta[1],
            alt_flag_1b,
            keep,
        ) = bb.add(
            qrom,
            selection=d,
            target0_=p,
            target1_=q,
            target2_=r,
            target3_=s,
            target4_=theta[0],
            target5_=flag_1b,
            target6_=alt_pqrs[0],
            target7_=alt_pqrs[1],
            target8_=alt_pqrs[2],
            target9_=alt_pqrs[3],
            target10_=theta[1],
            target11_=alt_flag_1b,
            target12_=keep,
        )
        lte_bloq = LessThanEqual(self.num_bits_state_prep, self.num_bits_state_prep)
        # prepare uniform superposition over sigma
        sigma = bb.add(OnEach(self.num_bits_state_prep, Hadamard()), q=sigma)
        keep, sigma, less_than = bb.add(lte_bloq, x=keep, y=sigma, target=less_than)
        less_than, theta[1] = bb.add(ZGate().controlled(), ctrl=less_than, q=theta[1])
        # TODO: This should be off control
        less_than, theta[0] = bb.add(ZGate().controlled(), ctrl=less_than, q=theta[0])
        # swap the ind and alt_pqrs values
        # TODO: These swaps are inverted at zero Toffoli cost in the reference.
        # The method is to copy all values being swapped before they are swapped. Then
        # to invert the controlled swap, perform measurements on the swapped
        # values in the X basis. We can perform phase fixups using
        # controlled-phase operations, where the control is the control qubit
        # for the controlled swaps, and the targets are the copies of the
        # registers.
        less_than, alt_pqrs[0], p = bb.add(CSwap(n_n), ctrl=less_than, x=alt_pqrs[0], y=p)
        less_than, alt_pqrs[1], q = bb.add(CSwap(n_n), ctrl=less_than, x=alt_pqrs[1], y=q)
        less_than, alt_pqrs[2], r = bb.add(CSwap(n_n), ctrl=less_than, x=alt_pqrs[2], y=r)
        less_than, alt_pqrs[3], s = bb.add(CSwap(n_n), ctrl=less_than, x=alt_pqrs[3], y=s)
        # swap the 1b/2b alt values
        # less_than, flag_1b, alt_flag_1b = bb.add(CSwap(1), ctrl=less_than, x=flag_1b, y=alt_flag_1b)
        # invert the comparator
        keep, sigma, less_than = bb.add(lte_bloq, x=keep, y=sigma, target=less_than)
        # prepare |+> states for symmetry swaps
        swap_pq = bb.add(Hadamard(), q=swap_pq)
        swap_rs = bb.add(Hadamard(), q=swap_rs)
        swap_pqrs = bb.add(Hadamard(), q=swap_pqrs)
        # Perform symmetry swaps
        swap_pqrs, p, r = bb.add(CSwap(n_n), ctrl=swap_pqrs, x=p, y=r)
        swap_pqrs, q, s = bb.add(CSwap(n_n), ctrl=swap_pqrs, x=q, y=s)
        swap_pq, p, q = bb.add(CSwap(n_n), ctrl=swap_pq, x=p, y=q)
        swap_rs, r, s = bb.add(CSwap(n_n), ctrl=swap_rs, x=r, y=s)
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
            'flag_1b': flag_1b,
            'alt_pqrs': alt_pqrs,
            'theta': theta,
            'keep': keep,
            'less_than': less_than,
            'alt_flag_1b': alt_flag_1b,
        }

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        num_bits_spat = (self.num_spin_orb // 2 - 1).bit_length()
        if self.qroam_block_size is None:
            target_bitsizes = (
                (num_bits_spat,) * 4
                + (1,) * 2
                + (num_bits_spat,) * 4
                + (1,) * 2
                + (self.num_bits_state_prep,)
            )
            block_size = 2 ** find_optimal_log_block_size(self.num_non_zero, sum(target_bitsizes))
        else:
            block_size = self.qroam_block_size
        if self.adjoint:
            num_toff_qrom = int(np.ceil(self.num_non_zero / block_size)) + block_size  # A15
        else:
            output_size = self.num_bits_state_prep + 8 * num_bits_spat + 4
            num_toff_qrom = int(np.ceil(self.num_non_zero / block_size)) + output_size * (
                block_size - 1
            )  # A14
        qrom_cost = (Toffoli(), num_toff_qrom)
        if self.adjoint:
            return {(PrepareUniformSuperposition(self.num_non_zero), 1), qrom_cost}
        swap_cost_state_prep = (CSwap(num_bits_spat), 4 + 4)  # 2. pg 39
        ineq_cost_state_prep = (Toffoli(), (self.num_bits_state_prep + 1))  # 2. pg 39

        return {
            (PrepareUniformSuperposition(self.num_non_zero), 1),
            qrom_cost,
            swap_cost_state_prep,
            ineq_cost_state_prep,
        }


@bloq_example
def _prep_sparse() -> PrepareSparse:
    num_spin_orb = 4
    tpq = np.random.random((num_spin_orb // 2, num_spin_orb // 2))
    tpq = 0.5 * (tpq + tpq.T)
    eris = np.random.random((num_spin_orb // 2,) * 4)
    eris += np.transpose(eris, (0, 1, 3, 2))
    eris += np.transpose(eris, (1, 0, 2, 3))
    eris += np.transpose(eris, (2, 3, 0, 1))
    prep_sparse = PrepareSparse.from_hamiltonian_coeffs(num_spin_orb, tpq, eris)
    return prep_sparse


_SPARSE_PREPARE = BloqDocSpec(
    bloq_cls=PrepareSparse,
    import_line='from qualtran.bloqs.chemistry.sparse.prepare import PrepareSparse',
    examples=(_prep_sparse,),
)
