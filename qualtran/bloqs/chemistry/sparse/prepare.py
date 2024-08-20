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

import attrs
import numpy as np
from numpy.typing import NDArray

from qualtran import (
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    BoundedQUInt,
    CtrlSpec,
    QAny,
    QBit,
    QUInt,
    Register,
    Side,
    SoquetT,
)
from qualtran.bloqs.arithmetic.comparison import LessThanEqual
from qualtran.bloqs.basic_gates import CSwap, Hadamard
from qualtran.bloqs.basic_gates.on_each import OnEach
from qualtran.bloqs.basic_gates.z_basis import CZ, ZGate
from qualtran.bloqs.data_loading.qroam_clean import (
    get_optimal_log_block_size_clean_ancilla,
    QROAMClean,
)
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
    PrepareUniformSuperposition,
)
from qualtran.linalg.lcu_util import preprocess_probabilities_for_reversible_sampling
from qualtran.symbolics import SymbolicFloat
from qualtran.symbolics.math_funcs import ceil, log2

if TYPE_CHECKING:
    from qualtran import Bloq
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
    eris_eight_np = np.array(eris_eight)
    pqrs_indx_np = np.array(pqrs_indx)
    keep_indx = np.where(np.abs(eris_eight_np) > drop_element_thresh)
    eris_eight_np = eris_eight_np[keep_indx]
    pqrs_indx_np = pqrs_indx_np[keep_indx[0]]
    return np.concatenate((tpq_indx, pqrs_indx_np)), np.concatenate((tpq_sparse, eris_eight_np))


@attrs.frozen
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
        is_adjoint: Whether we are apply PREPARE or PREPARE^dag
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
    alt_pqrs: Tuple[Tuple[int, ...], ...] = attrs.field(repr=False)
    alt_theta: Tuple[int, ...] = attrs.field(repr=False)
    alt_one_body: Tuple[int, ...] = attrs.field(repr=False)
    ind_pqrs: Tuple[Tuple[int, ...], ...] = attrs.field(repr=False)
    theta: Tuple[int, ...] = attrs.field(repr=False)
    one_body: Tuple[int, ...] = attrs.field(repr=False)
    keep: Tuple[int, ...] = attrs.field(repr=False)
    num_bits_rot_aa: int = 8
    is_adjoint: bool = False
    sum_of_l1_coeffs: SymbolicFloat = 0.0
    qroam_block_size: int = 1

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
            Register("sigma", BoundedQUInt(self.num_bits_state_prep)),
            Register("alpha", BoundedQUInt(1)),
            Register("beta", BoundedQUInt(1)),
            Register("rot_aa", BoundedQUInt(1)),
            Register("swap_pq", BoundedQUInt(1)),
            Register("swap_rs", BoundedQUInt(1)),
            Register("swap_pqrs", BoundedQUInt(1)),
        )

    def adjoint(self) -> 'PrepareSparse':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)

    @cached_property
    def _side(self) -> Side:
        return Side.RIGHT if not self.is_adjoint else Side.LEFT

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        extra_junk = (Register("less_than", QBit()),)
        return extra_junk + self.qroam_junk_regs + self.qroam_extra_junk_regs

    @cached_property
    def qroam_junk_regs(self) -> Tuple[Register, ...]:
        """Target registers for QROAMClean."""
        bs = (self.num_spin_orb // 2 - 1).bit_length()
        return (
            Register("p", QUInt(bitsize=bs), side=self._side),
            Register("q", QUInt(bitsize=bs), side=self._side),
            Register("r", QUInt(bitsize=bs), side=self._side),
            Register("s", QUInt(bitsize=bs), side=self._side),
            Register('theta', QBit(), side=self._side),
            Register("flag_1b", QBit(), side=self._side),
            Register('alt_p', QAny(bitsize=bs), side=self._side),
            Register('alt_q', QAny(bitsize=bs), side=self._side),
            Register('alt_r', QAny(bitsize=bs), side=self._side),
            Register('alt_s', QAny(bitsize=bs), side=self._side),
            Register('alt_theta', QBit(), side=self._side),
            Register("alt_flag_1b", QBit(), side=self._side),
            Register('keep', QAny(bitsize=self.num_bits_state_prep), side=self._side),
        )

    @cached_property
    def qroam_extra_junk_regs(self) -> Tuple[Register, ...]:
        """Extra junk registers required for QROAMClean."""
        return tuple(
            Register(
                name=f'jnk_{reg.name}',
                dtype=reg.dtype,
                shape=reg.shape + (self.qroam_block_size - 1,),
                side=self._side,
            )
            for reg in self.qroam_junk_regs
        )

    @property
    def l1_norm_of_coeffs(self) -> SymbolicFloat:
        return self.sum_of_l1_coeffs

    @cached_property
    def num_bits_spat_orb(self) -> int:
        return (self.num_spin_orb // 2 - 1).bit_length()

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
            [Even More Efficient Quantum Computations of Chemistry Through Tensor hypercontraction](https://arxiv.org/abs/2011.03494)
            Eq. A11.

            [Qubitization of Arbitrary Basis Quantum Chemistry Leveraging Sparsity and Low Rank Factorization](https://quantum-journal.org/papers/q-2019-12-02-208/)
            Sec 5 page 15
        """
        indicies, integrals = get_sparse_inputs_from_integrals(
            tpq_prime, eris, drop_element_thresh=drop_element_thresh
        )
        num_non_zero = len(integrals)
        alt, keep, _ = preprocess_probabilities_for_reversible_sampling(
            np.abs(integrals), sub_bit_precision=num_bits_state_prep
        )
        theta = (1 - np.sign(integrals)) // 2
        num_lt = num_spin_orb // 2 * (num_spin_orb // 2 + 1)
        one_body = np.array([0] * num_lt + [1] * len(integrals[num_lt:]))
        alt_pqrs = indicies[alt]
        alt_theta = theta[alt]
        alt_one_body = one_body[alt]
        if qroam_block_size is None:
            n_n = (num_spin_orb // 2 - 1).bit_length()
            target_bitsizes = (n_n,) * 4 + (1,) * 2 + (n_n,) * 4 + (1,) * 2 + (num_bits_state_prep,)
            log_block_sizes = get_optimal_log_block_size_clean_ancilla(
                num_non_zero, sum(target_bitsizes)
            )
        else:
            log_block_sizes = ceil(log2(qroam_block_size))
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
            sum_of_l1_coeffs=np.sum(np.abs(tpq_prime)) + 0.5 * np.sum(np.abs(eris)),
            qroam_block_size=2**log_block_sizes,
        )

    def build_qrom_bloq(self) -> 'Bloq':
        n_n = self.num_bits_spat_orb
        target_bitsizes = (
            (n_n,) * 4 + (1,) * 2 + (n_n,) * 4 + (1,) * 2 + (self.num_bits_state_prep,)
        )
        log_block_sizes = ceil(log2(self.qroam_block_size))
        qrom = QROAMClean.build_from_data(
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
            log_block_sizes=log_block_sizes,
        )
        if self.is_adjoint:
            return qrom.adjoint()
        return qrom

    def add_qrom(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        qrom = self.build_qrom_bloq()
        # The qroam_junk_regs won't be present initially when building the
        # composite bloq as they're RIGHT registers.
        if soqs.get(self.qroam_junk_regs[0].name) is not None:
            # doing the adjoint prepare
            # Need to merge target and junk target LEFT registers which will be eaten by QROAM^
            qroam_in_soqs = {'selection': soqs['d']}
            for ireg, (reg, jnk_reg) in enumerate(
                zip(self.qroam_junk_regs, self.qroam_extra_junk_regs)
            ):
                soq = soqs.pop(reg.name)
                jnk_soq = soqs.pop(jnk_reg.name)
                qroam_in_soqs |= {
                    f'target{ireg}_': np.concatenate(
                        [[soq], jnk_soq]
                    )  # merge target and junk_target
                }
            qroam_out_soqs = bb.add_d(qrom, **qroam_in_soqs)
            return soqs | {'d': qroam_out_soqs.pop('selection')}
        else:
            # doing prepare
            qroam_out_soqs = bb.add_d(qrom, selection=soqs['d'])
            out_soqs: Dict[str, 'SoquetT'] = {'d': qroam_out_soqs.pop('selection')}
            # map output soqs to Prepare junk registers names
            out_soqs |= {
                reg.name: qroam_out_soqs.pop(f'target{i}_')
                for (i, reg) in enumerate(self.qroam_junk_regs)
            }
            out_soqs |= {
                reg.name: qroam_out_soqs.pop(f'junk_target{i}_')
                for (i, reg) in enumerate(self.qroam_extra_junk_regs)
            }
            return soqs | out_soqs

    def _build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        n_n = self.num_bits_spat_orb
        # 1. Prepare \sum_d |d\rangle
        soqs['d'] = bb.add(PrepareUniformSuperposition(self.num_non_zero), target=soqs['d'])
        # 2. QROM the ind_d alt_d values
        soqs |= self.add_qrom(bb, **soqs)
        # prepare uniform superposition over sigma
        soqs['sigma'] = bb.add(OnEach(self.num_bits_state_prep, Hadamard()), q=soqs['sigma'])
        # Inequality test for alias sampling
        lte_bloq = LessThanEqual(self.num_bits_state_prep, self.num_bits_state_prep)
        soqs['keep'], soqs['sigma'], soqs['less_than'] = bb.add(
            lte_bloq, x=soqs['keep'], y=soqs['sigma'], target=soqs['less_than']
        )
        soqs['less_than'], soqs['alt_theta'] = bb.add(
            CZ(), q1=soqs['less_than'], q2=soqs['alt_theta']
        )
        ctrl_spec = CtrlSpec(QBit(), 0b0)
        soqs['less_than'], soqs['theta'] = bb.add(
            ZGate().controlled(ctrl_spec), ctrl=soqs['less_than'], q=soqs['theta']
        )
        # swap the ind and alt_pqrs values
        # TODO: These swaps are inverted at zero Toffoli cost in the reference.
        # The method is to copy all values being swapped before they are swapped. Then
        # to invert the controlled swap, perform measurements on the swapped
        # values in the X basis. We can perform phase fixups using
        # controlled-phase operations, where the control is the control qubit
        # for the controlled swaps, and the targets are the copies of the
        # registers.
        soqs['less_than'], soqs['alt_p'], soqs['p'] = bb.add(
            CSwap(n_n), ctrl=soqs['less_than'], x=soqs['alt_p'], y=soqs['p']
        )
        soqs['less_than'], soqs['alt_q'], soqs['q'] = bb.add(
            CSwap(n_n), ctrl=soqs['less_than'], x=soqs['alt_q'], y=soqs['q']
        )
        soqs['less_than'], soqs['alt_r'], soqs['r'] = bb.add(
            CSwap(n_n), ctrl=soqs['less_than'], x=soqs['alt_r'], y=soqs['r']
        )
        soqs['less_than'], soqs['alt_s'], soqs['s'] = bb.add(
            CSwap(n_n), ctrl=soqs['less_than'], x=soqs['alt_s'], y=soqs['s']
        )
        # swap the 1b/2b alt values
        soqs['less_than'], soqs['flag_1b'], soqs['alt_flag_1b'] = bb.add(
            CSwap(1), ctrl=soqs['less_than'], x=soqs['flag_1b'], y=soqs['alt_flag_1b']
        )
        # prepare |+> states for symmetry swaps
        soqs['swap_pq'] = bb.add(Hadamard(), q=soqs['swap_pq'])
        soqs['swap_rs'] = bb.add(Hadamard(), q=soqs['swap_rs'])
        soqs['swap_pqrs'] = bb.add(Hadamard(), q=soqs['swap_pqrs'])
        # Perform symmetry swaps
        soqs['swap_pqrs'], soqs['p'], soqs['r'] = bb.add(
            CSwap(n_n), ctrl=soqs['swap_pqrs'], x=soqs['p'], y=soqs['r']
        )
        soqs['swap_pqrs'], soqs['q'], soqs['s'] = bb.add(
            CSwap(n_n), ctrl=soqs['swap_pqrs'], x=soqs['q'], y=soqs['s']
        )
        soqs['swap_pq'], soqs['p'], soqs['q'] = bb.add(
            CSwap(n_n), ctrl=soqs['swap_pq'], x=soqs['p'], y=soqs['q']
        )
        soqs['swap_rs'], soqs['r'], soqs['s'] = bb.add(
            CSwap(n_n), ctrl=soqs['swap_rs'], x=soqs['r'], y=soqs['s']
        )
        return soqs

    def _build_composite_bloq_adj(
        self, bb: 'BloqBuilder', **soqs: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        n_n = self.num_bits_spat_orb
        soqs['swap_rs'], soqs['r'], soqs['s'] = bb.add(
            CSwap(n_n), ctrl=soqs['swap_rs'], x=soqs['r'], y=soqs['s']
        )
        soqs['swap_pq'], soqs['p'], soqs['q'] = bb.add(
            CSwap(n_n), ctrl=soqs['swap_pq'], x=soqs['p'], y=soqs['q']
        )
        soqs['swap_pqrs'], soqs['q'], soqs['s'] = bb.add(
            CSwap(n_n), ctrl=soqs['swap_pqrs'], x=soqs['q'], y=soqs['s']
        )
        soqs['swap_pqrs'], soqs['p'], soqs['r'] = bb.add(
            CSwap(n_n), ctrl=soqs['swap_pqrs'], x=soqs['p'], y=soqs['r']
        )
        soqs['swap_pqrs'] = bb.add(Hadamard(), q=soqs['swap_pqrs'])
        soqs['swap_rs'] = bb.add(Hadamard(), q=soqs['swap_rs'])
        soqs['swap_pq'] = bb.add(Hadamard(), q=soqs['swap_pq'])
        soqs['less_than'], soqs['flag_1b'], soqs['alt_flag_1b'] = bb.add(
            CSwap(1), ctrl=soqs['less_than'], x=soqs['flag_1b'], y=soqs['alt_flag_1b']
        )
        soqs['less_than'], soqs['alt_s'], soqs['s'] = bb.add(
            CSwap(n_n), ctrl=soqs['less_than'], x=soqs['alt_s'], y=soqs['s']
        )
        soqs['less_than'], soqs['alt_r'], soqs['r'] = bb.add(
            CSwap(n_n), ctrl=soqs['less_than'], x=soqs['alt_r'], y=soqs['r']
        )
        soqs['less_than'], soqs['alt_q'], soqs['q'] = bb.add(
            CSwap(n_n), ctrl=soqs['less_than'], x=soqs['alt_q'], y=soqs['q']
        )
        soqs['less_than'], soqs['alt_p'], soqs['p'] = bb.add(
            CSwap(n_n), ctrl=soqs['less_than'], x=soqs['alt_p'], y=soqs['p']
        )
        ctrl_spec = CtrlSpec(QBit(), 0b0)
        soqs['less_than'], soqs['theta'] = bb.add(
            ZGate().controlled(ctrl_spec), ctrl=soqs['less_than'], q=soqs['theta']
        )
        soqs['less_than'], soqs['alt_theta'] = bb.add(
            CZ(), q1=soqs['less_than'], q2=soqs['alt_theta']
        )
        # Inequality test for alias sampling
        lte_bloq = LessThanEqual(self.num_bits_state_prep, self.num_bits_state_prep)
        soqs['keep'], soqs['sigma'], soqs['less_than'] = bb.add(
            lte_bloq, x=soqs['keep'], y=soqs['sigma'], target=soqs['less_than']
        )
        # prepare uniform superposition over sigma
        soqs['sigma'] = bb.add(OnEach(self.num_bits_state_prep, Hadamard()), q=soqs['sigma'])
        # 2. QROM the ind_d alt_d values
        soqs = self.add_qrom(bb, **soqs)
        # 1. Prepare \sum_d |d\rangle
        soqs['d'] = bb.add(
            PrepareUniformSuperposition(self.num_non_zero).adjoint(), target=soqs['d']
        )
        return soqs

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        if self.is_adjoint:
            return self._build_composite_bloq_adj(bb, **soqs)
        else:
            return self._build_composite_bloq(bb, **soqs)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {
            (PrepareUniformSuperposition(self.num_non_zero), 1),
            (self.build_qrom_bloq(), 1),
            (OnEach(self.num_bits_state_prep, Hadamard()), 1),
            (Hadamard(), 3),
            (CSwap(1), 1),
            (CSwap(self.num_bits_spat_orb), 4 + 4),
            (LessThanEqual(self.num_bits_state_prep, self.num_bits_state_prep), 1),
        }


@bloq_example
def _prep_sparse() -> PrepareSparse:
    from qualtran.bloqs.chemistry.sparse.prepare_test import build_random_test_integrals

    num_spin_orb = 6
    tpq, eris = build_random_test_integrals(num_spin_orb // 2)
    prep_sparse = PrepareSparse.from_hamiltonian_coeffs(
        num_spin_orb, tpq, eris, num_bits_state_prep=4, qroam_block_size=2
    )
    return prep_sparse


_SPARSE_PREPARE = BloqDocSpec(bloq_cls=PrepareSparse, examples=(_prep_sparse,))
