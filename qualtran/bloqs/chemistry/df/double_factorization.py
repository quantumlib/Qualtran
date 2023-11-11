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
r"""Bloqs for the double-factorized chemistry Hamiltonian in second quantization.

Recall that for the single factorized Hamiltonian we have
$$
    H = \sum_{pq}T'_{pq} a_p^\dagger a_q + \frac{1}{2} \sum_l \left(\sum_{pq}
W_{pq}^{(l)} a_p^\dagger a_q\right)^2.
$$
One arrives at the double factorized (DF) Hamiltonian by further factorizing the
$W_{pq}^{(l)}$ terms as
$$
    W^{(l)}_{pq} = \sum_{k} U^{(l)}_{pk} f_k^{(l)} U^{(l)*}_{qk},
$$
so that
$$
    H = \sum_{pq}T'_{pq} a_p^\dagger a_q + \frac{1}{2} \sum_l U^{(l)}\left(\sum_{k}^{\Xi^{(l)}}
        f_k^{(l)} n_k\right)^2 U^{(l)\dagger}
$$
where $\Xi^{(l)} $ is the rank of second factorization.
"""
from functools import cached_property
from typing import Dict, Iterable, Set, TYPE_CHECKING

import cirq
import numpy as np
from attrs import frozen

from qualtran import Bloq, bloq_example, BloqBuilder, BloqDocSpec, Register, Signature, SoquetT
from qualtran.bloqs.basic_gates import CSwap, Toffoli
from qualtran.bloqs.chemistry.df.common_bitsize import get_num_bits_lxi
from qualtran.bloqs.chemistry.df.prepare import (
    InnerPrepareDoubleFactorization,
    OuterPrepareDoubleFactorization,
    OutputIndexedData,
)
from qualtran.bloqs.chemistry.df.select import ProgRotGateArray
from qualtran.bloqs.multi_control_multi_target_pauli import MultiControlPauli
from qualtran.bloqs.util_bloqs import ArbitraryClifford

if TYPE_CHECKING:
    from qualtran.resource_counting import SympySymbolAllocator


@frozen
class DoubleFactorizationOneBody(Bloq):
    r"""Block encoding of double factorization one-body Hamiltonian.

    Implements inner "half" of Fig. 15 in the reference. This block encoding is
    applied twice (with a reflection around the inner state preparation
    registers) to implement (roughly) the square of this one-body operator.

    Note succ_pq will be allocated as an ancilla during decomposition and it is not relected on.

    Args:
        num_aux: Dimension of auxiliary index for double factorized Hamiltonian. Call L in Ref[1].
        num_spin_orb: The number of spin orbitals. Typically called $N$.
        num_xi: Rank of second factorization. Full rank implies $\Xi$ = num_spin_orb // 2.
        num_bits_state_prep: The number of bits of precision for coherent alias
            sampling. Called $\aleph$ in the reference.
        num_bits_rot_aa_outer: Number of bits of precision for single qubit
            rotation for amplitude amplification. Called $b_r$ in the reference.
        num_bits_rot: Number of bits of precision for rotations for amplitude
            amplification in uniform state preparation. Called $\beth$ in Ref[1].
        adjoint: Whether this bloq is daggered or not. This affects the QROM cost.

    Registers:
        succ_l: control for success for outer state preparation.
        l_ne_zero: control for one-body part of Hamiltonian.
        xi: data register for number storing $\Xi^{(l)}$.
        p: Register for inner state preparation.
        rot_aa: A qubit to be rotated for amplitude amplification.
        spin: A single qubit register for spin superpositions.
        xi: Register for rank parameter.
        offset: Offset for p register.
        rot: Amplitude amplification angles for inner preparations.
        rotations: Rotations for basis rotations.
        sys: The system register.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://arxiv.org/abs/2011.03494)
    """
    num_aux: int
    num_spin_orb: int
    num_xi: int
    num_bits_state_prep: int
    num_bits_rot_aa: int = 8
    num_bits_rot: int = 24
    adjoint: bool = False

    @property
    def control_registers(self) -> Iterable[Register]:
        return (Register("succ_l", bitsize=1), Register("l_ne_zero", bitsize=1))

    @property
    def selection_registers(self) -> Iterable[Register]:
        return (
            Register("p", bitsize=(self.num_xi - 1).bit_length()),
            Register("rot_aa", bitsize=1),
            Register("spin", bitsize=1),
        )

    @property
    def junk_registers(self) -> Iterable[Register]:
        nlxi = get_num_bits_lxi(self.num_aux, self.num_xi, self.num_spin_orb)
        nxi = (self.num_xi - 1).bit_length()  # C14
        return (
            Register("xi", bitsize=nxi),
            Register("offset", bitsize=nlxi),
            Register("rot", bitsize=self.num_bits_rot_aa),
            Register("rotations", bitsize=(self.num_spin_orb // 2) * self.num_bits_rot),
        )

    @property
    def target_registers(self) -> Iterable[Register]:
        return (Register("sys", bitsize=self.num_spin_orb),)

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                *self.control_registers,
                *self.selection_registers,
                *self.junk_registers,
                *self.target_registers,
            ]
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        in_prep = InnerPrepareDoubleFactorization(
            num_aux=self.num_aux,
            num_spin_orb=self.num_spin_orb,
            num_xi=self.num_xi,
            num_bits_rot_aa=self.num_bits_rot_aa,
            num_bits_state_prep=self.num_bits_state_prep,
            adjoint=False,
        )
        in_prep_dag = InnerPrepareDoubleFactorization(
            num_aux=self.num_aux,
            num_spin_orb=self.num_spin_orb,
            num_xi=self.num_xi,
            num_bits_rot_aa=self.num_bits_rot_aa,
            num_bits_state_prep=self.num_bits_state_prep,
            adjoint=True,
        )
        rot = ProgRotGateArray(
            num_aux=self.num_aux,
            num_xi=self.num_xi,
            num_spin_orb=self.num_spin_orb,
            num_bits_rot=self.num_bits_rot,
            adjoint=False,
        )
        rot_dag = ProgRotGateArray(
            num_aux=self.num_aux,
            num_xi=self.num_xi,
            num_spin_orb=self.num_spin_orb,
            num_bits_rot=self.num_bits_rot,
            adjoint=True,
        )
        # 2*In-prep_l, addition, Rotations, 2*H, 2*SWAPS, subtraction
        return {
            (in_prep, 1),  # in-prep_l listing 3 page 52/53
            (in_prep_dag, 1),  # in_prep_l^dag
            (rot, 1),  # rotate into system basis  listing 4 pg 54
            (
                Toffoli(),
                1,
            ),  # apply CCZ first then CCCZ, the cost is 1 + 2 Toffolis (step 4e, and 7)
            (rot_dag, 1),  # Undo rotations
            (CSwap(self.num_spin_orb // 2), 2),  # Swaps for spins
            (ArbitraryClifford(n=1), 1),  # 2 Hadamards for spin superposition
        }


@frozen
class DoubleFactorizationBlockEncoding(Bloq):
    r"""Block encoding of double factorization Hamiltonian.

    Implements Fig. 15 in the reference.

    Args:
        num_spin_orb: The number of spin orbitals. Typically called N.
        num_aux: Dimension of auxiliary index for double factorized Hamiltonian. Called L in Ref[1].
        num_xi: Rank of second factorization. Full rank implies $Xi$ = num_spin_orb // 2.
        num_bits_state_prep: The number of bits of precision for coherent alias
            sampling. Called $\aleph$ in Ref[1]. We assume this is the same for
            both outer and inner state preparations.
        num_bits_rot: Number of bits of precision for rotations
            amplification in uniform state preparation. Called $\beth$ in Ref[1].
        num_bits_rot_aa_outer: Number of bits of precision for single qubit
            rotation for amplitude amplification in outer state preparation.
            Called $b_r$ in the reference. Keeping inner and outer separate for
            consistency with openfermion.
        num_bits_rot_aa_inner: Number of bits of precision for single qubit
            rotation for amplitude amplification in inner state preparation.
            Called $b_r$ in the reference.

    Registers:
        ctrl: Registers used to control application of Hamiltonian terms / preparation.
        l: Register for outer state preparation.
        p: Register for inner state preparation.
        rot_aa: A qubit to be rotated for amplitude amplification.
        spin: A single qubit register for spin superpositions.
        xi: Register for rank parameter.
        offset: Offset for p register.
        rot: Amplitude amplification angles for inner preparations.
        rotations: Rotations for basis rotations.
        sys: The system register.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            hypercontraction](https://arxiv.org/abs/2011.03494)
    """
    num_spin_orb: int
    num_aux: int
    num_xi: int
    num_bits_state_prep: int = 8
    num_bits_rot: int = 24
    num_bits_rot_aa_outer: int = 8
    num_bits_rot_aa_inner: int = 8

    @classmethod
    def build_from_coeffs(cls, one_body_ham, factorized_two_body_ham) -> 'DoubleFactorization':
        """Factory method to build double factorization block encoding given Hamiltonian inputs.

        Args:
            one_body_ham: One body hamiltonian ($T_{pq}$') matrix elements. (includes exchange terms).
            factorized_two_body_ham: One body hamiltonian ($W^{(l)}_{pq}$).

        Returns:
            Double factorized bloq with alt/keep values appropriately constructed.

        Refererences:
            [Even More Efficient Quantum Computations of Chemistry Through Tensor
                hypercontraction]
                (https://arxiv.org/abs/2011.03494). Eq. B7 pg 43.
        """
        assert len(one_body_ham.shape) == 2
        assert len(factorized_two_body_ham.shape) == 3
        raise NotImplementedError("Factory method not implemented yet.")

    @property
    def control_registers(self) -> Iterable[Register]:
        return [Register('ctrl', bitsize=1, shape=(4,))]

    @property
    def selection_registers(self) -> Iterable[Register]:
        return [
            Register("l", bitsize=self.num_aux.bit_length()),
            Register("p", bitsize=(self.num_xi - 1).bit_length()),
            Register("spin", bitsize=1),
            Register('rot_aa', bitsize=1),
        ]

    @property
    def junk_registers(self) -> Iterable[Register]:
        nlxi = get_num_bits_lxi(self.num_aux, self.num_xi, self.num_spin_orb)
        nxi = (self.num_xi - 1).bit_length()  # C14
        return (
            Register("xi", bitsize=nxi),
            Register("offset", bitsize=nlxi),
            Register("rot", bitsize=self.num_bits_rot_aa_inner),
            Register("rotations", bitsize=(self.num_spin_orb // 2) * self.num_bits_rot),
        )

    @property
    def target_registers(self) -> Iterable[Register]:
        return [Register("sys", bitsize=self.num_spin_orb)]

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                *self.control_registers,
                *self.selection_registers,
                *self.junk_registers,
                *self.target_registers,
            ]
        )

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        ctrl: SoquetT,
        l: SoquetT,
        p: SoquetT,
        spin: SoquetT,
        rot_aa: SoquetT,
        xi: SoquetT,
        offset: SoquetT,
        rot: SoquetT,
        rotations: SoquetT,
        sys: SoquetT,
    ) -> Dict[str, 'SoquetT']:
        succ_l, l_ne_zero, theta, succ_p = ctrl
        num_bits_xi = (self.num_xi - 1).bit_length()  # C14
        outer_prep = OuterPrepareDoubleFactorization(
            self.num_aux,
            num_bits_state_prep=self.num_bits_state_prep,
            num_bits_rot_aa=self.num_bits_rot_aa_outer,
        )
        l, succ_l = bb.add(outer_prep, l=l, succ_l=succ_l)
        in_l_data_l = OutputIndexedData(
            num_aux=self.num_aux,
            num_spin_orb=self.num_spin_orb,
            num_xi=self.num_xi,
            num_bits_rot_aa=self.num_bits_rot_aa_inner,
        )
        l, l_ne_zero, xi, rot, offset = bb.add(
            in_l_data_l, l=l, l_ne_zero=l_ne_zero, xi=xi, rot_data=rot, offset=offset
        )
        one_body = DoubleFactorizationOneBody(
            self.num_aux,
            self.num_spin_orb,
            self.num_xi,
            self.num_bits_state_prep,
            num_bits_rot_aa=self.num_bits_rot_aa_inner,
            num_bits_rot=self.num_bits_rot,
        )
        succ_l, l_ne_zero, p, rot_aa, spin, xi, offset, rot, rotations, sys = bb.add(
            one_body,
            succ_l=succ_l,
            l_ne_zero=l_ne_zero,
            p=p,
            rot_aa=rot_aa,
            spin=spin,
            xi=xi,
            offset=offset,
            rot=rot,
            rotations=rotations,
            sys=sys,
        )
        # The last ctrl is the 'target' register for the MCP gate.
        cvs = (1, 1) + (0,) * num_bits_xi
        mcp = MultiControlPauli(cvs, cirq.Z)
        ctrls = bb.join(np.concatenate([[succ_l, l_ne_zero], bb.split(p)]))
        ctrls, spin = bb.add(mcp, controls=ctrls, target=spin)
        ctrls = bb.split(ctrls)
        succ_l, l_ne_zero = ctrls[:2]
        p = bb.join(ctrls[2:])
        succ_l, l_ne_zero, p, rot_aa, spin, xi, offset, rot, rotations, sys = bb.add(
            one_body,
            succ_l=succ_l,
            l_ne_zero=l_ne_zero,
            p=p,
            rot_aa=rot_aa,
            spin=spin,
            xi=xi,
            offset=offset,
            rot=rot,
            rotations=rotations,
            sys=sys,
        )
        # ancilla for amplitude amplifcation
        in_l_data_l = OutputIndexedData(
            num_aux=self.num_aux,
            num_spin_orb=self.num_spin_orb,
            num_xi=self.num_xi,
            num_bits_rot_aa=self.num_bits_rot_aa_inner,
            adjoint=True,
        )
        l, l_ne_zero, xi, rot, offset = bb.add(
            in_l_data_l, l=l, l_ne_zero=l_ne_zero, xi=xi, rot_data=rot, offset=offset
        )
        # prepare_l^dag
        outer_prep = OuterPrepareDoubleFactorization(
            self.num_aux,
            num_bits_state_prep=self.num_bits_state_prep,
            num_bits_rot_aa=self.num_bits_rot_aa_outer,
            adjoint=True,
        )
        l, succ_l = bb.add(outer_prep, l=l, succ_l=succ_l)
        ctrl = succ_l, l_ne_zero, theta, succ_p
        return {
            'ctrl': ctrl,
            'l': l,
            'p': p,
            'spin': spin,
            'rot_aa': rot_aa,
            'xi': xi,
            'offset': offset,
            'rot': rot,
            'rotations': rotations,
            'sys': sys,
        }


@bloq_example
def _df_one_body() -> DoubleFactorizationOneBody:
    num_aux = 50
    num_bits_state_prep = 12
    num_bits_rot = 7
    num_spin_orb = 10
    num_aux = 50
    num_eig = num_spin_orb // 2
    df_one_body = DoubleFactorizationOneBody(
        num_aux=num_aux,
        num_spin_orb=num_spin_orb,
        num_xi=num_eig,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot=num_bits_rot,
        adjoint=False,
    )
    return df_one_body


@bloq_example
def _df_block_encoding() -> DoubleFactorizationBlockEncoding:
    num_spin_orb = 10
    num_aux = 50
    num_eig = num_spin_orb // 2
    num_bits_state_prep = 12
    num_bits_rot = 7
    df_block_encoding = DoubleFactorizationBlockEncoding(
        num_spin_orb=num_spin_orb,
        num_aux=num_aux,
        num_xi=num_eig,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa_outer=1,
        num_bits_rot_aa_inner=7,
        num_bits_rot=num_bits_rot,
    )
    return df_block_encoding


_DF_ONE_BODY = BloqDocSpec(
    bloq_cls=DoubleFactorizationOneBody,
    import_line='from qualtran.bloqs.chemistry.df.double_factorization import DoubleFactorizationOneBody',
    examples=(_df_one_body,),
)

_DF_BLOCK_ENCODING = BloqDocSpec(
    bloq_cls=DoubleFactorizationBlockEncoding,
    import_line='from qualtran.bloqs.chemistry.df.double_factorization import DoubleFactorizationBlockEncoding',
    examples=(_df_block_encoding,),
)
