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

import numpy as np
from attrs import frozen
from numpy.typing import NDArray

from qualtran import Bloq, bloq_example, BloqBuilder, BloqDocSpec, Register, Signature, SoquetT
from qualtran.bloqs.basic_gates import CSwap, Hadamard, Toffoli
from qualtran.bloqs.chemistry.black_boxes import ApplyControlledZs
from qualtran.bloqs.chemistry.df.prepare import (
    InnerPrepareDoubleFactorization,
    OuterPrepareDoubleFactorization,
    OutputIndexedData,
)
from qualtran.bloqs.chemistry.df.select_bloq import ProgRotGateArray
from qualtran.bloqs.reflection import Reflection
from qualtran.bloqs.util_bloqs import ArbitraryClifford

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class DoubleFactorizationOneBody(Bloq):
    r"""Block encoding of double factorization one-body Hamiltonian.

    Implements inner "half" of Fig. 15 in the reference. This block encoding is
    applied twice (with a reflection around the inner state preparation
    registers) to implement (roughly) the square of this one-body operator.

    Args:
        num_aux: Dimension of auxiliary index for double factorized Hamiltonian. Call L in Ref[1].
        num_spin_orb: The number of spin orbitals. Typically called $N$.
        num_eig: The total number of eigenvalues.
        num_bits_state_prep: The number of bits of precision for coherent alias
            sampling. Called $\aleph$ in the reference.
        num_bits_rot_aa_outer: Number of bits of precision for single qubit
            rotation for amplitude amplification. Called $b_r$ in the reference.
        num_bits_rot: Number of bits of precision for rotations for amplitude
            amplification in uniform state preparation. Called $\beth$ in Ref[1].

    Registers:
        succ_l: control for success for outer state preparation.
        succ_p: control for success for inner state preparation, this is reused
            in second application.
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
    num_eig: int
    num_bits_state_prep: int
    num_bits_rot_aa: int = 8
    num_bits_rot: int = 24

    @property
    def control_registers(self) -> Iterable[Register]:
        return (
            Register("succ_l", bitsize=1),
            Register("l_ne_zero", bitsize=1),
            Register("succ_p", bitsize=1),
        )

    @property
    def selection_registers(self) -> Iterable[Register]:
        return (
            Register("p", bitsize=(self.num_spin_orb // 2 - 1).bit_length()),
            Register("rot_aa", bitsize=1),
            Register("spin", bitsize=1),
        )

    @property
    def junk_registers(self) -> Iterable[Register]:
        nlxi = (self.num_eig + self.num_spin_orb // 2 - 1).bit_length()
        nxi = (self.num_spin_orb // 2 - 1).bit_length()
        return (
            Register("xi", bitsize=nxi),
            Register("offset", bitsize=nlxi),
            Register("rot", bitsize=self.num_bits_rot_aa),
            Register("rotations", bitsize=(self.num_spin_orb // 2) * self.num_bits_rot),
        )

    @property
    def target_registers(self) -> Iterable[Register]:
        return (Register("sys", bitsize=self.num_spin_orb // 2, shape=(2,)),)

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

    def short_name(self) -> str:
        return '$B[H_1]$'

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        succ_l: SoquetT,
        l_ne_zero: SoquetT,
        succ_p: SoquetT,
        p: SoquetT,
        rot_aa: SoquetT,
        spin: SoquetT,
        xi: SoquetT,
        offset: SoquetT,
        rot: SoquetT,
        rotations: SoquetT,
        sys: SoquetT,
    ) -> Dict[str, 'SoquetT']:
        # 1st half
        in_prep = InnerPrepareDoubleFactorization(
            num_aux=self.num_aux,
            num_spin_orb=self.num_spin_orb,
            num_eig=self.num_eig,
            num_bits_rot_aa=self.num_bits_rot_aa,
            num_bits_state_prep=self.num_bits_state_prep,
        )
        xi, offset, rot, succ_p, p = bb.add(
            in_prep, xi=xi, offset=offset, rot=rot, succ_p=succ_p, p=p
        )
        spin = bb.add(Hadamard(), q=spin)
        spin, sys[0], sys[1] = bb.add(CSwap(self.num_spin_orb // 2), ctrl=spin, x=sys[0], y=sys[1])
        prot = ProgRotGateArray(
            num_aux=self.num_aux,
            num_eig=self.num_eig,
            num_spin_orb=self.num_spin_orb,
            num_bits_rot=self.num_bits_rot,
        )
        offset, p, rotations, spin, sys = bb.add(
            prot, offset=offset, p=p, rotations=rotations, spin=spin, sys=sys
        )
        # missing l_ne_zero
        [succ_l, succ_p], sys[0] = bb.add(
            ApplyControlledZs((1, 1), self.num_spin_orb // 2), ctrls=[succ_l, succ_p], system=sys[0]
        )
        # 2nd half (invert preparation / swaps)
        prot = ProgRotGateArray(
            num_aux=self.num_aux,
            num_eig=self.num_eig,
            num_spin_orb=self.num_spin_orb,
            num_bits_rot=self.num_bits_rot,
        ).adjoint()
        offset, p, rotations, spin, sys = bb.add(
            prot, offset=offset, p=p, rotations=rotations, spin=spin, sys=sys
        )
        spin, sys[0], sys[1] = bb.add(CSwap(self.num_spin_orb // 2), ctrl=spin, x=sys[0], y=sys[1])
        in_prep_dag = InnerPrepareDoubleFactorization(
            num_aux=self.num_aux,
            num_spin_orb=self.num_spin_orb,
            num_eig=self.num_eig,
            num_bits_rot_aa=self.num_bits_rot_aa,
            num_bits_state_prep=self.num_bits_state_prep,
        ).adjoint()
        xi, offset, rot, succ_p, p = bb.add(
            in_prep_dag, xi=xi, offset=offset, rot=rot, succ_p=succ_p, p=p
        )
        spin = bb.add(Hadamard(), q=spin)

        return {
            'succ_l': succ_l,
            'l_ne_zero': l_ne_zero,
            'succ_p': succ_p,
            'p': p,
            'spin': spin,
            'rot_aa': rot_aa,
            'xi': xi,
            'offset': offset,
            'rot': rot,
            'rotations': rotations,
            'sys': sys,
        }

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        in_prep = InnerPrepareDoubleFactorization(
            num_aux=self.num_aux,
            num_spin_orb=self.num_spin_orb,
            num_eig=self.num_eig,
            num_bits_rot_aa=self.num_bits_rot_aa,
            num_bits_state_prep=self.num_bits_state_prep,
        )
        rot = ProgRotGateArray(
            num_aux=self.num_aux,
            num_eig=self.num_eig,
            num_spin_orb=self.num_spin_orb,
            num_bits_rot=self.num_bits_rot,
        )
        in_prep_dag = in_prep.adjoint()
        rot_dag = rot.adjoint()
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
        num_spin_orb: The number of spin orbitals. Typically called $N$.
        num_aux: Dimension of auxiliary index for double factorized Hamiltonian.
            Typically called $L$.
        num_eig: The total number of eigenvalues.
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
    num_eig: int
    num_bits_state_prep: int = 8
    num_bits_rot: int = 24
    num_bits_rot_aa_outer: int = 8
    num_bits_rot_aa_inner: int = 8

    @classmethod
    def build_from_coeffs(
        cls, one_body_ham: NDArray[np.float64], factorized_two_body_ham: NDArray[np.float64]
    ) -> 'DoubleFactorizationBlockEncoding':
        """Factory method to build double factorization block encoding given Hamiltonian inputs.

        Args:
            one_body_ham: One body hamiltonian ($T_{pq}$') matrix elements.
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
        return (Register('ctrl', bitsize=1, shape=(4,)),)

    @property
    def selection_registers(self) -> Iterable[Register]:
        return (
            Register("l", bitsize=self.num_aux.bit_length()),
            Register("p", bitsize=(self.num_spin_orb // 2 - 1).bit_length()),
            Register("spin", bitsize=1),
            Register('rot_aa', bitsize=1),
        )

    @property
    def junk_registers(self) -> Iterable[Register]:
        nlxi = (self.num_eig + self.num_spin_orb // 2 - 1).bit_length()
        nxi = (self.num_spin_orb // 2 - 1).bit_length()
        return (
            Register("xi", bitsize=nxi),
            Register("offset", bitsize=nlxi),
            Register("rot", bitsize=self.num_bits_rot_aa_inner),
            Register("rotations", bitsize=(self.num_spin_orb // 2) * self.num_bits_rot),
        )

    @property
    def target_registers(self) -> Iterable[Register]:
        return (Register("sys", bitsize=self.num_spin_orb // 2, shape=(2,)),)

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
        n_n = (self.num_spin_orb // 2 - 1).bit_length()  # C14
        outer_prep = OuterPrepareDoubleFactorization(
            self.num_aux,
            num_bits_state_prep=self.num_bits_state_prep,
            num_bits_rot_aa=self.num_bits_rot_aa_outer,
        )
        l, succ_l = bb.add(outer_prep, l=l, succ_l=succ_l)
        in_l_data_l = OutputIndexedData(
            num_aux=self.num_aux,
            num_spin_orb=self.num_spin_orb,
            num_eig=self.num_eig,
            num_bits_rot_aa=self.num_bits_rot_aa_inner,
        )
        l, l_ne_zero, xi, rot, offset = bb.add(
            in_l_data_l, l=l, l_ne_zero=l_ne_zero, xi=xi, rot_data=rot, offset=offset
        )
        one_body = DoubleFactorizationOneBody(
            self.num_aux,
            self.num_spin_orb,
            self.num_eig,
            self.num_bits_state_prep,
            num_bits_rot_aa=self.num_bits_rot_aa_inner,
            num_bits_rot=self.num_bits_rot,
        )
        succ_l, l_ne_zero, succ_p, p, rot_aa, spin, xi, offset, rot, rotations, sys = bb.add(
            one_body,
            succ_l=succ_l,
            l_ne_zero=l_ne_zero,
            succ_p=succ_p,
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
        succ_l, l_ne_zero, p, spin = bb.add(
            Reflection((1, 1, n_n, 1), (1, 1, 0, 0)), reg0=succ_l, reg1=l_ne_zero, reg2=p, reg3=spin
        )
        succ_l, l_ne_zero, succ_p, p, rot_aa, spin, xi, offset, rot, rotations, sys = bb.add(
            one_body,
            succ_l=succ_l,
            l_ne_zero=l_ne_zero,
            succ_p=succ_p,
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
            num_eig=self.num_eig,
            num_bits_rot_aa=self.num_bits_rot_aa_inner,
        ).adjoint()
        l, l_ne_zero, xi, rot, offset = bb.add(
            in_l_data_l, l=l, l_ne_zero=l_ne_zero, xi=xi, rot_data=rot, offset=offset
        )
        # prepare_l^dag
        outer_prep = OuterPrepareDoubleFactorization(
            self.num_aux,
            num_bits_state_prep=self.num_bits_state_prep,
            num_bits_rot_aa=self.num_bits_rot_aa_outer,
        ).adjoint()
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
    num_eig = num_aux * (num_spin_orb // 2)
    df_one_body = DoubleFactorizationOneBody(
        num_aux=num_aux,
        num_spin_orb=num_spin_orb,
        num_eig=num_eig,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot=num_bits_rot,
    )
    return df_one_body


@bloq_example
def _df_block_encoding() -> DoubleFactorizationBlockEncoding:
    num_spin_orb = 10
    num_aux = 50
    num_eig = num_aux * (num_spin_orb // 2)
    num_bits_state_prep = 12
    num_bits_rot = 7
    df_block_encoding = DoubleFactorizationBlockEncoding(
        num_spin_orb=num_spin_orb,
        num_aux=num_aux,
        num_eig=num_eig,
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
