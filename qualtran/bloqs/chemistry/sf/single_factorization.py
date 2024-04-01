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
r"""Bloqs for the single-factorized chemistry Hamiltonian in second quantization.

Recall that for the single factorized Hamiltonian we have
$$
    H = \sum_{pq}T'_{pq} a_p^\dagger a_q + \frac{1}{2} \sum_l \left(\sum_{pq}
W_{pq}^{(l)} a_p^\dagger a_q\right)^2.
$$
where $\sum_l^L W_{pq}^{(l)} W_{rs}^{(l)} = (pq|rs)$ are the standard chemist's
electron repulsion integrals.
"""

from functools import cached_property
from typing import Dict, Iterable, Set, TYPE_CHECKING

import numpy as np
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QAny,
    QBit,
    Register,
    Signature,
    SoquetT,
)
from qualtran._infra.data_types import BoundedQUInt
from qualtran.bloqs.basic_gates import Hadamard
from qualtran.bloqs.basic_gates.swap import CSwap
from qualtran.bloqs.chemistry.sf.prepare import (
    InnerPrepareSingleFactorization,
    OuterPrepareSingleFactorization,
)
from qualtran.bloqs.chemistry.sf.select_bloq import SelectSingleFactorization
from qualtran.bloqs.reflection import Reflection

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class SingleFactorizationOneBody(Bloq):
    r"""Block encoding of single factorization one-body Hamiltonian.

    Implements inner "half" of Fig. 15 in the reference. This block encoding is
    applied twice (with a reflection around the inner state preparation
    registers) to implement (roughly) the square of this one-body operator.

    Args:
        num_aux: Dimension of auxiliary index for single factorized Hamiltonian.
            Called $L$ in the reference.
        num_spin_orb: The number of spin orbitals. Typically called $N$.
        num_bits_state_prep: The number of bits of precision for coherent alias
            sampling. Called $\aleph$ in the reference.
        num_bits_rot_aa: Number of bits of precision for rotations for amplitude
            amplification in uniform state preparation. Called $b_r$ in the reference.
        adjoint: Whether this bloq is daggered or not. This affects the QROM cost.
        kp1: QROAM blocking factor for data prepared over l (auxiliary) index.
            Defaults to 1 (i.e. QROM).
        kp1: QROAM blocking factor for data prepared over pq indicies. Defaults to 1 (i.e.) QROM.
        kp1_inv: QROAM blocking factor for inverting data prepared over l (auxiliary) index.
        kp1_inv: QROAM blocking factor for inverting of data prepared over pq.

    Registers:
        succ_l: Flag for success of l state preparation.
        l_ne_zero: Flag for l != 0.
        succ_pq: Flag for success of pq.
        l: register to store L values for auxiliary index.
        p: spatial orbital index. range(0, num_spin_orb // 2)
        q: spatial orbital index. range(0, num_spin_orb // 2)
        rot_aa: Qubit to rotate for amplitude amplification for state preparation.
        swap_pq: Qubit for controlling swaps over p and q registers.
        spin: Qubit for controlling swaps over system registers.
        sys: The system registers.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://arxiv.org/abs/2011.03494) Fig. 15 page 43.
    """
    num_aux: int
    num_spin_orb: int
    num_bits_state_prep: int
    num_bits_rot_aa: int = 8
    adjoint: bool = False
    kp1: int = 1
    kp2: int = 1
    kp1_inv: int = 1
    kp2_inv: int = 1

    @property
    def control_registers(self) -> Iterable[Register]:
        return (
            Register("succ_l", QBit()),
            Register("l_ne_zero", QBit()),
            Register('succ_pq', QBit()),
        )

    @property
    def selection_registers(self) -> Iterable[Register]:
        return (
            Register(
                "l",
                BoundedQUInt(bitsize=self.num_aux.bit_length(), iteration_length=self.num_aux + 1),
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
            Register("rot_aa", BoundedQUInt(bitsize=1)),
            Register("swap_pq", BoundedQUInt(bitsize=1)),
            Register("spin", BoundedQUInt(bitsize=1)),
        )

    @property
    def target_registers(self) -> Iterable[Register]:
        return (Register("sys", QAny(bitsize=self.num_spin_orb // 2), shape=(2,)),)

    @property
    def junk_registers(self) -> Iterable[Register]:
        return ()

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
        *,
        succ_l: SoquetT,
        l_ne_zero: SoquetT,
        succ_pq: SoquetT,
        l: SoquetT,
        p: SoquetT,
        q: SoquetT,
        rot_aa: SoquetT,
        swap_pq: SoquetT,
        spin: SoquetT,
        sys: SoquetT,
    ) -> Dict[str, 'SoquetT']:
        iprep = InnerPrepareSingleFactorization(
            self.num_aux,
            self.num_spin_orb,
            self.num_bits_state_prep,
            self.num_bits_rot_aa,
            kp1=self.kp1,
            kp2=self.kp2,
        )
        l, p, q, succ_pq = bb.add(iprep, l=l, p=p, q=q, succ_pq=succ_pq)
        spin = bb.add(Hadamard(), q=spin)
        swap_pq = bb.add(Hadamard(), q=swap_pq)
        n = (self.num_spin_orb // 2 - 1).bit_length()
        swap_pq, p, q = bb.add(CSwap(n), ctrl=swap_pq, x=p, y=q)
        p, q, spin, succ_pq, succ_l = bb.add(
            SelectSingleFactorization(num_spin_orb=self.num_spin_orb),
            p=p,
            q=q,
            spin=spin,
            succ_pq=succ_pq,
            succ_l=succ_l,
        )
        swap_pq, p, q = bb.add(CSwap(n), ctrl=swap_pq, x=p, y=q)
        spin = bb.add(Hadamard(), q=spin)
        swap_pq = bb.add(Hadamard(), q=swap_pq)
        iprep_dag = InnerPrepareSingleFactorization(
            self.num_aux,
            self.num_spin_orb,
            self.num_bits_state_prep,
            self.num_bits_rot_aa,
            kp1=self.kp1_inv,
            kp2=self.kp2_inv,
        ).adjoint()
        l, p, q, succ_pq = bb.add(iprep_dag, l=l, p=p, q=q, succ_pq=succ_pq)
        return {
            'succ_l': succ_l,
            'l_ne_zero': l_ne_zero,
            'succ_pq': succ_pq,
            'l': l,
            'p': p,
            'q': q,
            'rot_aa': rot_aa,
            'swap_pq': swap_pq,
            'spin': spin,
            'sys': sys,
        }

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        iprep = InnerPrepareSingleFactorization(
            self.num_aux,
            self.num_spin_orb,
            self.num_bits_state_prep,
            self.num_bits_rot_aa,
            kp1=self.kp1,
            kp2=self.kp2,
        )
        iprep_dag = InnerPrepareSingleFactorization(
            self.num_aux,
            self.num_spin_orb,
            self.num_bits_state_prep,
            self.num_bits_rot_aa,
            kp1=self.kp1_inv,
            kp2=self.kp2_inv,
        ).adjoint()
        n = (self.num_spin_orb // 2 - 1).bit_length()
        return {
            (iprep, 1),
            (iprep_dag, 1),
            (CSwap(n), 2),
            (SelectSingleFactorization(num_spin_orb=self.num_spin_orb), 1),
            (Hadamard(), 4),
        }


@frozen
class SingleFactorizationBlockEncoding(Bloq):
    r"""Block encoding of single factorization Hamiltonian.

    Implements Fig. 15 in the reference.

    Args:
        num_spin_orb: The number of spin orbitals. Typically called N.
        num_aux: Dimension of auxiliary index for single factorized Hamiltonian.
            Called $L$ in the reference.
        num_bits_state_prep: The number of bits of precision for coherent alias
            sampling. Called $\aleph$ in the reference.
        num_bits_rot_aa_outer: Number of bits of precision for rotations for amplitude
            amplification in outer uniform state preparation. Called $b_r$ in the reference.
        num_bits_rot_aa_inner: Number of bits of precision for rotations for amplitude
            amplification in inner uniform state preparation. Called $b_r$ in the reference.
        kp1: QROAM blocking factor for data prepared over l (auxiliary) index.
            Defaults to 1 (i.e. QROM).
        kp1: QROAM blocking factor for data prepared over pq indicies. Defaults to 1 (i.e.) QROM.
        kp1_inv: QROAM blocking factor for inverting the data prepared over l (auxiliary) index.
        kp1_inv: QROAM blocking factor for inverting the data prepared over pq.

    Registers:
        ctrl: Control registers for state preparation.
        l: register to store L values for auxiliary index.
        p: spatial orbital index. range(0, num_spin_orb // 2)
        q: spatial orbital index. range(0, num_spin_orb // 2)
        rot_aa: Qubit to rotate for amplitude amplification for state preparation.
        swap_pq: Qubit for controlling swaps over p and q registers.
        spin: Qubit for controlling swaps over system registers.
        sys: The system registers.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://arxiv.org/abs/2011.03494) Fig 15, page 43.
    """
    num_spin_orb: int
    num_aux: int
    num_bits_state_prep: int
    num_bits_rot_aa_outer: int = 8
    num_bits_rot_aa_inner: int = 8
    kp1: int = 1
    kp2: int = 1
    kp1_inv: int = 1
    kp2_inv: int = 1

    @property
    def control_registers(self) -> Iterable[Register]:
        return (Register('ctrl', QBit(), shape=(3,)),)

    @property
    def selection_registers(self) -> Iterable[Register]:
        return (
            Register("l", QAny(bitsize=self.num_aux.bit_length())),
            Register("pq", QAny(bitsize=(self.num_spin_orb // 2 - 1).bit_length()), shape=(2,)),
            Register("rot_aa", QBit(), shape=(2,)),
            Register("swap_pq", QBit()),
            Register("spin", QBit()),
        )

    @property
    def junk_registers(self) -> Iterable[Register]:
        return ()

    @property
    def target_registers(self) -> Iterable[Register]:
        return (Register("sys", QAny(bitsize=self.num_spin_orb // 2), shape=(2,)),)

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
        *,
        ctrl: SoquetT,
        l: SoquetT,
        pq: SoquetT,
        rot_aa: SoquetT,
        swap_pq: SoquetT,
        spin: SoquetT,
        sys: SoquetT,
    ) -> Dict[str, 'SoquetT']:
        succ_l, l_ne_zero, succ_pq = ctrl
        p, q = pq
        # prepare_l
        outer_prep = OuterPrepareSingleFactorization(
            self.num_aux,
            num_bits_state_prep=self.num_bits_state_prep,
            num_bits_rot_aa=self.num_bits_rot_aa_outer,
        )
        l, succ_l, l_ne_zero, rot_aa[0] = bb.add(
            outer_prep, l=l, succ_l=succ_l, l_ne_zero=l_ne_zero, rot_aa=rot_aa[0]
        )
        one_body = SingleFactorizationOneBody(
            self.num_aux,
            self.num_spin_orb,
            self.num_bits_state_prep,
            self.num_bits_rot_aa_inner,
            kp1=self.kp1,
            kp1_inv=self.kp1_inv,
            kp2=self.kp2,
            kp2_inv=self.kp2_inv,
        )
        succ_l, l_ne_zero, succ_pq, l, p, q, swap_pq, spin, rot_aa[1], sys = bb.add(
            one_body,
            succ_l=succ_l,
            l_ne_zero=l_ne_zero,
            succ_pq=succ_pq,
            l=l,
            p=p,
            q=q,
            swap_pq=swap_pq,
            spin=spin,
            rot_aa=rot_aa[1],
            sys=sys,
        )
        # reflect about the inner state preparation registers, controlled on succ_l and l_ne_zero.
        n_n = (self.num_spin_orb // 2 - 1).bit_length()
        succ_l, l_ne_zero, p, q, swap_pq, spin = bb.add(
            Reflection((1, 1, n_n, n_n, 1, 1), cvs=(1, 1, 0, 0, 0, 0)),
            reg0=succ_l,
            reg1=l_ne_zero,
            reg2=p,
            reg3=q,
            reg4=swap_pq,
            reg5=spin,
        )
        # apply one-body again
        succ_l, l_ne_zero, succ_pq, l, p, q, swap_pq, spin, rot_aa[1], sys = bb.add(
            one_body,
            succ_l=succ_l,
            l_ne_zero=l_ne_zero,
            succ_pq=succ_pq,
            l=l,
            p=p,
            q=q,
            swap_pq=swap_pq,
            spin=spin,
            rot_aa=rot_aa[1],
            sys=sys,
        )
        # prepare_l^dag
        outer_prep = OuterPrepareSingleFactorization(
            self.num_aux,
            num_bits_state_prep=self.num_bits_state_prep,
            num_bits_rot_aa=self.num_bits_rot_aa_outer,
        ).adjoint()
        l, succ_l, l_ne_zero, rot_aa[1] = bb.add(
            outer_prep, l=l, succ_l=succ_l, l_ne_zero=l_ne_zero, rot_aa=rot_aa[1]
        )
        out_regs = {
            'ctrl': [succ_l, l_ne_zero, succ_pq],
            'l': l,
            'pq': np.array([p, q]),
            'rot_aa': rot_aa,
            'swap_pq': swap_pq,
            'spin': spin,
            'sys': sys,
        }
        return out_regs


@bloq_example
def _sf_one_body() -> SingleFactorizationOneBody:
    num_aux = 50
    num_bits_state_prep = 12
    num_bits_rot_aa = 7
    num_spin_orb = 10
    sf_one_body = SingleFactorizationOneBody(
        num_aux=num_aux,
        num_spin_orb=num_spin_orb,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa=num_bits_rot_aa,
        adjoint=False,
    )
    return sf_one_body


@bloq_example
def _sf_block_encoding() -> SingleFactorizationBlockEncoding:
    num_spin_orb = 10
    num_aux = 50
    num_bits_state_prep = 12
    sf_block_encoding = SingleFactorizationBlockEncoding(
        num_spin_orb=num_spin_orb,
        num_aux=num_aux,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_rot_aa_outer=7,
        num_bits_rot_aa_inner=7,
    )
    return sf_block_encoding


_SF_ONE_BODY = BloqDocSpec(
    bloq_cls=SingleFactorizationOneBody,
    import_line='from qualtran.bloqs.chemistry.sf.single_factorization import SingleFactorizationOneBody',
    examples=(_sf_one_body,),
)

_SF_BLOCK_ENCODING = BloqDocSpec(
    bloq_cls=SingleFactorizationBlockEncoding,
    import_line='from qualtran.bloqs.chemistry.sf.single_factorization import SingleFactorizationBlockEncoding',
    examples=(_sf_block_encoding,),
)
