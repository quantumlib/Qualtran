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
r"""SELECT and PREPARE for the first quantized chemistry Hamiltonian."""
from functools import cached_property
from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING
import numpy as np

from attrs import frozen

from qualtran import Bloq, BloqBuilder, Register, Signature, SoquetT, SelectionRegister
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.basic_gates import Hadamard
from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_t import PrepareTFirstQuantization
from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_uv import PrepareUVFirstQuantization
from qualtran.bloqs.chemistry.pbc.first_quantization.select_t import SelectTFirstQuantization
from qualtran.bloqs.chemistry.pbc.first_quantization.select_uv import SelectUVFirstQuantization
from qualtran.bloqs.select_and_prepare import PrepareOracle, SelectOracle

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class PrepareTUVSuperpositions(Bloq):
    r"""Prepare the superposition over registers selecting between T U and V.

    This will produce 3 qubits for flagging which term to apply. xx0 -> T, x0x -> U or V, 0xx -> V.

    Args:
        adjoint: whether to dagger the bloq or not.

    Registers:
        tuv: a single qubit rotated to appropriately weight T and U or V.
        uv: a single qubit rotated to appropriately weight U or V.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) page 15, section A
    """
    num_bits_t: int
    eta: int
    lambda_zeta: int
    num_bits_rot_aa: int = 8
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(tuv=1, uv=1)

    def short_name(self) -> str:
        return 'PREP TUV'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        n_eta_zeta = (self.eta + 2 * self.lambda_zeta - 1).bit_length()
        # The cost arises from rotating a qubit, and uniform state preparation
        # over eta + 2 lambda_zeta numbers along.
        return {(Toffoli(), self.num_bits_t + 4 * n_eta_zeta + 2 * self.num_bits_rot_aa - 12)}


@frozen
class UniformSuperpostionIJFirstQuantization(Bloq):
    r"""Uniform superposition over $\eta$ values of $i$ and $j$ in unary such that $i \ne j$.

    Args:
        eta: The number of electrons.
        num_bits_rot_aa: The number of bits of precision for the single qubit
            rotation for amplitude amplification. Called $b_r$ in the reference.
        adjoint: whether to dagger the bloq or not.

    Registers:
        i: a n_eta bit register for unary encoding of eta numbers.
        j: a n_eta bit register for unary encoding of eta numbers.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) page 18, section A, around Eq 62.
    """
    eta: int
    num_bits_rot_aa: int
    adjoint: int = False

    @cached_property
    def signature(self) -> Signature:
        n_eta = (self.eta - 1).bit_length()
        return Signature.build(i=n_eta, j=n_eta)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        n_eta = (self.eta - 1).bit_length()
        # Half of Eq. 62 which is the cost for prep and prep^\dagger
        return {(Toffoli(), (7 * n_eta + 4 * self.num_bits_rot_aa - 18))}


@frozen
class SWAPIJ(Bloq):
    """Placeholder for swap combined with unary iteration over i/j."""

    eta: int
    trg_bitsize: int

    @cached_property
    def signature(self) -> Signature:
        sel_bitsize = (self.eta - 1).bit_length()
        return Signature(
            [
                Register("sel", bitsize=sel_bitsize),
                Register("sys", bitsize=self.trg_bitsize, shape=(self.eta, 3)),
                Register("anc", bitsize=self.trg_bitsize, shape=(3)),
            ]
        )

    def short_name(self) -> str:
        return r'SWAP'

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        return {(3 * self.eta * self.trg_bitsize + self.eta - 2, Toffoli())}


@frozen
class PrepareFirstQuantization(PrepareOracle):
    """State preparation for the first quantized chemistry Hamiltonian."""

    num_bits_p: int
    eta: int
    num_atoms: int
    lambda_zeta: int
    m_param: int = 2**8
    num_bits_nuc_pos: int = 16
    num_bits_t: int = 16
    num_bits_rot_aa: int = 8
    adjoint: bool = False

    @property
    @abc.abstractmethod
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        n_nu = self.num_bits_p + 1
        n_eta = (self.eta - 1).bit_length()
        n_at = (self.num_atoms - 1).bit_length()
        n_m = (self.m_param - 1).bit_length()
        # Note actual reflections costs:
        # uv: should be really n_{eta zeta} + 1 qubits, we're abstracting this to a single qubit.
        # ij: + 2 for rotated qubits during AA.
        # w: missing one for rotated qubit.
        # overflow: 3 * 2 qubits are missing.
        # l: should not be reflected on.
        return (
            SelectionRegister('tuv', bitsize=1, 2),
            SelectionRegister('uv', bitize=1, 2),
            SelectionRegister('ij', bitsize=n_eta, shape=(2,)),
            SelectionRegister("w", bitsize=3),
            SelectionRegister("r", bitsize=self.num_bits_p),
            SelectionRegister("s", bitsize=self.num_bits_p),
            SelectionRegister("mu", bitsize=self.num_bits_p),
            SelectionRegister("nu", bitsize=n_nu, shape=(3,)),
            SelectionRegister("m", bitsize=n_m),
            SelectionRegister("l", bitsize=n_at),
            )

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return (
            Register("succ_nu", bitsize=1),
            Register("plus_t", bitsize=1),
        )

    def short_name(self) -> str:
        return r'PREP'

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        tuv: SoquetT,
        uv: SoquetT,
        plus_t: SoquetT,
        ij: SoquetT,
        w: SoquetT,
        r: SoquetT,
        s: SoquetT,
        mu: SoquetT,
        nu: SoquetT,
        m: SoquetT,
        succ_nu: SoquetT,
        l: SoquetT,
    ) -> Dict[str, 'SoquetT']:
        tuv, uv = bb.add(
            PrepareTUVSuperpositions(
                self.num_bits_t,
                self.eta,
                self.lambda_zeta,
                self.num_bits_rot_aa,
                adjoint=self.adjoint,
            ),
            tuv=tuv,
            uv=uv,
        )
        ij = bb.add(UniformSuperpostionIJFirstQuantization(self.eta, self.num_bits_rot_aa, self.adjoint), ij=ij)
        # |+>
        plus_t = bb.add(Hadamard(), q=plus_t)
        w, r, s = bb.add(
            PrepareTFirstQuantization(
                self.num_bits_p, self.eta, self.num_bits_rot_aa, adjoint=self.adjoint
            ),
            w=w,
            r=r,
            s=s,
        )
        mu, nu, m, l, succ_nu = bb.add(
            PrepareUVFirstQuantization(
                self.num_bits_p,
                self.eta,
                self.num_atoms,
                self.m_param,
                self.lambda_zeta,
                self.num_bits_nuc_pos,
                adjoint=self.adjoint,
            ),
            mu=mu,
            nu=nu,
            m=m,
            l=l,
            succ_nu=succ_nu,
        )
        return {
            'tuv': tuv,
            'uv': uv,
            'plus_t': plus_t,
            'ij': ij,
            'w': w,
            'r': r,
            's': s,
            'mu': mu,
            'nu': nu,
            'm': m,
            'l': l,
            'succ_nu': succ_nu,
        }


def allocate_shaped(bb, shape, bitsize):
    size = bb.prod(shape)
    anc = bb.allocate(size * bitsize)
    anc = bb.split(anc)
    anc = [bb.join(anc[s*bitsize:(s+1)*bitsize] for s in range(size))]
    return np.reshape(anc, shape)


@frozen
class SelectFirstQuantization(SelectOracle):
    """State preparation for the first quantized chemistry Hamiltonian."""

    num_bits_p: int
    eta: int
    num_atoms: int
    lambda_zeta: int
    m_param: int = 2**8
    num_bits_nuc_pos: int = 16
    num_bits_t: int = 16
    num_bits_rot_aa: int = 8
    adjoint: bool = False

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return (Register("tuv", bitsize=1),
                Register("uv", bitsize=1), 
                Register("succ_nu", bitsize=1),
                Register("plus_t", bitsize=1),
                )

    @cached_property
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        n_nu = self.num_bits_p + 1
        n_eta = (self.eta - 1).bit_length()
        n_at = (self.num_atoms - 1).bit_length()
        n_m = (self.m_param - 1).bit_length()
        return (
            SelectionRegister('ij', bitsize=n_eta, shape=(2,)),
            SelectionRegister("w", bitsize=3),
            SelectionRegister("r", bitsize=self.num_bits_p),
            SelectionRegister("s", bitsize=self.num_bits_p),
            SelectionRegister("mu", bitsize=self.num_bits_p),
            SelectionRegister("nu", bitsize=n_nu, shape=(3,)),
            SelectionRegister("m", bitsize=n_m),
            SelectionRegister("l", bitsize=n_at),
            )

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register("sys", bitsize=self.num_bits_p, shape=(self.eta, 3)),)

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )

    def short_name(self) -> str:
        return r'SELECT'

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        tuv: SoquetT,
        uv: SoquetT,
        plus_t: SoquetT,
        ij: SoquetT,
        w: SoquetT,
        r: SoquetT,
        s: SoquetT,
        mu: SoquetT,
        nu: SoquetT,
        m: SoquetT,
        succ_nu: SoquetT,
        l: SoquetT,
        sys: SoquetT,
        pq: SoquetT,
    ) -> Dict[str, 'SoquetT']:
        p = allocate_shaped(bb, (3,), self.num_bits_p + 1) 
        q = allocate_shaped(bb, (3,), self.num_bits_p + 1) 
        rl = bb.allocate(self.num_bits_nuc_pos)
        i, j = ij
        i, sys, p = bb.add(SWAPIJ(self.eta, self.num_bits_p + 1), sel=i, sys=sys, anc=p)
        j, sys, q = bb.add(SWAPIJ(self.eta, self.num_bits_p + 1), sel=j, sys=sys, anc=q)
        p, plus_t, tuv, w, r, s = bb.add(
            SelectTFirstQuantization(self.num_bits_p, self.eta),
            p=p,
            plus=plus_t,
            flag_T=tuv,
            w=w,
            r=r,
            s=s,
        )
        tuv, uv, l, rl, nu, p, q = bb.add(
            SelectUVFirstQuantization(self.num_bits_nuc_pos, self.eta, self.num_bits_nuc_pos),
            flag_tuv=tuv,
            flag_uv=uv,
            l=l,
            rl=rl,
            nu=nu,
            p=p,
            q=q,
        )
        i, sys, p = bb.add(SWAPIJ(self.eta, self.num_bits_p + 1), sel=i, sys=sys, anc=p)
        j, sys, q = bb.add(SWAPIJ(self.eta, self.num_bits_p + 1), sel=j, sys=sys, anc=q)
        bb.free(bb.join(p.ravel()))
        bb.free(bb.join(q.ravel()))
        bb.free(rl)
        ij = [i, j]
        return {
            'tuv': tuv,
            'uv': uv,
            'plus_t': plus_t,
            'ij': ij,
            'w': w,
            'r': r,
            's': s,
            'mu': mu,
            'nu': nu,
            'm': m,
            'l': l,
            'succ_nu': succ_nu,
            'sys': sys,
        }
