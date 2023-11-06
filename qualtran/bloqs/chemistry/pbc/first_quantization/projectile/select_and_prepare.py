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
r"""SELECT and PREPARE for the first quantized chemistry Hamiltonian with a quantum projectile."""
from functools import cached_property
from typing import Dict, Set, Tuple, TYPE_CHECKING

from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    Register,
    SelectionRegister,
    Signature,
    SoquetT,
)
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_t import (
    PrepareTFirstQuantizationWithProj,
)
from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_uv import (
    PrepareUVFirstQuantizationWithProj,
)
from qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare import (
    UniformSuperpostionIJFirstQuantization,
)
from qualtran.bloqs.chemistry.pbc.first_quantization.select_t import SelectTFirstQuantization
from qualtran.bloqs.chemistry.pbc.first_quantization.select_uv import SelectUVFirstQuantization
from qualtran.bloqs.select_and_prepare import PrepareOracle, SelectOracle

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class PrepareTUVSuperpositions(Bloq):
    r"""Prepare the superposition over registers selecting between T U and V.

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
        if self.adjoint:
            # inverting inequality tests at zero Toffoli.
            return {}
        else:
            return {(Toffoli(), 6 * self.num_bits_t + 2)}


@frozen
class PrepareFirstQuantizationWithProj(PrepareOracle):
    r"""State preparation for the first quantized chemistry Hamiltonian with a quntum projectile.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        num_bits_n: The number of bits to represent each dimension of the
            momentum register for the projectile. Should be larger than num_bits_p.
        eta: The number of electrons.
        num_atoms: The number of atoms. $L$ in the reference.
        lambda_zeta: sum of nuclear charges.
        m_param: $\mathcal{M}$ in the reference.
        num_bits_nuc_pos: The number of bits of precision for representing the nuclear coordinates.
        num_bits_t: The number of bits of precision for the state preparation
            over the register selecting between the different components of the
            Hamiltonian.
        num_bits_rot_aa: The number of bits of precision for the rotation for
            amplitude amplification.
        adjoint: Whether to dagger the bloq or not.

    Registers:
        tuv: Flag register for selecting between kinetic and potential terms in the Hamiltonian.
        uv: Flag register for selecting between the different potential
            components of the Hamiltonian.
        i: A register for selecting electronic registers.
        j: A register for selecting electronic registers.
        w: A register for selecting x, y and z components of the momentum register.
        r: A register for controlling elements of the momentum register. Used
            for block encodiding kinetic energy operator.
        s: A register for controlling elements of the momentum register. Used
            for block encodiding kinetic energy operator.
        mu: A register used for implementing nested boxes for the momentum state preparation.
        nu_x: x component of the momentum register for Coulomb potential.
        nu_y: y component of the momentum register for Coulomb potential.
        nu_z: z component of the momentum register for Coulomb potential.
        m: an ancilla register in a uniform superposition.
        l: The register for selecting the nuclei.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767)
    """

    num_bits_p: int
    num_bits_n: int
    eta: int
    num_atoms: int
    lambda_zeta: int
    m_param: int = 2**8
    num_bits_nuc_pos: int = 16
    num_bits_t: int = 16
    num_bits_rot_aa: int = 8
    adjoint: bool = False

    @property
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        n_nu = self.num_bits_n + 1
        n_eta = (self.eta - 1).bit_length()
        n_at = (self.num_atoms - 1).bit_length()
        n_m = (self.m_param - 1).bit_length()
        # Note actual reflections costs:
        # uv: should be really n_{eta zeta} + 1 qubits, we're abstracting this to a single qubit.
        # ij: + 2 for rotated qubits during AA.
        # w: missing one for rotated qubit. + 1 flag qubit (no reflection)
        # overflow: 3 * 2 qubits are missing.
        # l: should not be reflected on.
        return (
            SelectionRegister('tuv', bitsize=1, iteration_length=2),
            SelectionRegister('uv', bitsize=1, iteration_length=2),
            SelectionRegister('i', bitsize=n_eta, iteration_length=self.eta),
            SelectionRegister('j', bitsize=n_eta, iteration_length=self.eta),
            SelectionRegister("w", iteration_length=3, bitsize=2),
            SelectionRegister("w_mean", iteration_length=3, bitsize=2),
            SelectionRegister("r", bitsize=self.num_bits_n),
            SelectionRegister("s", bitsize=self.num_bits_n),
            SelectionRegister("mu", bitsize=self.num_bits_n),
            SelectionRegister("nu_x", bitsize=n_nu),
            SelectionRegister("nu_y", bitsize=n_nu),
            SelectionRegister("nu_z", bitsize=n_nu),
            SelectionRegister("m", bitsize=n_m),
            SelectionRegister("l", bitsize=n_at),
        )

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return (Register("succ_nu", bitsize=1), Register("plus_t", bitsize=1))

    def short_name(self) -> str:
        return r'PREP'

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        tuv: SoquetT,
        uv: SoquetT,
        plus_t: SoquetT,
        i: SoquetT,
        j: SoquetT,
        w: SoquetT,
        w_mean: SoquetT,
        r: SoquetT,
        s: SoquetT,
        mu: SoquetT,
        nu_x: SoquetT,
        nu_y: SoquetT,
        nu_z: SoquetT,
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
        i, j = bb.add(
            UniformSuperpostionIJFirstQuantization(
                self.eta, self.num_bits_rot_aa, adjoint=self.adjoint
            ),
            i=i,
            j=j,
        )
        # # |+>
        # plus_t = bb.add(Hadamard(), q=plus_t)
        w, w_mean, r, s = bb.add(
            PrepareTFirstQuantizationWithProj(
                self.num_bits_p,
                self.num_bits_n,
                self.eta,
                self.num_bits_rot_aa,
                adjoint=self.adjoint,
            ),
            w=w,
            w_mean=w_mean,
            r=r,
            s=s,
        )
        mu, [nu_x, nu_y, nu_z], m, l, succ_nu = bb.add(
            PrepareUVFirstQuantizationWithProj(
                self.num_bits_p,
                self.num_bits_n,
                self.eta,
                self.num_atoms,
                self.m_param,
                self.lambda_zeta,
                self.num_bits_nuc_pos,
                adjoint=self.adjoint,
            ),
            mu=mu,
            nu=[nu_x, nu_y, nu_z],
            m=m,
            l=l,
            flag_nu=succ_nu,
        )
        return {
            'tuv': tuv,
            'uv': uv,
            'plus_t': plus_t,
            'i': i,
            'j': j,
            'w': w,
            'w_mean': w_mean,
            'r': r,
            's': s,
            'mu': mu,
            'nu_x': nu_x,
            'nu_y': nu_y,
            'nu_z': nu_z,
            'm': m,
            'l': l,
            'succ_nu': succ_nu,
        }


@frozen
class SelectFirstQuantizationWithProj(SelectOracle):
    r"""SELECT operation for the first quantized chemistry Hamiltonian.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        eta: The number of electrons.
        num_atoms: The number of atoms. $L$ in the reference.
        lambda_zeta: sum of nuclear charges.
        m_param: $\mathcal{M}$ in the reference.
        num_bits_nuc_pos: The number of bits of precision for representing the nuclear coordinates.
        num_bits_t: The number of bits of precision for the state preparation
            over the register selecting between the different components of the
            Hamiltonian.
        num_bits_rot_aa: The number of bits of precision for the rotation for
            amplitude amplification.
        adjoint: Whether to dagger the bloq or not.

    Registers:
        tuv: Flag register for selecting between kinetic and potential terms in the Hamiltonian.
        uv: Flag register for selecting between the different potential
            components of the Hamiltonian.
        i_ne_j: Register flagging $i \ne j$
        plus_t: A register prepared in the $|+\rangle$ state.
        i: A register for selecting electronic registers.
        j: A register for selecting electronic registers.
        w: A register for selecting x, y and z components of the momentum register.
        r: A register for controlling elements of the momentum register. Used
            for block encodiding kinetic energy operator.
        s: A register for controlling elements of the momentum register. Used
            for block encodiding kinetic energy operator.
        mu: A register used for implementing nested boxes for the momentum state preparation.
        nu_x: x component of the momentum register for Coulomb potential.
        nu_y: y component of the momentum register for Coulomb potential.
        nu_z: z component of the momentum register for Coulomb potential.
        m: an ancilla register in a uniform superposition.
        l: The register for selecting the nuclei.
        sys: The system register. Will store $\eta$ registers (x, y and z)
            compents of size num_bits_p.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767)
    """

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
        return (
            Register("tuv", bitsize=1),
            Register("uv", bitsize=1),
            Register("i_ne_j", bitsize=1),
            Register("plus_t", bitsize=1),
        )

    @cached_property
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        n_nu = self.num_bits_p + 1
        n_eta = (self.eta - 1).bit_length()
        n_at = (self.num_atoms - 1).bit_length()
        n_m = (self.m_param - 1).bit_length()
        return (
            SelectionRegister('i', bitsize=n_eta, iteration_length=self.eta),
            SelectionRegister('j', bitsize=n_eta, iteration_length=self.eta),
            SelectionRegister("w", bitsize=3),
            SelectionRegister("r", bitsize=self.num_bits_p),
            SelectionRegister("s", bitsize=self.num_bits_p),
            SelectionRegister("mu", bitsize=self.num_bits_p),
            SelectionRegister("nu_x", bitsize=n_nu),
            SelectionRegister("nu_y", bitsize=n_nu),
            SelectionRegister("nu_z", bitsize=n_nu),
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
        i_ne_j: SoquetT,
        plus_t: SoquetT,
        i: SoquetT,
        j: SoquetT,
        w: SoquetT,
        r: SoquetT,
        s: SoquetT,
        mu: SoquetT,
        nu_x: SoquetT,
        nu_y: SoquetT,
        nu_z: SoquetT,
        m: SoquetT,
        l: SoquetT,
        sys: SoquetT,
    ) -> Dict[str, 'SoquetT']:
        # ancilla for swaps from electronic system registers.
        # we assume these are left in a clean state after SELECT operations
        p = [bb.allocate(self.num_bits_p) for _ in range(3)]
        q = [bb.allocate(self.num_bits_p) for _ in range(3)]
        rl = bb.allocate(self.num_bits_nuc_pos)
        i, sys, p = bb.add(
            MultiplexedCSwap3D(self.num_bits_p, self.eta), sel=i, targets=sys, junk=p
        )
        j, sys, q = bb.add(
            MultiplexedCSwap3D(self.num_bits_p, self.eta), sel=j, targets=sys, junk=q
        )
        tuv, plus_t, w, r, s, p = bb.add(
            SelectTFirstQuantization(self.num_bits_p, self.eta),
            plus=plus_t,
            flag_T=tuv,
            w=w,
            r=r,
            s=s,
            p=p,
        )
        tuv, uv, l, rl, [nu_x, nu_y, nu_z], p, q = bb.add(
            SelectUVFirstQuantization(
                self.num_bits_p, self.eta, self.num_atoms, self.num_bits_nuc_pos
            ),
            flag_tuv=tuv,
            flag_uv=uv,
            l=l,
            rl=rl,
            nu=[nu_x, nu_y, nu_z],
            p=p,
            q=q,
        )
        i, sys, p = bb.add(
            MultiplexedCSwap3D(self.num_bits_p, self.eta), sel=i, targets=sys, junk=p
        )
        j, sys, q = bb.add(
            MultiplexedCSwap3D(self.num_bits_p, self.eta), sel=j, targets=sys, junk=q
        )
        _ = [bb.free(pi) for pi in p]
        _ = [bb.free(qi) for qi in q]
        bb.free(rl)
        return {
            'tuv': tuv,
            'uv': uv,
            'plus_t': plus_t,
            'i_ne_j': i_ne_j,
            'i': i,
            'j': j,
            'w': w,
            'r': r,
            's': s,
            'mu': mu,
            'nu_x': nu_x,
            'nu_y': nu_y,
            'nu_z': nu_z,
            'm': m,
            'l': l,
            'sys': sys,
        }


@bloq_example
def _prep_first_quant() -> PrepareFirstQuantizationWithProj:
    num_bits_p = 6
    num_bits_n = 8
    eta = 10
    num_atoms = 10
    lambda_zeta = 10
    prep_first_quant = PrepareFirstQuantizationWithProj(
        num_bits_p, num_bits_n, eta, num_atoms, lambda_zeta
    )

    return prep_first_quant


@bloq_example
def _sel_first_quant() -> SelectFirstQuantizationWithProj:
    num_bits_p = 6
    num_bits_n = 8
    eta = 10
    num_atoms = 10
    lambda_zeta = 10
    sel_first_quant = SelectFirstQuantizationWithProj(
        num_bits_p, num_bits_n, eta, num_atoms, lambda_zeta
    )

    return sel_first_quant


_FIRST_QUANTIZED_PREPARE_DOC = BloqDocSpec(
    bloq_cls=PrepareFirstQuantizationWithProj,
    import_line='from qualtran.bloqs.chemistry.pbc.first_quantization.projectile import FirstQuantizedPrepare',
    examples=(_prep_first_quant,),
)

_FIRST_QUANTIZED_PREPARE_DOC = BloqDocSpec(
    bloq_cls=SelectFirstQuantizationWithProj,
    import_line='from qualtran.bloqs.chemistry.pbc.first_quantization.projectile import FirstQuantizedSelect',
    examples=(_sel_first_quant,),
)
