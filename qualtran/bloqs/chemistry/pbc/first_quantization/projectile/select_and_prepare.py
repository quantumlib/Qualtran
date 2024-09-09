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
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
from attrs import evolve, field, frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    BQUInt,
    QAny,
    QBit,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.basic_gates import CSwap, Toffoli
from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_t import (
    PrepareTFirstQuantizationWithProj,
)
from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_uv import (
    PrepareUVFirstQuantizationWithProj,
)
from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_t import (
    SelectTFirstQuantizationWithProj,
)
from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_uv import (
    SelectUVFirstQuantizationWithProj,
)
from qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare import (
    MultiplexedCSwap3D,
    UniformSuperpostionIJFirstQuantization,
)
from qualtran.bloqs.multiplexers.select_base import SelectOracle
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.bloqs.swap_network import MultiplexedCSwap
from qualtran.drawing import Circle, Text, TextBox, WireSymbol

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@frozen
class PrepareTUVSuperpositions(Bloq):
    r"""Prepare the superposition over registers selecting between T U and V.

    Args:
        num_bits_t: The number of bits of precision for the state preparation
            over the register selecting between the different components of the
            Hamiltonian.
        eta: The number of electrons.
        lambda_zeta: sum of nuclear charges.
        num_bits_rot_aa: The number of bits of precision for the rotation for
            amplitude amplification.

    Registers:
        tuv: Register to prepare to select between T or UV.
        tepm: Register to prepare to select between (e)lectron, (p)rojectile, or (m)ean terms.
        uv: Register to prepare to select between coulombic terms.
        flags: Flag register signalling which of the terms to apply. This is not
            a complete picture and we only produce 4 flag qubits to flag
            kinetic, kinetic (mean-projectile), UV, and projectile only.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](https://arxiv.org/abs/2105.12767)
        page 15, section A
    """
    num_bits_t: int
    eta: int
    lambda_zeta: int
    num_bits_rot_aa: int = 8
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('tuv', QBit()),
                Register('tepm', QAny(bitsize=2)),
                Register('uv', QAny(bitsize=2)),
                Register('flags', QBit(), shape=(4,)),
            ]
        )

    def adjoint(self) -> 'Bloq':
        return evolve(self, is_adjoint=not self.is_adjoint)

    def pretty_name(self) -> str:
        return 'PREP TUV'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        if self.is_adjoint:
            # inverting inequality tests at zero Toffoli.
            return {}
        else:
            return {Toffoli(): 6 * self.num_bits_t + 2}


@frozen
class ControlledMultiplexedCSwap3D(MultiplexedCSwap3D):
    """Controlled Multiplexed swap network.

    Handles case of 3D registers and padding this register with zeros.
    """

    num_bits_p: int
    num_bits_n: int
    eta: int
    cvs: Tuple[int, ...] = field(converter=lambda v: (v,) if isinstance(v, int) else tuple(v))

    @cached_property
    def signature(self) -> Signature:
        n_eta = (self.eta - 1).bit_length()
        return Signature(
            [
                Register('ctrl', QBit(), shape=(len(self.cvs),)),
                Register('sel', BQUInt(bitsize=n_eta, iteration_length=self.eta)),
                Register('targets', QAny(bitsize=self.num_bits_p), shape=(self.eta, 3)),
                Register('junk', QAny(bitsize=self.num_bits_n), shape=(3,)),
            ]
        )

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text(self.pretty_name())
        if reg.name == 'sel':
            return TextBox('In')
        elif reg.name == 'targets':
            return TextBox('×(x)')
        elif reg.name == 'junk':
            return TextBox('×(y)')
        elif reg.name == 'ctrl':
            (c_idx,) = idx
            filled = bool(self.cvs[c_idx])
            return Circle(filled)
        raise ValueError(f'Unknown name: {reg.name}')

    def build_composite_bloq(
        self, bb: BloqBuilder, ctrl: SoquetT, sel: SoquetT, targets: SoquetT, junk: SoquetT
    ) -> Dict[str, 'SoquetT']:
        flat_sys = self._reshape_reg(bb, targets, (self.eta,), bitsize=3 * self.num_bits_p)
        # we need to extract first n_p bits of each n_n sized ancilla register (i.e. pad with zeros).
        # This is not a contiguous chunk of qubits so we need to first flatten,
        # extract and reshape to swap into these qubits.
        # (3,), n_n ->  (3, n_n), 1
        junk = self._reshape_reg(bb, junk, (3, self.num_bits_n), bitsize=1)
        elec_reg = []
        # (3,n_n), 1 ->  (3,), n_p
        for xyz in range(3):
            elec_reg.append(bb.join(junk[xyz][: self.num_bits_p]))
        # (3,), n_p ->  (,), 3 * n_p
        flat_p = self._reshape_reg(bb, np.array(elec_reg), (), bitsize=3 * self.num_bits_p)
        ctrl, sel, flat_sys, flat_p = bb.add(
            MultiplexedCSwap(
                self.signature.get_left('sel'),
                target_bitsize=3 * self.num_bits_p,
                control_regs=self.signature.get_left('ctrl'),
            ),
            ctrl=ctrl,
            sel=sel,
            targets=flat_sys,
            output=flat_p,
        )
        elec_reg = self._reshape_reg(bb, flat_p, (3,), bitsize=self.num_bits_p)
        for xyz in range(3):
            junk[xyz, : self.num_bits_p] = bb.split(elec_reg[xyz])
        targets = self._reshape_reg(bb, flat_sys, (self.eta, 3), bitsize=self.num_bits_p)
        junk = self._reshape_reg(bb, junk, (3,), bitsize=self.num_bits_n)
        return {'ctrl': ctrl, 'sel': sel, 'targets': targets, 'junk': junk}


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
        is_adjoint: Whether to dagger the bloq or not.

    Registers:
        tuv: Register for preparing superposition for selecting between kinetic
            and potential terms in the Hamiltonian.
        tepm: Register to prepare to select between (e)lectron, (p)rojectile, or (m)ean terms.
        uv: Register to prepare to select between coulombic terms.
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
        succ_nu: A flag to indiciate the success of the $\nu$ state preparation.
        plus_t: A flag qubit prepared in the $|+\rangle$ state.
        flags: A 4 qubit flag register indicating which component of the Hamiltonian to apply.

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

    @property
    def selection_registers(self) -> Tuple[Register, ...]:
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
            Register('tuv', BQUInt(bitsize=1, iteration_length=2)),
            Register('tepm', BQUInt(bitsize=2, iteration_length=3)),
            Register('uv', BQUInt(bitsize=2, iteration_length=4)),
            Register('i', BQUInt(bitsize=n_eta, iteration_length=self.eta)),
            Register('j', BQUInt(bitsize=n_eta, iteration_length=self.eta)),
            Register("w", BQUInt(iteration_length=3, bitsize=2)),
            Register("w_mean", BQUInt(iteration_length=3, bitsize=2)),
            Register("r", BQUInt(bitsize=self.num_bits_n)),
            Register("s", BQUInt(bitsize=self.num_bits_n)),
            Register("mu", BQUInt(bitsize=self.num_bits_n)),
            Register("nu_x", BQUInt(bitsize=n_nu)),
            Register("nu_y", BQUInt(bitsize=n_nu)),
            Register("nu_z", BQUInt(bitsize=n_nu)),
            Register("m", BQUInt(bitsize=n_m)),
            Register("l", BQUInt(bitsize=n_at)),
        )

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return (
            Register("succ_nu", QBit()),
            Register("plus_t", QBit()),
            Register('flags', QBit(), shape=(4,)),
        )

    def pretty_name(self) -> str:
        return r'PREP'

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        tuv: SoquetT,
        tepm: SoquetT,
        uv: SoquetT,
        plus_t: SoquetT,
        i: SoquetT,
        j: SoquetT,
        w: SoquetT,
        w_mean: SoquetT,
        r: SoquetT,
        s: SoquetT,
        mu: SoquetT,
        nu_x: Soquet,
        nu_y: Soquet,
        nu_z: Soquet,
        m: SoquetT,
        succ_nu: SoquetT,
        l: SoquetT,
        flags: SoquetT,
    ) -> Dict[str, 'SoquetT']:
        prep_tuv = PrepareTUVSuperpositions(
            self.num_bits_t, self.eta, self.lambda_zeta, self.num_bits_rot_aa
        )
        tuv, tepm, uv, flags = bb.add(prep_tuv, tuv=tuv, tepm=tepm, uv=uv, flags=flags)
        i, j = bb.add(
            UniformSuperpostionIJFirstQuantization(self.eta, self.num_bits_rot_aa), i=i, j=j
        )
        # |+>
        # plus_t = bb.add(Hadamard(), q=plus_t)
        w, w_mean, r, s = bb.add(
            PrepareTFirstQuantizationWithProj(
                self.num_bits_p, self.num_bits_n, self.eta, self.num_bits_rot_aa
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
            ),
            mu=mu,
            nu=[nu_x, nu_y, nu_z],
            m=m,
            l=l,
            flag_nu=succ_nu,
        )
        out_flags = {
            'tuv': tuv,
            'tepm': tepm,
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
            'flags': flags,
        }
        return out_flags


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

    Registers:
        ham_ctrl: Control bits flagging which component of the Hamiltonian to apply.
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
            components of size num_bits_p.
        proj: The system register. Will store a single register (x, y and z)
            components of size num_bits_n.

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

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return (
            # flags for which component of Hamiltonian to apply.
            Register("ham_ctrl", QBit(), shape=(4,)),
            Register("i_ne_j", QBit()),
            Register("plus_t", QBit()),
        )

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        n_nu = self.num_bits_n + 1
        n_eta = (self.eta - 1).bit_length()
        n_at = (self.num_atoms - 1).bit_length()
        n_m = (self.m_param - 1).bit_length()
        return (
            Register('i', BQUInt(bitsize=n_eta, iteration_length=self.eta)),
            Register('j', BQUInt(bitsize=n_eta, iteration_length=self.eta)),
            Register("w", BQUInt(bitsize=3)),
            Register("w_mean", BQUInt(bitsize=3)),
            Register("r", BQUInt(bitsize=self.num_bits_n)),
            Register("s", BQUInt(bitsize=self.num_bits_n)),
            Register("mu", BQUInt(bitsize=self.num_bits_n)),
            Register("nu_x", BQUInt(bitsize=n_nu)),
            Register("nu_y", BQUInt(bitsize=n_nu)),
            Register("nu_z", BQUInt(bitsize=n_nu)),
            Register("m", BQUInt(bitsize=n_m)),
            Register("l", BQUInt(bitsize=n_at)),
        )

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (
            Register("sys", QAny(bitsize=self.num_bits_p), shape=(self.eta, 3)),
            Register('proj', QAny(bitsize=self.num_bits_n), shape=(3,)),
        )

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )

    def pretty_name(self) -> str:
        return r'SELECT'

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        ham_ctrl: NDArray[Soquet],  # type: ignore[type-var]
        i_ne_j: SoquetT,
        plus_t: SoquetT,
        i: SoquetT,
        j: SoquetT,
        w: SoquetT,
        w_mean: SoquetT,
        r: SoquetT,
        s: SoquetT,
        mu: SoquetT,
        nu_x: Soquet,
        nu_y: Soquet,
        nu_z: Soquet,
        m: SoquetT,
        l: SoquetT,
        sys: SoquetT,
        proj: NDArray[Soquet],  # type: ignore[type-var]
    ) -> Dict[str, 'SoquetT']:
        # ancilla for swaps from electronic and projectile system registers.
        # we assume these are left in a clean state after SELECT operations
        # We only need one of the ancilla registers to be of the size of the projectile's register.
        p = [bb.allocate(self.num_bits_n) for _ in range(3)]
        q = [bb.allocate(self.num_bits_p) for _ in range(3)]
        rl = bb.allocate(self.num_bits_nuc_pos)
        # flags for selecting different components of the Hamiltonian.
        # TODO: This is not very pretty, maybe just add as we need
        flag_t, flag_t_mean, flag_uv, flag_proj = ham_ctrl
        # negative control on h_proj control (i.e. swap electronic registers if h_proj is off.)
        [flag_proj], i, sys, p = bb.add(
            ControlledMultiplexedCSwap3D(self.num_bits_p, self.num_bits_n, self.eta, cvs=(0,)),
            ctrl=[flag_proj],
            sel=i,
            targets=sys,
            junk=p,
        )
        # swap projectile register if h_proj is on
        for xyz in range(3):
            flag_proj, proj[xyz], p[xyz] = bb.add(
                CSwap(self.num_bits_n), ctrl=flag_proj, x=proj[xyz], y=p[xyz]
            )
        # Always swap electron j to ancilla
        j, sys, q = bb.add(
            MultiplexedCSwap3D(self.num_bits_p, self.eta), sel=j, targets=sys, junk=q
        )
        flag_t, flag_t_mean, plus_t, w, w_mean, r, s, p = bb.add(
            SelectTFirstQuantizationWithProj(self.num_bits_n, self.eta),
            flag_T=flag_t,
            flag_mean=flag_t_mean,
            plus=plus_t,
            w=w,
            w_mean=w_mean,
            r=r,
            s=s,
            p=p,
        )
        flag_t, flag_uv, l, rl, [nu_x, nu_y, nu_z], p, q = bb.add(
            SelectUVFirstQuantizationWithProj(
                self.num_bits_p, self.num_bits_n, self.eta, self.num_atoms, self.num_bits_nuc_pos
            ),
            flag_tuv=flag_t,
            flag_uv=flag_uv,
            l=l,
            rl=rl,
            nu=[nu_x, nu_y, nu_z],
            p=p,
            q=q,
        )
        [flag_proj], i, sys, p = bb.add(
            ControlledMultiplexedCSwap3D(self.num_bits_p, self.num_bits_n, self.eta, cvs=(0,)),
            ctrl=[flag_proj],
            sel=i,
            targets=sys,
            junk=p,
        )
        for xyz in range(3):
            flag_proj, proj[xyz], p[xyz] = bb.add(
                CSwap(self.num_bits_n), ctrl=flag_proj, x=proj[xyz], y=p[xyz]
            )
        j, sys, q = bb.add(
            MultiplexedCSwap3D(self.num_bits_p, self.eta), sel=j, targets=sys, junk=q
        )
        for pi in p:
            bb.free(pi)
        for qi in q:
            bb.free(qi)
        ham_ctrl[:] = flag_t, flag_t_mean, flag_uv, flag_proj
        bb.free(rl)
        return {
            'ham_ctrl': ham_ctrl,
            'i_ne_j': i_ne_j,
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
            'sys': sys,
            'proj': proj,
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


_FIRST_QUANTIZED_WITH_PROJ_PREPARE_DOC = BloqDocSpec(
    bloq_cls=PrepareFirstQuantizationWithProj, examples=(_prep_first_quant,)
)

_FIRST_QUANTIZED_WITH_PROJ_SELECT_DOC = BloqDocSpec(
    bloq_cls=SelectFirstQuantizationWithProj, examples=(_sel_first_quant,)
)
