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
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
from attrs import frozen
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
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_t import PrepareTFirstQuantization
from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_uv import PrepareUVFirstQuantization
from qualtran.bloqs.chemistry.pbc.first_quantization.select_t import SelectTFirstQuantization
from qualtran.bloqs.chemistry.pbc.first_quantization.select_uv import SelectUVFirstQuantization
from qualtran.bloqs.multiplexers.select_base import SelectOracle
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.bloqs.swap_network import MultiplexedCSwap
from qualtran.drawing import Text, TextBox, WireSymbol
from qualtran.symbolics import SymbolicFloat

if TYPE_CHECKING:
    from qualtran import Soquet
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@frozen
class PrepareTUVSuperpositions(Bloq):
    r"""Prepare the superposition over registers selecting between T U and V.

    This will produce 3 qubits for flagging which term to apply. xx0 -> T, x0x -> U or V, 0xx -> V.

    Note in reality this involves some state preparation and inequality testing.

    Registers:
        tuv: a single qubit rotated to appropriately weight T and U or V.
        uv: a single qubit rotated to appropriately weight U or V.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](https://arxiv.org/abs/2105.12767)
        page 15, section A
    """
    num_bits_t: int
    eta: int
    lambda_zeta: int
    num_bits_rot_aa: int = 8

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(tuv=1, uv=1)

    def pretty_name(self) -> str:
        return 'PREP TUV'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        n_eta_zeta = (self.eta + 2 * self.lambda_zeta - 1).bit_length()
        # The cost arises from rotating a qubit, and uniform state preparation
        # over eta + 2 lambda_zeta numbers along.
        return {Toffoli(): self.num_bits_t + 4 * n_eta_zeta + 2 * self.num_bits_rot_aa - 12}


@frozen
class UniformSuperpostionIJFirstQuantization(Bloq):
    r"""Uniform superposition over $\eta$ values of $i$ and $j$ in unary such that $i \ne j$.

    Args:
        eta: The number of electrons.
        num_bits_rot_aa: The number of bits of precision for the single qubit
            rotation for amplitude amplification. Called $b_r$ in the reference.

    Registers:
        i: a n_eta bit register for unary encoding of eta numbers.
        j: a n_eta bit register for unary encoding of eta numbers.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](https://arxiv.org/abs/2105.12767).
        page 18, section A, around Eq 62.
    """
    eta: int
    num_bits_rot_aa: int

    @cached_property
    def signature(self) -> Signature:
        n_eta = (self.eta - 1).bit_length()
        return Signature.build(i=n_eta, j=n_eta)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        n_eta = (self.eta - 1).bit_length()
        # Half of Eq. 62 which is the cost for prep and prep^\dagger
        return {Toffoli(): (7 * n_eta + 4 * self.num_bits_rot_aa - 18)}


@frozen
class MultiplexedCSwap3D(Bloq):
    """Wrapper around MultiplexedCSwap to avoid unsightly split/joins in diagrams."""

    num_bits_p: int
    eta: int

    @cached_property
    def signature(self) -> Signature:
        n_eta = (self.eta - 1).bit_length()
        return Signature(
            [
                Register('sel', BQUInt(bitsize=n_eta, iteration_length=self.eta)),
                Register('targets', QAny(bitsize=self.num_bits_p), shape=(self.eta, 3)),
                Register('junk', QAny(bitsize=self.num_bits_p), shape=(3,)),
            ]
        )

    @staticmethod
    def _reshape_reg(
        bb: BloqBuilder, in_reg: SoquetT, out_shape: Tuple[int, ...], bitsize: int
    ) -> NDArray[Soquet]:  # type: ignore[type-var]
        """Reshape registers allocated as a big register.

        Example:
            >>> xdim, ydim, bitsize = 2, 3, 8
            >>> junk = bb.allocate(xdim*ydim*bitsize)
            >>> out = _reshape_reg(junk, (xdim, ydim), bitsize)
            >>> assert out.shape == (xdim, ydim)
            >>> assert out[0,0].bitsize == bitsize
            >>> big_reg = _reshape_reg(out, (), xdim*ydim*bitsize)
            >>> assert out.bitsize == xdim*ydim*bitsize
        """
        # np.prod(()) returns a float (1.0), so take int
        size = int(np.prod(out_shape))
        if isinstance(in_reg, np.ndarray):
            # split an array of bitsize qubits into flat list of qubits
            split_qubits = bb.split(bb.join(np.concatenate([bb.split(x) for x in in_reg.ravel()])))
        else:
            split_qubits = bb.split(in_reg)
        merged_qubits = np.array(
            [bb.join(split_qubits[i * bitsize : (i + 1) * bitsize]) for i in range(size)]
        )
        return merged_qubits.reshape(out_shape)

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text(self.pretty_name())
        if reg.name == 'sel':
            return TextBox('In')
        elif reg.name == 'targets':
            return TextBox('×(x)')
        elif reg.name == 'junk':
            return TextBox('×(y)')
        raise ValueError(f'Unknown name: {reg.name}')

    def pretty_name(self) -> str:
        return 'MultiSwap'

    def build_composite_bloq(
        self, bb: BloqBuilder, sel: SoquetT, targets: SoquetT, junk: SoquetT
    ) -> Dict[str, 'SoquetT']:
        flat_sys = self._reshape_reg(bb, targets, (self.eta,), bitsize=3 * self.num_bits_p)
        flat_p = self._reshape_reg(bb, junk, (), bitsize=3 * self.num_bits_p)
        sel, flat_sys, flat_p = bb.add(
            MultiplexedCSwap(self.signature.get_left('sel'), target_bitsize=3 * self.num_bits_p),
            sel=sel,
            targets=flat_sys,
            output=flat_p,
        )
        targets = self._reshape_reg(bb, flat_sys, (self.eta, 3), bitsize=self.num_bits_p)
        junk = self._reshape_reg(bb, flat_p, (3,), bitsize=self.num_bits_p)
        return {'sel': sel, 'targets': targets, 'junk': junk}


@frozen
class PrepareFirstQuantization(PrepareOracle):
    r"""State preparation for the first quantized chemistry Hamiltonian.

    Prepares the state in Eq. 48 of the reference.

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
        sum_of_l1_coeffs: The one-norm of the Hamiltonian coefficients to
            prepare (often called lambda in the literature.)

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
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](https://arxiv.org/abs/2105.12767)
    """

    num_bits_p: int
    eta: int
    num_atoms: int
    lambda_zeta: int
    m_param: int = 2**8
    num_bits_nuc_pos: int = 16
    num_bits_t: int = 16
    num_bits_rot_aa: int = 8
    sum_of_l1_coeffs: Optional[SymbolicFloat] = None

    @property
    def selection_registers(self) -> Tuple[Register, ...]:
        n_nu = self.num_bits_p + 1
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
            Register('uv', BQUInt(bitsize=1, iteration_length=2)),
            Register('i', BQUInt(bitsize=n_eta, iteration_length=self.eta)),
            Register('j', BQUInt(bitsize=n_eta, iteration_length=self.eta)),
            Register("w", BQUInt(iteration_length=3, bitsize=2)),
            Register("r", BQUInt(bitsize=self.num_bits_p)),
            Register("s", BQUInt(bitsize=self.num_bits_p)),
            Register("mu", BQUInt(bitsize=self.num_bits_p)),
            Register("nu_x", BQUInt(bitsize=n_nu)),
            Register("nu_y", BQUInt(bitsize=n_nu)),
            Register("nu_z", BQUInt(bitsize=n_nu)),
            Register("m", BQUInt(bitsize=n_m)),
            Register("l", BQUInt(bitsize=n_at, iteration_length=n_at)),
        )

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return (Register("succ_nu", QBit()), Register("plus_t", QBit()))

    @property
    def l1_norm_coeffs(self) -> SymbolicFloat:
        if self.sum_of_l1_coeffs is None:
            raise ValueError(
                "sum_of_l1_coeffs not specified in PrepareFirstQuantization constructor."
            )
        return self.sum_of_l1_coeffs

    def pretty_name(self) -> str:
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
        r: SoquetT,
        s: SoquetT,
        mu: SoquetT,
        nu_x: Soquet,
        nu_y: Soquet,
        nu_z: Soquet,
        m: SoquetT,
        succ_nu: SoquetT,
        l: SoquetT,
    ) -> Dict[str, 'SoquetT']:
        tuv, uv = bb.add(
            PrepareTUVSuperpositions(
                self.num_bits_t, self.eta, self.lambda_zeta, self.num_bits_rot_aa
            ),
            tuv=tuv,
            uv=uv,
        )
        i, j = bb.add(
            UniformSuperpostionIJFirstQuantization(self.eta, self.num_bits_rot_aa), i=i, j=j
        )
        # # |+>
        # plus_t = bb.add(Hadamard(), q=plus_t)
        w, r, s = bb.add(
            PrepareTFirstQuantization(self.num_bits_p, self.eta, self.num_bits_rot_aa),
            w=w,
            r=r,
            s=s,
        )
        mu, [nu_x, nu_y, nu_z], m, l, succ_nu = bb.add(
            PrepareUVFirstQuantization(
                self.num_bits_p,
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
        return {
            'tuv': tuv,
            'uv': uv,
            'plus_t': plus_t,
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
            'succ_nu': succ_nu,
        }


@frozen
class SelectFirstQuantization(SelectOracle):
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
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](https://arxiv.org/abs/2105.12767)
    """

    num_bits_p: int
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
            Register("tuv", QBit()),
            Register("uv", QBit()),
            Register("i_ne_j", QBit()),
            Register("plus_t", QBit()),
        )

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        n_nu = self.num_bits_p + 1
        n_eta = (self.eta - 1).bit_length()
        n_at = (self.num_atoms - 1).bit_length()
        n_m = (self.m_param - 1).bit_length()
        return (
            Register('i', BQUInt(bitsize=n_eta, iteration_length=self.eta)),
            Register('j', BQUInt(bitsize=n_eta, iteration_length=self.eta)),
            Register("w", BQUInt(bitsize=3)),
            Register("r", BQUInt(bitsize=self.num_bits_p)),
            Register("s", BQUInt(bitsize=self.num_bits_p)),
            Register("mu", BQUInt(bitsize=self.num_bits_p)),
            Register("nu_x", BQUInt(bitsize=n_nu)),
            Register("nu_y", BQUInt(bitsize=n_nu)),
            Register("nu_z", BQUInt(bitsize=n_nu)),
            Register("m", BQUInt(bitsize=n_m)),
            Register("l", BQUInt(bitsize=n_at)),
        )

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register("sys", QAny(bitsize=self.num_bits_p), shape=(self.eta, 3)),)

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
        nu_x: Soquet,
        nu_y: Soquet,
        nu_z: Soquet,
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
        for pi in p:
            bb.free(pi)
        for qi in q:
            bb.free(qi)
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
def _prep_first_quant() -> PrepareFirstQuantization:
    num_bits_p = 6
    eta = 10
    num_atoms = 10
    lambda_zeta = 10
    prep_first_quant = PrepareFirstQuantization(num_bits_p, eta, num_atoms, lambda_zeta)

    return prep_first_quant


@bloq_example
def _sel_first_quant() -> SelectFirstQuantization:
    num_bits_p = 6
    eta = 10
    num_atoms = 10
    lambda_zeta = 10
    sel_first_quant = SelectFirstQuantization(num_bits_p, eta, num_atoms, lambda_zeta)

    return sel_first_quant


_FIRST_QUANTIZED_PREPARE_DOC = BloqDocSpec(
    bloq_cls=PrepareFirstQuantization, examples=(_prep_first_quant,)
)

_FIRST_QUANTIZED_SELECT_DOC = BloqDocSpec(
    bloq_cls=SelectFirstQuantization, examples=(_sel_first_quant,)
)
