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
"""SELECT for the molecular tensor hypercontraction (THC) hamiltonian"""
from functools import cached_property
from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING

import cirq
import numpy as np
from attrs import frozen
from numpy.typing import NDArray

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
from qualtran.bloqs.basic_gates import CNOT, CSwap, Hadamard, Toffoli, XGate
from qualtran.bloqs.chemistry.black_boxes import ApplyControlledZs
from qualtran.bloqs.qrom import QROM
from qualtran.bloqs.select_and_prepare import SelectOracle
from qualtran.cirq_interop import CirqGateAsBloq
from qualtran.cirq_interop.bit_tools import iter_bits_fixed_point

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


def find_givens_angles(chi: NDArray[np.float64], num_bits_prec: int = 20) -> Tuple[Tuple[int]]:
    """Find rotation angles for givens rotations.

    Args:
        chi: THC leaf tensor of shape [num_spatial_orbs, num_mu]. Assumed that
            each column is individually normalized.
        num_bits_prec: The number of bits of precision with which to represent
            the angles as fixed point floats.

    Returns:
        thetas: An [num_mu, num_spatial_orbitals, num_bits_prec] array of rotation angles.

    References:
        https://arxiv.org/pdf/2007.14460.pdf Eq. 57
    """
    thetas = np.zeros(chi.T.shape, dtype=int)
    for mu, chi_mu in enumerate(chi.T):
        div_fac = 1.0
        for p, u_p in enumerate(chi_mu):
            rhs = u_p / (2 * div_fac)
            if abs(rhs) > 0.5:
                rhs = 0.5 * np.sign(rhs)
            theta_p = 0.5 * np.arccos(rhs)
            div_fac *= np.sin(2 * theta_p)
            theta_p_fp = ''.join(
                str(b) for b in iter_bits_fixed_point(theta_p, width=num_bits_prec)
            )
            thetas[mu, p] = int(theta_p_fp, 2)
    # for qrom we want to load num_spatial_orbs data sets for each mu
    return tuple(tuple(int(tm) for tm in tp) for tp in thetas.T)


@frozen
class InterleavedCliffordA(Bloq):
    r"""Clifford gates required to apply $e^{i \theta X_a Y_b}$ using just $R_z$ gates."""

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(a=1, b=1)

    def pretty_name(self) -> str:
        return "CA"

    def build_composite_bloq(self, bb: 'BloqBuilder', a: SoquetT, b: SoquetT):
        a = bb.add(Hadamard(), q=a)
        b = bb.add(CirqGateAsBloq(cirq.S**-1), q=b)
        b = bb.add(Hadamard(), q=b)
        a, b = bb.add(CNOT(), ctrl=a, target=b)
        return {'a': a, 'b': b}


@frozen
class InterleavedCliffordB(Bloq):
    r"""Clifford gates required to apply $e^{i \theta Y_a X_b}$ using just $R_z$ gates."""

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(a=1, b=1)

    def pretty_name(self) -> str:
        return "CB"

    def build_composite_bloq(self, bb: 'BloqBuilder', a: SoquetT, b: SoquetT):
        a = bb.add(CirqGateAsBloq(cirq.S**-1), q=a)
        a = bb.add(Hadamard(), q=a)
        b = bb.add(Hadamard(), q=b)
        a, b = bb.add(CNOT(), ctrl=a, target=b)
        return {'a': a, 'b': b}


@frozen
class PhaseGradientRz(Bloq):
    """Placeholder for Rz rotation using phase gradient register/gate.

    Args:
        bitsize: the number of bits of precision for the rotation angle.

    Regerence
        [Even more efficient quantum computations of chemistry through
            tensor hypercontraction](https://arxiv.org/pdf/2011.03494.pdf) Listing. 3, page 17.
    """

    bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(data=self.bitsize, target=1)

    def short_name(self) -> str:
        return r"$R_z(\theta)$"

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        rot_cost = self.bitsize - 2
        return {(Toffoli(), rot_cost)}


@frozen
class THCRotations(Bloq):
    r"""Bloq for rotating into THC basis through Givens rotation network.

    This is accounting for In-R in Fig. 7 of the THC paper (Ref.
    1).

    Args:
        num_mu: THC auxiliary index dimension $M$
        num_spin_orb: number of spin orbitals $N$
        num_bits_theta: Number of bits of precision for the rotations. Called
            $\beth$ in the reference.

    References:
        [Even more efficient quantum computations of chemistry through
            tensor hypercontraction](https://arxiv.org/pdf/2011.03494.pdf) Fig. 7.
        [Quantum computing enhanced computational catalysis](https://arxiv.org/abs/2007.14460).
            Burg, Low et. al. 2021. Eq. 73
    """

    num_mu: int
    num_spin_orb: int
    num_bits_theta: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("data", bitsize=self.num_bits_theta, shape=(self.num_spin_orb // 2,)),
                Register("target", bitsize=self.num_spin_orb // 2),
            ]
        )

    def pretty_name(self) -> str:
        return "In-Rot"

    def build_composite_bloq(
        self, bb: 'BloqBuilder', data: SoquetT, target: SoquetT
    ) -> Dict[str, 'SoquetT']:
        trg_bits = bb.split(target)
        for qi in range(self.num_spin_orb // 2 - 1):
            # X_i Y_{i+1} rotation
            trg_bits[qi], trg_bits[qi + 1] = bb.add(
                InterleavedCliffordA(), a=trg_bits[qi], b=trg_bits[qi + 1]
            )
            data[qi], trg_bits[qi + 1] = bb.add(
                PhaseGradientRz(self.num_bits_theta), data=data[qi], target=trg_bits[qi + 1]
            )
            # TODO: should be adjointed!
            trg_bits[qi], trg_bits[qi + 1] = bb.add(
                InterleavedCliffordA(), a=trg_bits[qi], b=trg_bits[qi + 1]
            )
            # Y_i X_{i+1} rotation
            trg_bits[qi], trg_bits[qi + 1] = bb.add(
                InterleavedCliffordB(), a=trg_bits[qi], b=trg_bits[qi + 1]
            )
            data[qi], trg_bits[qi + 1] = bb.add(
                PhaseGradientRz(self.num_bits_theta), data=data[qi], target=trg_bits[qi + 1]
            )
            # TODO: should be adjointed!
            trg_bits[qi], trg_bits[qi + 1] = bb.add(
                InterleavedCliffordB(), a=trg_bits[qi], b=trg_bits[qi + 1]
            )

        target = bb.join(trg_bits)
        return {'data': data, 'target': target}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # xref https://github.com/quantumlib/Qualtran/issues/370, the cost below
        # assume a phase gradient.
        # In Burg et al there are N//2-1 rotations not N//2.
        rot_cost = self.num_spin_orb * (self.num_bits_theta - 2)
        return {(Toffoli(), rot_cost)}


@frozen
class SelectTHC(SelectOracle):
    r"""SELECT for THC Hamiltonian.

    Args:
        num_mu: THC auxiliary index dimension $M$
        num_spin_orb: number of spin orbitals $N$
        num_bits_theta: Number of bits of precision for the rotations. Called
            $\beth$ in the reference.
        rotation_angles: Sequence of num_bits_theta-bit fixed-point
            representation of the basis rotation angles. For each mu thera
            num_spin_orb // 2 angles, which can be determined from
            find_givens_angles.
        kr1: block sizes for QROM erasure for outputting rotation angles. See Eq 34.
        kr2: block sizes for QROM erasure for outputting rotation angles. This
            is for the second QROM (eq 35)
        control_val: A control bit for the entire gate.

    Registers:
        succ: success flag qubit from uniform state preparation
        nu_eq_mp1: flag for if $nu = M+1$
        mu: $\mu$ register.
        nu: $\nu$ register.
        theta: sign register.
        plus_mn: Flag controlling swaps between mu and nu. Note that as per the
            Reference, the swaps are NOT performed as part of SELECT as they're
            acounted for during Prepare.
        plus_a / plus_b: plus state for controlled swaps on spins.
        sys_a / sys_b : System registers for (a)lpha/(b)eta orbitals.

    References:
        [Even more efficient quantum computations of chemistry through
            tensor hypercontraction](https://arxiv.org/pdf/2011.03494.pdf) Fig. 7.
    """

    num_mu: int
    num_spin_orb: int
    num_bits_theta: int
    rotation_angles: Tuple[Tuple[int]]
    kr1: int = 1
    kr2: int = 1
    control_val: Optional[int] = None

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return () if self.control_val is None else (Register('control', 1),)

    @cached_property
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        return (
            SelectionRegister("succ", bitsize=1),
            SelectionRegister("nu_eq_mp1", bitsize=1),
            SelectionRegister(
                "mu", bitsize=(self.num_mu).bit_length(), iteration_length=self.num_mu + 1
            ),
            SelectionRegister(
                "nu", bitsize=(self.num_mu).bit_length(), iteration_length=self.num_mu + 1
            ),
            SelectionRegister("plus_mn", bitsize=1),
            SelectionRegister("plus_a", bitsize=1),
            SelectionRegister("plus_b", bitsize=1),
        )

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (
            Register("sys_a", bitsize=self.num_spin_orb // 2),
            Register("sys_b", bitsize=self.num_spin_orb // 2),
        )

    def add_qrom_bloq(
        self,
        bb: 'BloqBuilder',
        mu: SoquetT,
        data: SoquetT,
        nu_eq_mp1: SoquetT,
        two_body_only: bool = False,
    ):
        """Helper to hide some boilerplate when adding a qrom with many target registers."""
        qrom = QROM(
            np.array(self.rotation_angles),
            selection_bitsizes=(self.num_mu.bit_length(),),
            target_bitsizes=(self.num_bits_theta,) * (self.num_spin_orb // 2),
            num_controls=0 if two_body_only else 1,
        )
        regs = {} if two_body_only else {'control': nu_eq_mp1}
        regs |= {'selection': mu}
        regs |= {f'target{i}': data[i] for i in range(self.num_spin_orb // 2)}
        regs = bb.add_d(qrom, **regs)
        mu = regs['selection']
        data = [regs[f'target{i}'] for i in range(self.num_spin_orb // 2)]
        if not two_body_only:
            nu_eq_mp1 = regs['control']
        return mu, data, nu_eq_mp1

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        succ: SoquetT,
        nu_eq_mp1: SoquetT,
        mu: SoquetT,
        nu: SoquetT,
        plus_mn: SoquetT,
        plus_a: SoquetT,
        plus_b: SoquetT,
        sys_a: SoquetT,
        sys_b: SoquetT,
    ) -> Dict[str, 'SoquetT']:
        plus_b, sys_a, sys_b = bb.add(CSwap(self.num_spin_orb // 2), ctrl=plus_b, x=sys_a, y=sys_b)

        # Allocate ancilla for qrom here due to https://github.com/quantumlib/Qualtran/issues/549
        data = np.array([bb.allocate(self.num_bits_theta) for _ in range(self.num_spin_orb // 2)])
        # 1. QROM the rotation angles.
        mu, data, nu_eq_mp1 = self.add_qrom_bloq(bb, mu, data, nu_eq_mp1)
        data, sys_a = bb.add(
            THCRotations(
                num_mu=self.num_mu,
                num_spin_orb=self.num_spin_orb,
                num_bits_theta=self.num_bits_theta,
            ),
            data=data,
            target=sys_a,
        )
        # Controlled Z_0
        (succ,), sys_b = bb.add(
            ApplyControlledZs(cvs=(1,), bitsize=self.num_spin_orb // 2), ctrls=(succ,), system=sys_b
        )
        # invert the rotations
        data, sys_a = bb.add(
            THCRotations(
                num_mu=self.num_mu,
                num_spin_orb=self.num_spin_orb,
                num_bits_theta=self.num_bits_theta,
            ),
            data=data,
            target=sys_a,
        )
        # erase qrom
        mu, data, nu_eq_mp1 = self.add_qrom_bloq(bb, mu, data, nu_eq_mp1)
        # undo spin swaps
        plus_b, sys_a, sys_b = bb.add(CSwap(self.num_spin_orb // 2), ctrl=plus_b, x=sys_a, y=sys_b)

        plus_mn = bb.add(XGate(), q=plus_mn)

        # Swap spins
        # Should be a negative control..
        nu_eq_mp1, plus_a, plus_b = bb.add(CSwap(1), ctrl=nu_eq_mp1, x=plus_a, y=plus_b)
        # swap mu / nu
        nu_eq_mp1, mu, nu = bb.add(CSwap(self.num_mu.bit_length()), ctrl=nu_eq_mp1, x=mu, y=nu)

        # System register spin swaps
        plus_b, sys_a, sys_b = bb.add(CSwap(self.num_spin_orb // 2), ctrl=plus_b, x=sys_a, y=sys_b)

        # Rotations
        mu, data, nu_eq_mp1 = self.add_qrom_bloq(bb, mu, data, nu_eq_mp1)
        data, sys_a = bb.add(
            THCRotations(
                num_mu=self.num_mu,
                num_spin_orb=self.num_spin_orb,
                num_bits_theta=self.num_bits_theta,
            ),
            data=data,
            target=sys_a,
        )
        # Controlled Z_0
        (succ, nu_eq_mp1), sys_b = bb.add(
            ApplyControlledZs(cvs=(1, 0), bitsize=self.num_spin_orb // 2),
            ctrls=(succ, nu_eq_mp1),
            system=sys_b,
        )
        # invert the rotations
        data, sys_a = bb.add(
            THCRotations(
                num_mu=self.num_mu,
                num_spin_orb=self.num_spin_orb,
                num_bits_theta=self.num_bits_theta,
            ),
            data=data,
            target=sys_a,
        )
        mu, data, nu_eq_mp1 = self.add_qrom_bloq(bb, mu, data, nu_eq_mp1)

        # Clean up
        plus_b, sys_a, sys_b = bb.add(CSwap(self.num_spin_orb // 2), ctrl=plus_b, x=sys_a, y=sys_b)

        # Undo the mu-nu swaps
        for d in data:
            bb.free(d)
        return {
            'succ': succ,
            'nu_eq_mp1': nu_eq_mp1,
            'mu': mu,
            'nu': nu,
            'plus_mn': plus_mn,
            'plus_a': plus_a,
            'plus_b': plus_b,
            'sys_a': sys_a,
            'sys_b': sys_b,
        }
        # Need build_call_graph here.
        # # from listings on page 17 of Ref. [1]
        # num_data_sets = self.num_mu + self.num_spin_orb // 2
        # if self.adjoint:
        #     if self.two_body_only:
        #         toff_cost_qrom = (
        #             int(np.ceil(self.num_mu / self.kr1))
        #             + int(np.ceil(self.num_spin_orb / (2 * self.kr1)))
        #             + self.kr1
        #         )
        #     else:
        #         toff_cost_qrom = int(np.ceil(self.num_mu / self.kr2)) + self.kr2
        # else:
        #     toff_cost_qrom = num_data_sets - 2
        #     if self.two_body_only:
        #         # we don't need the only body bit for the second application of the
        #         # rotations for the nu register.
        #         toff_cost_qrom -= self.num_spin_orb // 2


@bloq_example
def _thc_sel() -> SelectTHC:
    num_mu = 10
    num_spin_orb = 2 * 4
    num_bits_theta = 18
    chi = np.random.random(size=(num_spin_orb // 2, num_mu))
    # should be individually normalized
    norms = np.sqrt(np.einsum("pm,pm->m", chi, chi))
    chi = np.einsum("pm,m->pm", chi, norms)
    rotation_angles = find_givens_angles(chi, num_bits_prec=num_bits_theta)
    thc_sel = SelectTHC(
        num_mu=num_mu,
        num_spin_orb=num_spin_orb,
        num_bits_theta=num_bits_theta,
        rotation_angles=rotation_angles,
    )
    return thc_sel


_THC_SELECT = BloqDocSpec(
    bloq_cls=SelectTHC,
    import_line='from qualtran.bloqs.chemistry.thc.select_bloq import find_givens_angles, SelectTHC',
    examples=(_thc_sel,),
)


@bloq_example
def _thc_rotations() -> THCRotations:
    num_mu = 10
    num_spin_orb = 2 * 4
    num_bits_theta = 20
    thc_rotations = THCRotations(
        num_mu=num_mu, num_spin_orb=num_spin_orb, num_bits_theta=num_bits_theta
    )
    return thc_rotations


_THC_ROTATIONS = BloqDocSpec(
    bloq_cls=THCRotations,
    import_line='from qualtran.bloqs.chemistry.thc.select_bloq import THCRotations',
    examples=(_thc_rotations,),
)
