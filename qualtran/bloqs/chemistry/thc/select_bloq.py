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
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
from attrs import evolve, frozen

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
    SoquetT,
)
from qualtran._infra.single_qubit_controlled import SpecializedSingleQubitControlledExtension
from qualtran.bloqs.basic_gates import CSwap, Toffoli, XGate
from qualtran.bloqs.chemistry.black_boxes import ApplyControlledZs
from qualtran.bloqs.multiplexers.select_base import SelectOracle

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@frozen
class THCRotations(Bloq):
    r"""Bloq for rotating into THC basis through Givens rotation network.

    This is accounting for In-data:rot and In-R in Fig. 7 of the THC paper (Ref.
    1). In practice this bloq is made up of a QROM load of the angles followed
    by controlled rotations. Equivalently it can be built from a modified
    version of the ProgrammableRotationGateArray from implemented in qualtran
    from Ref. 2. This is a placeholder waiting for an actual implementation.
    See https://github.com/quantumlib/Qualtran/issues/386.

    Args:
        num_mu: THC auxiliary index dimension $M$
        num_spin_orb: number of spin orbitals $N$
        num_bits_theta: Number of bits of precision for the rotations. Called
            $\beth$ in the reference.
        kr1: block sizes for QROM erasure for outputting rotation angles. See Eq 34.
        kr2: block sizes for QROM erasure for outputting rotation angles. This
            is for the second QROM (eq 35)
        two_body_only: Whether to only apply the two body Hamiltonian. This reduces the QROM size.
        is_adjoint: Whether to dagger this bloq or not.

    References:
        [Even more efficient quantum computations of chemistry through
            tensor hypercontraction](https://arxiv.org/pdf/2011.03494.pdf) Fig. 7.
        [Quantum computing enhanced computational catalysis](https://arxiv.org/abs/2007.14460).
            Burg, Low et. al. 2021. Eq. 73
    """

    num_mu: int
    num_spin_orb: int
    num_bits_theta: int
    kr1: int = 1
    kr2: int = 1
    two_body_only: bool = False
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("nu_eq_mp1", QBit()),
                Register("data", QAny(bitsize=self.num_bits_theta)),
                Register("sel", QAny(bitsize=self.num_mu.bit_length())),
                Register("trg", QAny(bitsize=self.num_spin_orb // 2)),
            ]
        )

    def adjoint(self) -> 'Bloq':
        return evolve(self, is_adjoint=not self.is_adjoint)

    def __str__(self) -> str:
        dag = '†' if self.is_adjoint else ''
        return f"In_mu-R{dag}"

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # from listings on page 17 of Ref. [1]
        num_data_sets = self.num_mu + self.num_spin_orb // 2
        if self.is_adjoint:
            if self.two_body_only:
                toff_cost_qrom = (
                    int(np.ceil(self.num_mu / self.kr1))
                    + int(np.ceil(self.num_spin_orb / (2 * self.kr1)))
                    + self.kr1
                )
            else:
                toff_cost_qrom = int(np.ceil(self.num_mu / self.kr2)) + self.kr2
        else:
            toff_cost_qrom = num_data_sets - 2
            if self.two_body_only:
                # we don't need the only body bit for the second application of the
                # rotations for the nu register.
                toff_cost_qrom -= self.num_spin_orb // 2
        # xref https://github.com/quantumlib/Qualtran/issues/370, the cost below
        # assume a phase gradient.
        rot_cost = self.num_spin_orb * (self.num_bits_theta - 2)
        return {Toffoli(): (rot_cost + toff_cost_qrom)}


@frozen
class SelectTHC(SpecializedSingleQubitControlledExtension, SelectOracle):  # type: ignore[misc]
    r"""SELECT for THC Hamiltonian.

    Args:
        num_mu: THC auxiliary index dimension $M$
        num_spin_orb: number of spin orbitals $N$
        num_bits_theta: Number of bits of precision for the rotations. Called
            $\beth$ in the reference.
        kr1: block sizes for QROM erasure for outputting rotation angles. See Eq 34.
        kr2: block sizes for QROM erasure for outputting rotation angles. This
            is for the second QROM (eq 35)
        control_val: A control bit for the entire gate.
        keep_bitsize: number of bits for keep register for coherent alias
        sampling. This can be determined from the PrepareTHC bloq. See
        https://github.com/quantumlib/Qualtran/issues/549

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
    keep_bitsize: int
    kr1: int = 1
    kr2: int = 1
    control_val: Optional[int] = None

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return () if self.control_val is None else (Register('control', QBit()),)

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (
            Register("succ", BQUInt(bitsize=1)),
            Register("nu_eq_mp1", BQUInt(bitsize=1)),
            Register(
                "mu", BQUInt(bitsize=(self.num_mu).bit_length(), iteration_length=self.num_mu + 1)
            ),
            Register(
                "nu", BQUInt(bitsize=(self.num_mu).bit_length(), iteration_length=self.num_mu + 1)
            ),
            Register("plus_mn", BQUInt(bitsize=1)),
            Register("plus_a", BQUInt(bitsize=1)),
            Register("plus_b", BQUInt(bitsize=1)),
            Register("sigma", BQUInt(bitsize=self.keep_bitsize)),
            Register("rot", BQUInt(bitsize=1)),
        )

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (
            Register("sys_a", QAny(bitsize=self.num_spin_orb // 2)),
            Register("sys_b", QAny(bitsize=self.num_spin_orb // 2)),
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        succ = soqs['succ']
        nu_eq_mp1 = soqs['nu_eq_mp1']
        mu = soqs['mu']
        nu = soqs['nu']
        plus_mn = soqs['plus_mn']
        plus_a = soqs['plus_a']
        plus_b = soqs['plus_b']
        sigma = soqs['sigma']
        rot = soqs['rot']
        sys_a = soqs['sys_a']
        sys_b = soqs['sys_b']
        plus_b, sys_a, sys_b = bb.add(CSwap(self.num_spin_orb // 2), ctrl=plus_b, x=sys_a, y=sys_b)
        # Rotations
        data = bb.allocate(self.num_bits_theta)
        nu_eq_mp1, data, mu, sys_a = bb.add(
            THCRotations(
                num_mu=self.num_mu,
                num_spin_orb=self.num_spin_orb,
                num_bits_theta=self.num_bits_theta,
                kr1=self.kr1,
                kr2=self.kr2,
            ),
            nu_eq_mp1=nu_eq_mp1,
            data=data,
            sel=mu,
            trg=sys_a,
        )
        # Controlled Z_0
        (succ,), sys_b = bb.add(
            ApplyControlledZs(cvs=(1,), bitsize=self.num_spin_orb // 2),
            ctrls=np.asarray([succ]),
            system=sys_b,
        )
        # Undo rotations
        nu_eq_mp1, data, mu, sys_a = bb.add(
            THCRotations(
                num_mu=self.num_mu,
                num_spin_orb=self.num_spin_orb,
                num_bits_theta=self.num_bits_theta,
                kr1=self.kr1,
                kr2=self.kr2,
                is_adjoint=True,
            ),
            nu_eq_mp1=nu_eq_mp1,
            data=data,
            sel=mu,
            trg=sys_a,
        )
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
        nu_eq_mp1, data, mu, sys_a = bb.add(
            THCRotations(
                num_mu=self.num_mu,
                num_spin_orb=self.num_spin_orb,
                num_bits_theta=self.num_bits_theta,
                kr1=self.kr1,
                kr2=self.kr2,
                two_body_only=True,
            ),
            nu_eq_mp1=nu_eq_mp1,
            data=data,
            sel=mu,
            trg=sys_a,
        )
        # Controlled Z_0
        (succ, nu_eq_mp1), sys_b = bb.add(
            ApplyControlledZs(cvs=(1, 0), bitsize=self.num_spin_orb // 2),
            ctrls=(succ, nu_eq_mp1),
            system=sys_b,
        )
        # Undo rotations
        nu_eq_mp1, data, mu, sys_a = bb.add(
            THCRotations(
                num_mu=self.num_mu,
                num_spin_orb=self.num_spin_orb,
                num_bits_theta=self.num_bits_theta,
                kr1=self.kr1,
                kr2=self.kr2,
                two_body_only=True,
                is_adjoint=True,
            ),
            nu_eq_mp1=nu_eq_mp1,
            data=data,
            sel=mu,
            trg=sys_a,
        )

        # Clean up
        plus_b, sys_a, sys_b = bb.add(CSwap(self.num_spin_orb // 2), ctrl=plus_b, x=sys_a, y=sys_b)

        # Undo the mu-nu swaps
        bb.free(data)
        out_soqs = {
            'succ': succ,
            'nu_eq_mp1': nu_eq_mp1,
            'mu': mu,
            'nu': nu,
            'plus_mn': plus_mn,
            'plus_a': plus_a,
            'plus_b': plus_b,
            'sigma': sigma,
            'rot': rot,
            'sys_a': sys_a,
            'sys_b': sys_b,
        }
        if self.control_val is not None:
            out_soqs['control'] = soqs['control']

        return out_soqs


@bloq_example
def _thc_sel() -> SelectTHC:
    num_mu = 8
    num_mu = 10
    num_spin_orb = 2 * 4
    thc_sel = SelectTHC(
        num_mu=num_mu, num_spin_orb=num_spin_orb, num_bits_theta=12, keep_bitsize=10
    )
    return thc_sel


_THC_SELECT = BloqDocSpec(bloq_cls=SelectTHC, examples=(_thc_sel,))
