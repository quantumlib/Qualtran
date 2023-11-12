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
from typing import Dict, Sequence, Set, TYPE_CHECKING

import cirq
import numpy as np
from attrs import frozen

from qualtran import Bloq, BloqBuilder, Register, Signature, SoquetT
from qualtran.bloqs.and_bloq import And
from qualtran.bloqs.basic_gates import TGate, XGate
from qualtran.bloqs.swap_network import CSwapApprox
from qualtran.cirq_interop import CirqGateAsBloq

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class THCRotations(Bloq):
    """Bloq for rotating into THC basis through Givens rotation network.

    This is accounting for In-data:rot and In-R in Fig. 7 of the THC paper (Ref.
    1). In practice this bloq is made up of a QROM load of the angles followed
    by controlled rotations. Equivalently it can be built from a modified
    version of the ProgrammableRotationGateArray from implemented in qualtran
    from Ref. 2. This is a placeholder waiting for an actual implementation.
    See https://github.com/quantumlib/Qualtran/issues/386.

    Args:
        num_bits_theta: Number of bits of precision for the rotation angles.
        rotation_angles: A list of rotation angles

    References:
        [Even more efficient quantum computations of chemistry through
            tensor hypercontraction](https://arxiv.org/pdf/2011.03494.pdf) Fig. 7.
        [Quantum computing enhanced computational catalysis]
        (https://arxiv.org/abs/2007.14460).
            Burg, Low et. al. 2021. Eq. 73
    """

    num_mu: int
    num_spin_orb: int
    num_bits_theta: int
    rotation_angles: Sequence[Sequence[int]]
    optimal_kr1: int = 1
    optimal_kr2: int = 1
    two_body_only: bool = False
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("eq_nu_mp1", bitsize=1),
                Register("data", bitsize=self.num_bits_theta),
                Register("sel", bitsize=self.num_mu.bit_length()),
                Register("trg", bitsize=self.num_spin_orb // 2),
            ]
        )

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f"In_mu-R{dag}"

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # from listings on page 17 of Ref. [1]
        num_data_sets = self.num_mu + self.num_spin_orb // 2
        if self.adjoint:
            if self.two_body_only:
                toff_cost_qrom = (
                    int(np.ceil(self.num_mu / self.optimal_kr1))
                    + int(np.ceil(self.num_spin_orb / (2 * self.optimal_kr1)))
                    + self.optimal_kr1
                )
            else:
                toff_cost_qrom = int(np.ceil(self.num_mu / self.optimal_kr2)) + self.optimal_kr2
        else:
            toff_cost_qrom = num_data_sets - 2
            if self.two_body_only:
                # we don't need the only body bit for the second application of the
                # rotations for the nu register.
                toff_cost_qrom -= self.num_spin_orb // 2
        # xref https://github.com/quantumlib/Qualtran/issues/370, the cost below
        # assume a phase gradient.
        rot_cost = self.num_spin_orb * (self.num_bits_theta - 2)
        return {(TGate(), 4 * (rot_cost + toff_cost_qrom))}


@frozen
class SelectTHC(Bloq):
    r"""SELECT for THC Hamilontian.

    Args:
        num_mu: THC auxiliary index dimension $M$
        num_spin_orb: number of spin orbitals $N$
        angles: list of num_mu + num_spin_orb / 2 lists of rotation angles each
            of length num_spin_orb//2. Each rotation angle should be a
            `num_bits_theta` fixed point approximation to the angle.

    Registers:
        mu: $\mu$ register.
        nu: $\nu$ register.
        theta: sign register.
        succ: success flag qubit from uniform state preparation
        eq_nu_mp1: flag for if $nu = M+1$
        plus_a / plus_b: plus state for controlled swaps on spins.
        sys_a / sys_b : System registers for (a)lpha/(b)eta orbitals.

    References:
        [Even more efficient quantum computations of chemistry through
            tensor hypercontraction](https://arxiv.org/pdf/2011.03494.pdf) Fig. 7.
    """

    num_mu: int
    num_spin_orb: int
    num_bits_theta: int
    rotation_angles: Sequence[Sequence[int]]

    @cached_property
    def signature(self) -> Signature:
        # Note we really need a range of num_mu + 1  for the mu register to
        # store the one-body hamiltonian too. Hence its num_mu.bit_length() not
        # (num_mu - 1).bit_length().
        return Signature(
            [
                Register("mu", bitsize=(self.num_mu).bit_length()),
                Register("nu", bitsize=(self.num_mu).bit_length()),
                Register("theta", bitsize=1),
                Register("succ", bitsize=1),
                Register("eq_nu_mp1", bitsize=1),
                Register("plus_a", bitsize=1),
                Register("plus_b", bitsize=1),
                Register("sys_a", bitsize=self.num_spin_orb // 2),
                Register("sys_b", bitsize=self.num_spin_orb // 2),
            ]
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', **regs: SoquetT) -> Dict[str, 'SoquetT']:
        # ancilla for and operation on |nu=M+1> and the |+> registers
        plus_anc = bb.allocate(1)
        eq_nu_mp1 = regs['eq_nu_mp1']
        mu, nu = regs['mu'], regs['nu']
        theta, succ = regs['theta'], regs['succ']
        plus_a, plus_b = regs['plus_a'], regs['plus_b']
        sys_a, sys_b = regs['sys_a'], regs['sys_b']
        # Decompose off-on-CSwap into And + CSwap
        [eq_nu_mp1, plus_anc], and_anc = bb.add(And(0, 1), ctrl=[eq_nu_mp1, plus_anc])
        num_bits_mu = self.signature[0].bitsize
        and_anc, mu, nu = bb.add(CSwapApprox(num_bits_mu), ctrl=and_anc, x=mu, y=nu)

        # System register spin swaps
        plus_b, sys_a, sys_b = bb.add(
            CSwapApprox(self.num_spin_orb // 2), ctrl=plus_b, x=sys_a, y=sys_b
        )

        # Rotations
        data = bb.allocate(self.num_bits_theta)
        eq_nu_mp1, data, mu, sys_a = bb.add(
            THCRotations(
                num_mu=self.num_mu,
                num_spin_orb=self.num_spin_orb,
                num_bits_theta=self.num_bits_theta,
                rotation_angles=self.rotation_angles,
            ),
            eq_nu_mp1=eq_nu_mp1,
            data=data,
            sel=mu,
            trg=sys_a,
        )
        # Controlled Z_0
        split_sys = bb.split(sys_a)
        succ, split_sys[0] = bb.add(CirqGateAsBloq(cirq.CZ), q=[succ, split_sys[0]])
        sys_a = bb.join(split_sys)
        # Undo rotations
        eq_nu_mp1, data, mu, sys_a = bb.add(
            THCRotations(
                num_mu=self.num_mu,
                num_spin_orb=self.num_spin_orb,
                num_bits_theta=self.num_bits_theta,
                rotation_angles=self.rotation_angles,
                adjoint=True,
            ),
            eq_nu_mp1=eq_nu_mp1,
            data=data,
            sel=mu,
            trg=sys_a,
        )

        # Clean up
        plus_b, sys_a, sys_b = bb.add(
            CSwapApprox(self.num_spin_orb // 2), ctrl=plus_b, x=sys_a, y=sys_b
        )

        plus_anc = bb.add(XGate(), q=plus_anc)

        # Swap spins
        # Should be a negative control..
        eq_nu_mp1, plus_a, plus_b = bb.add(CSwapApprox(1), ctrl=eq_nu_mp1, x=plus_a, y=plus_b)

        # Swap mu-nu
        eq_nu_mp1, mu, nu = bb.add(CSwapApprox(num_bits_mu), ctrl=eq_nu_mp1, x=mu, y=nu)

        # System register spin swaps
        plus_b, sys_a, sys_b = bb.add(
            CSwapApprox(self.num_spin_orb // 2), ctrl=plus_b, x=sys_a, y=sys_b
        )

        # Rotations
        eq_nu_mp1, data, nu, sys_a = bb.add(
            THCRotations(
                num_mu=self.num_mu,
                num_spin_orb=self.num_spin_orb,
                num_bits_theta=self.num_bits_theta,
                rotation_angles=self.rotation_angles,
                two_body_only=True,
            ),
            eq_nu_mp1=eq_nu_mp1,
            data=data,
            sel=nu,
            trg=sys_a,
        )
        # Controlled Z_0
        split_sys = bb.split(sys_a)
        succ, split_sys[0] = bb.add(CirqGateAsBloq(cirq.CZ), q=[succ, split_sys[0]])
        sys_a = bb.join(split_sys)
        # Undo rotations
        eq_nu_mp1, data, nu, sys_a = bb.add(
            THCRotations(
                num_mu=self.num_mu,
                num_spin_orb=self.num_spin_orb,
                num_bits_theta=self.num_bits_theta,
                rotation_angles=self.rotation_angles,
                two_body_only=True,
                adjoint=True,
            ),
            eq_nu_mp1=eq_nu_mp1,
            data=data,
            sel=nu,
            trg=sys_a,
        )

        # Clean up
        plus_b, sys_a, sys_b = bb.add(
            CSwapApprox(self.num_spin_orb // 2), ctrl=plus_b, x=sys_a, y=sys_b
        )

        # Undo the mu-nu swaps
        and_anc, mu, nu = bb.add(CSwapApprox(num_bits_mu), ctrl=and_anc, x=mu, y=nu)
        [eq_nu_mp1, plus_anc] = bb.add(
            And(0, 1, adjoint=True), ctrl=[eq_nu_mp1, plus_anc], target=and_anc
        )
        bb.free(data)
        bb.free(plus_anc)
        return {
            'mu': mu,
            'nu': nu,
            'theta': theta,
            'succ': succ,
            'eq_nu_mp1': eq_nu_mp1,
            'plus_a': plus_a,
            'plus_b': plus_b,
            'sys_a': sys_a,
            'sys_b': sys_b,
        }
