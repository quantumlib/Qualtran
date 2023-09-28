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
from typing import Dict, Sequence

import cirq
import cirq_ft
import numpy as np
from attrs import frozen
from cirq_ft.algos.programmable_rotation_gate_array import ProgrammableRotationGateArrayBase
from cirq_ft.infra import GateWithRegisters
from numpy.typing import NDArray

from qualtran import Bloq, BloqBuilder, Register, Signature, SoquetT
from qualtran.bloqs.and_bloq import And
from qualtran.bloqs.basic_gates import XGate
from qualtran.bloqs.swap_network import CSwapApprox


def find_givens_angles(chi: NDArray[float]):
    """Find rotation angles for givens rotations.

    Args:
        chi: THC leaf tensor of shape [num_spatial_orbs, num_mu]. Assumed that
            each column is individually normalized.
    Returns:
        thetas: An [num_mu, num_spatial_orbitals] array of rotation angles.

    References:
        https://arxiv.org/pdf/2007.14460.pdf Eq. 57
    """
    # verify Eq 55. Build Vs using these thetas, Build U, and then compute
    # thetas again check they match.
    thetas = np.zeros(chi.T.shape)
    for mu, chi_mu in enumerate(chi.T):
        div_fac = 1.0
        for p, u_p in enumerate(chi_mu):
            rhs = u_p / (2 * div_fac)
            if abs(rhs) > 0.5:
                rhs = 0.5 * np.sign(rhs)
            theta_p = 0.5 * np.arccos(rhs)
            div_fac *= np.sin(2 * theta_p)
            thetas[mu, p] = theta_p
    return thetas


# class THCRotationNetwork(cirq_ft.infra.GateWithRegistersBase):
#     """Programmable Rotation Gate Array for THC SELECT rotations.

#     References:
#         Fig 73.
#         [Quantum computing enhanced computational catalysis]
#         (https://arxiv.org/abs/2007.14460).
#             Burg, Low et. al. 2021.
#     """

#     def __init__(self, *angles: Sequence[int], kappa: int, rotation_gate: cirq.Gate):
#         super().__init__(*angles, kappa=kappa, rotation_gate=rotation_gate)

#     def interleaved_unitary(
#         self, index: int, inverse: bool = False, **qubit_regs: NDArray[cirq.Qid]
#     ) -> cirq.Operation:
#         C0 = [
#             cirq.H(qubit_regs['target'][index]),
#             cirq.S(qubit_regs['target'][index + 1]) ** -1,
#             cirq.H(qubit_regs['target'][index + 1]),
#             cirq.CNOT(qubit_regs['target'][index], qubit_regs['target'][index + 1]),
#         ]
#         C1 = [
#             cirq.S(qubit_regs['target'][index]),
#             cirq.H(qubit_regs['target'][index]),
#             cirq.H(qubit_regs['target'][index + 1]),
#             cirq.CNOT(qubit_regs['target'][index], qubit_regs['target'][index + 1]),
#         ]
#         two_qubit_ops_factory = [C0, C1]
#         if inverse:
#             return cirq.inverse(two_qubit_ops_factory[index % 2])
#         else:
#             return two_qubit_ops_factory[index % 2]

#     def decompose_from_registers(
#         self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
#     ) -> cirq.OP_TREE:
#         selection, kappa_load_target = quregs.pop('selection'), quregs.pop('kappa_load_target')
#         rotations_target = quregs.pop('rotations_target')
#         interleaved_unitary_target = quregs

#         # 1. Find a convenient way to process batches of size kappa.
#         num_bits = sum(max(thetas).bit_length() for thetas in self.angles)
#         iteration_length = self.selection_registers[0].iteration_length
#         selection_bitsizes = [s.total_bits() for s in self.selection_registers]
#         angles_bits = np.zeros(shape=(iteration_length, num_bits), dtype=int)
#         angles_bit_pow = np.zeros(shape=(num_bits,), dtype=int)
#         angles_idx = np.zeros(shape=(num_bits,), dtype=int)
#         st, en = 0, 0
#         for i, thetas in enumerate(self.angles):
#             bit_width = max(thetas).bit_length()
#             st, en = en, en + bit_width
#             angles_bits[:, st:en] = [[*iter_bits(t, bit_width)] for t in thetas]
#             angles_bit_pow[st:en] = [*range(bit_width)][::-1]
#             angles_idx[st:en] = i
#         assert en == num_bits
#         # 2. Process batches of size kappa.
#         power_of_2s = 2 ** np.arange(self.kappa)[::-1]
#         last_id = 0
#         data = np.zeros(iteration_length, dtype=int)
#         for st in range(0, num_bits, self.kappa):
#             en = min(st + self.kappa, num_bits)
#             data ^= angles_bits[:, st:en].dot(power_of_2s[: en - st])
#             yield qrom.QROM(
#                 [data], selection_bitsizes=tuple(selection_bitsizes), target_bitsizes=(self.kappa,)
#             ).on_registers(selection=selection, target0=kappa_load_target)
#             data = angles_bits[:, st:en].dot(power_of_2s[: en - st])
#             for cqid, bpow, idx in zip(kappa_load_target, angles_bit_pow[st:en], angles_idx[st:en]):
#                 if idx != last_id:
#                     yield self.interleaved_unitary(
#                         last_id, rotations_target=rotations_target, **interleaved_unitary_target
#                     )
#                     last_id = idx
#                 yield self.rotation_gate(bpow).on(*rotations_target).controlled_by(cqid)
#         yield qrom.QROM(
#             [data], selection_bitsizes=tuple(selection_bitsizes), target_bitsizes=(self.kappa,)
#         ).on_registers(selection=selection, target0=kappa_load_target)


@frozen
class SelectTHC(Bloq):
    r"""SELECT for THC Hamilontian.

    Args:
        num_mu: THC auxiliary index dimension $M$
        num_spin_orb: number of spin orbitals $N$
        angles: Sequence of integer-approximated rotation angles s.t.
            `rotation_gate ** float_from_integer_approximation(angles[i][k])` should be applied
            to the target register when the selection register of ith multiplexed rotation array
            stores integer `k`.

    Registers:
     - mu: $\mu$ register.
     - nu: $\nu$ register.
     - theta: sign register.
     - succ: success flag qubit from uniform state preparation
     - eq_nu_mp1: flag for if $nu = M+1$
     - plus_a / plus_b: plus state for controlled swaps on spins.
     - sys_a / sys_b : System registers for (a)lpha/(b)eta orbitals.

    References:
        [Even more efficient quantum computations of chemistry through
            tensor hypercontraction](https://arxiv.org/pdf/2011.03494.pdf) Fig. 7.
    """

    num_mu: int
    num_spin_orb: int
    rotation_angles: Sequence[int]

    @cached_property
    def signature(self) -> Signature:
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
        # ancilla for and operation on |vu=M+1> and the |+> registers
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
        plus_anc = bb.add(XGate(), q=plus_anc)

        # System register spin swaps
        plus_b, sys_a, sys_b = bb.add(
            CSwapApprox(self.num_spin_orb // 2), ctrl=plus_b, x=sys_a, y=sys_b
        )

        # Rotations

        # Controlled Z

        # Clean up
        plus_b, sys_a, sys_b = bb.add(
            CSwapApprox(self.num_spin_orb // 2), ctrl=plus_b, x=sys_a, y=sys_b
        )

        # Swap spins
        # Should be a negative control (add some Xs?)
        eq_nu_mp1, plus_a, plus_b = bb.add(CSwapApprox(1), ctrl=eq_nu_mp1, x=plus_a, y=plus_b)

        # Swap mu-nu
        eq_nu_mp1, mu, nu = bb.add(CSwapApprox(num_bits_mu), ctrl=eq_nu_mp1, x=mu, y=nu)

        # System register spin swaps
        plus_b, sys_a, sys_b = bb.add(
            CSwapApprox(self.num_spin_orb // 2), ctrl=plus_b, x=sys_a, y=sys_b
        )

        # Rotations
        # gate_array = Prog
        # Controlled Z

        # Clean up
        plus_b, sys_a, sys_b = bb.add(
            CSwapApprox(self.num_spin_orb // 2), ctrl=plus_b, x=sys_a, y=sys_b
        )

        # Undo the mu-nu swaps
        and_anc, mu, nu = bb.add(CSwapApprox(num_bits_mu), ctrl=and_anc, x=mu, y=nu)
        [eq_nu_mp1, plus_anc] = bb.add(
            And(0, 1, adjoint=True), ctrl=[eq_nu_mp1, plus_anc], target=and_anc
        )
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
