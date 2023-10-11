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
"""SELECT and PREPARE for the molecular tensor hypercontraction (THC) hamiltonian"""
from functools import cached_property
from typing import Dict, Tuple

import cirq
import numpy as np
from attrs import frozen
from cirq_ft.algos.arithmetic_gates import LessThanEqualGate, LessThanGate
from cirq_ft.algos.multi_control_multi_target_pauli import MultiControlPauli
from cirq_ft.algos.select_swap_qrom import SelectSwapQROM
from cirq_ft.linalg.lcu_util import preprocess_lcu_coefficients_for_reversible_sampling

from qualtran import Bloq, BloqBuilder, Register, Signature, SoquetT
from qualtran.bloqs.arithmetic import EqualsAConstant, GreaterThanConstant, ToContiguousIndex
from qualtran.bloqs.basic_gates import Hadamard, Ry, Toffoli, XGate
from qualtran.bloqs.on_each import OnEach
from qualtran.bloqs.swap_network import CSwapApprox
from qualtran.cirq_interop import CirqGateAsBloq


def add_from_bloq_register_flat_qubits(
    bb: 'BloqBuilder', cirq_bloq: Bloq, **regs: SoquetT
) -> Tuple[SoquetT, ...]:
    """Helper function to split / join registers for cirq gates expeciting single 'qubits' register.

    Args:
        bb: Bloq builder used during decompostion.
        cirq_bloq: A CirqGateAsBloq wrapped arithmetic gate.
        regs: bloq registers we wish to use as flat list of qubits for cirq gate.

    Returns:
        regs: bloq registers appropriately rejoined following split.
    """
    flat_regs = []
    for _, v in regs.items():
        if v.reg.bitsize == 1:
            flat_regs.append([v])
        else:
            flat_regs.append(bb.split(v))
    qubits = np.concatenate(flat_regs)
    qubits = bb.add(cirq_bloq, qubits=qubits)
    out_soqs = {}
    start = 0
    for _, v in regs.items():
        if v.reg.bitsize == 1:
            end = start + 1
            out_soqs[v] = qubits[start:end][0]
            start += 1
        else:
            end = start + v.reg.bitsize
            out_soqs[v] = bb.join(qubits[start:end])
            start += v.reg.bitsize
    return tuple(s for _, s in out_soqs.items())


@frozen
class UniformSuperpositionTHC(Bloq):
    r"""Prepare uniform superposition state for THC.

    $$
        |0\rangle^{\otimes 2\log(M+1)} \rightarrow \sum_{\mu\le\nu}^{M} |\mu\rangle|\nu\rangle 
        + \sum_{\mu}^{N/2}|\mu\rangle|\nu=M+1\rangle,
    $$

    where $M$ is the THC auxiliary dimension, and $N$ is the number of spin orbitals.

    The toffoli complexity of this gate should be 10 * log(M+1) + 2 b_r - 9.
    Currently it is a good deal larger due to:
        1. inverting inequality tests should not need more toffolis.
        2. We are not using phase-gradient gate toffoli cost for Ry rotations
        3. Small differences in quoted vs implemented comparator costs.
    
    See: https://github.com/quantumlib/Qualtran/issues/390

    Args:
        num_mu: THC auxiliary index dimension $M$
        num_spin_orb: number of spin orbitals $N$

    Registers:
    - mu: $\mu$ register.
    - nu: $\nu$ register.
    - succ: ancilla flagging success of amplitude amplification.
    - eq_nu_mp1: ancillas for flagging if $\nu = M+1$.

    References:
        [Even more efficient quantum computations of chemistry through
            tensor hypercontraction](https://arxiv.org/pdf/2011.03494.pdf). Eq. 29.
    """

    num_mu: int
    num_spin_orb: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("mu", bitsize=(self.num_mu).bit_length()),
                Register("nu", bitsize=(self.num_mu).bit_length()),
                Register("succ", bitsize=1),
                Register("eq_nu_mp1", bitsize=1),
            ]
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', **regs: SoquetT) -> Dict[str, 'SoquetT']:
        mu = regs['mu']
        nu = regs['nu']
        succ = regs['succ']
        eq_nu_mp1 = regs['eq_nu_mp1']
        lte_mu_nu, lte_nu_mp1, gt_mu_n, junk, amp = bb.split(bb.allocate(5))
        num_bits_mu = self.num_mu.bit_length()
        # 1. Prepare uniform superposition over all mu and nu
        mu = bb.add(OnEach(num_bits_mu, Hadamard()), q=mu)
        nu = bb.add(OnEach(num_bits_mu, Hadamard()), q=nu)
        # 2. Rotate an ancilla by `an angle`, appropriately chosen to be related
        # to the amount of data we actually want to load (upper triangle +
        # one-body)
        data_size = self.num_mu * (self.num_mu + 1) // 2 + self.num_spin_orb // 2
        angle = np.arccos(1 - 2 ** np.floor(np.log2(data_size)) / data_size)
        amp = bb.add(Ry(angle), q=amp)
        # 3. nu <= mu + 1 (zero indexing we use mu)
        lt_gate = CirqGateAsBloq(LessThanGate(num_bits_mu, self.num_mu))
        nu, lte_nu_mp1 = add_from_bloq_register_flat_qubits(
            bb, lt_gate, nu=nu, lte_mu_mp1=lte_nu_mp1
        )
        # 4. mu <= nu (upper triangular)
        lte_gate = CirqGateAsBloq(LessThanEqualGate(num_bits_mu, num_bits_mu))
        mu, nu, lte_mu_nu = add_from_bloq_register_flat_qubits(
            bb, lte_gate, mu=mu, nu=nu, lte_mu_nu=lte_mu_nu
        )
        # 5. nu == M (i.e. flag one-body contribution)
        nu, eq_nu_mp1 = bb.add(
            EqualsAConstant(num_bits_mu, self.num_mu + 1), x=nu, target=eq_nu_mp1
        )
        # 6. nu > N / 2 (flag out of range for one-body bits)
        mu, gt_mu_n = bb.add(
            GreaterThanConstant(num_bits_mu, self.num_spin_orb // 2), x=mu, target=gt_mu_n
        )
        # 7. Control off of 5 and 6 to not prepare if these conditions are met
        (eq_nu_mp1, gt_mu_n), junk = bb.add(Toffoli(), ctrl=[eq_nu_mp1, gt_mu_n], target=junk)
        # 6. Reflect on comparitors, rotated qubit and |+>.
        ctrls = bb.join(np.array([amp, lte_nu_mp1, lte_mu_nu]))
        ctrls, junk = bb.add(
            CirqGateAsBloq(MultiControlPauli(cvs=(1, 1, 1), target_gate=cirq.Z)),
            controls=ctrls,
            target=junk,
        )
        (amp, lte_nu_mp1, lte_mu_nu) = bb.split(ctrls)
        # We now undo comparitors and rotations and repeat the steps
        nu, lte_nu_mp1 = add_from_bloq_register_flat_qubits(
            bb, lt_gate, nu=nu, lte_mu_mp1=lte_nu_mp1
        )
        mu, nu, lte_mu_nu = add_from_bloq_register_flat_qubits(
            bb, lte_gate, mu=mu, nu=nu, lte_mu_nu=lte_mu_nu
        )
        nu, eq_nu_mp1 = bb.add(
            EqualsAConstant(num_bits_mu, self.num_mu + 1), x=nu, target=eq_nu_mp1
        )
        mu, gt_mu_n = bb.add(
            GreaterThanConstant(num_bits_mu, self.num_spin_orb // 2), x=mu, target=gt_mu_n
        )
        (eq_nu_mp1, gt_mu_n), junk = bb.add(Toffoli(), ctrl=[eq_nu_mp1, gt_mu_n], target=junk)
        amp = bb.add(Ry(-angle), q=amp)
        mu = bb.add(OnEach(num_bits_mu, Hadamard()), q=mu)
        nu = bb.add(OnEach(num_bits_mu, Hadamard()), q=nu)
        ctrls, amp = bb.add(
            CirqGateAsBloq(MultiControlPauli(((1,) * num_bits_mu + (1,) * num_bits_mu), cirq.Z)),
            controls=bb.join(np.concatenate([bb.split(mu), bb.split(nu)])),
            target=amp,
        )
        # amp = trg[0]
        mu_nu = bb.split(ctrls)
        mu = bb.join(mu_nu[:num_bits_mu])
        nu = bb.join(mu_nu[num_bits_mu:])
        mu = bb.add(OnEach(num_bits_mu, Hadamard()), q=mu)
        nu = bb.add(OnEach(num_bits_mu, Hadamard()), q=nu)
        nu, lte_nu_mp1 = add_from_bloq_register_flat_qubits(
            bb, lt_gate, nu=nu, lte_mu_mp1=lte_nu_mp1
        )
        mu, nu, lte_mu_nu = add_from_bloq_register_flat_qubits(
            bb, lte_gate, mu=mu, nu=nu, lte_mu_nu=lte_mu_nu
        )
        nu, eq_nu_mp1 = bb.add(
            EqualsAConstant(num_bits_mu, self.num_mu + 1), x=nu, target=eq_nu_mp1
        )
        mu, gt_mu_n = bb.add(
            GreaterThanConstant(num_bits_mu, self.num_spin_orb // 2), x=mu, target=gt_mu_n
        )
        (eq_nu_mp1, gt_mu_n), junk = bb.add(Toffoli(), ctrl=[eq_nu_mp1, gt_mu_n], target=junk)
        ctrls = bb.join(np.array([lte_nu_mp1, lte_mu_nu, junk]))
        ctrls, succ = bb.add(
            CirqGateAsBloq(MultiControlPauli(cvs=(1, 1, 1), target_gate=cirq.X)),
            controls=ctrls,
            target=succ,
        )
        lte_nu_mp1, lte_mu_nu, junk = bb.split(ctrls)
        (eq_nu_mp1, gt_mu_n), junk = bb.add(Toffoli(), ctrl=[eq_nu_mp1, gt_mu_n], target=junk)
        nu, lte_nu_mp1 = add_from_bloq_register_flat_qubits(
            bb, lt_gate, nu=nu, lte_mu_mp1=lte_nu_mp1
        )
        mu, nu, lte_mu_nu = add_from_bloq_register_flat_qubits(
            bb, lte_gate, mu=mu, nu=nu, lte_mu_nu=lte_mu_nu
        )
        mu, gt_mu_n = bb.add(
            GreaterThanConstant(num_bits_mu, self.num_spin_orb // 2), x=mu, target=gt_mu_n
        )
        junk = bb.add(XGate(), q=junk)
        bb.free(bb.join(np.array([lte_mu_nu, lte_nu_mp1, gt_mu_n, junk, amp])))
        out_regs = {'mu': mu, 'nu': nu, 'succ': succ, 'eq_nu_mp1': eq_nu_mp1}
        return out_regs


@frozen
class PrepareTHC(Bloq):
    r"""State Preparation for THC Hamilontian.

    Prepares the state

    $$
        \frac{1}{\sqrt{\lambda}}|+\rangle|+\rangle\left[
            \sum_\ell^{N/2} \sqrt{t_\ell}|\ell\rangle|M+1\rangle
            + \frac{1}{\sqrt{2}} \sum_{\mu\le\nu}^M \sqrt{\zeta_{\mu\nu}} |\mu\rangle|\nu\rangle
        \right].
    $$

    Note we use UniformSuperpositionTHC as a subroutine as part of this bloq in
    contrast to the reference which keeps them separate.

    Args:
        num_mu: THC auxiliary index dimension $M$
        num_spin_orb: number of spin orbitals $N$
        alt_mu: Alternate values for mu indices.
        alt_nu: Alternate values for nu indices.
        alt_theta: Alternate values for theta indices.
        theta: Signs of lcu coefficients.
        keep: keep values.
        keep_bitsize: number of bits for keep register for coherent alias sampling.

    Registers:
     - mu: $\mu$ register.
     - nu: $\nu$ register.
     - theta: sign register.
     - succ: success flag qubit from uniform state preparation
     - eq_nu_mp1: flag for if $nu = M+1$
     - plus_a / plus_b: plus state for controlled swaps on spins.

    References:
        [Even more efficient quantum computations of chemistry through
            tensor hypercontraction](https://arxiv.org/pdf/2011.03494.pdf) Fig. 2 and Fig. 3.
    """

    num_mu: int
    num_spin_orb: int
    alt_mu: Tuple[int, ...]
    alt_nu: Tuple[int, ...]
    alt_theta: Tuple[int, ...]
    theta: Tuple[int, ...]
    keep: Tuple[int, ...]
    keep_bitsize: int

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
            ]
        )

    def __repr__(self) -> str:
        return (
            f"PrepareTHC(num_mu={self.num_mu}, num_spin_orb={self.num_spin_orb},"
            "keep_bitsize={self.keep_bitsize})"
        )

    @classmethod
    def build(cls, t_l, zeta, probability_epsilon=1e-8) -> 'PrepareTHC':
        """Factory method to build PrepareTHC from Hamiltonian coefficients."""
        assert len(t_l.shape) == 1
        assert len(zeta.shape) == 2
        num_mu = zeta.shape[0]
        num_spat = t_l.shape[0]
        triu_indices = np.triu_indices(num_mu)
        num_ut = len(triu_indices[0])
        flat_data = np.abs(np.concatenate([zeta[triu_indices], t_l]))
        thetas = [int(t) for t in (1 - np.sign(flat_data)) // 2]
        alt, keep, mu = preprocess_lcu_coefficients_for_reversible_sampling(
            flat_data, epsilon=probability_epsilon
        )
        num_up_t = len(triu_indices[0])
        alt_mu = []
        alt_nu = []
        alt_theta = []
        for k in alt:
            if k < num_up_t:
                # if k < n * (n + 1) / 2 we are dealing with mu / nu indices
                alt_mu.append(int(triu_indices[0][k]))
                alt_nu.append(int(triu_indices[1][k]))
            else:
                # else we are dealing with the one-body bit
                alt_mu.append(int(k - num_ut))
                alt_nu.append(int(num_mu))
            alt_theta.append(thetas[k])
        return PrepareTHC(
            num_mu,
            2 * num_spat,
            alt_mu=tuple(alt_mu),
            alt_nu=tuple(alt_nu),
            alt_theta=tuple(alt_theta),
            theta=tuple(thetas),
            keep=tuple(keep),
            keep_bitsize=mu,
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', **regs: SoquetT) -> Dict[str, 'SoquetT']:
        data_size = self.num_spin_orb // 2 + self.num_mu * (self.num_mu + 1) // 2
        log_mu = (self.num_mu).bit_length()
        log_d = (data_size - 1).bit_length()
        # Allocate ancillae
        less_than, alt_theta = bb.split(bb.allocate(2))
        s = bb.allocate(log_d)
        sigma = bb.allocate(self.keep_bitsize)
        keep = bb.allocate(self.keep_bitsize)
        alt_bitsize = max(max(self.alt_mu).bit_length(), max(self.alt_nu).bit_length())
        alt_mu = bb.allocate(alt_bitsize)
        alt_nu = bb.allocate(alt_bitsize)
        # 1. Prepare THC uniform superposition over mu, nu. succ flags success.
        mu, nu, succ, eq_nu_mp1 = bb.add(
            UniformSuperpositionTHC(num_mu=self.num_mu, num_spin_orb=self.num_spin_orb),
            mu=regs['mu'],
            nu=regs['nu'],
            succ=regs['succ'],
            eq_nu_mp1=regs['eq_nu_mp1'],
        )
        # 2. Make contiguous register from mu and nu and store in register `s`.
        mu, nu, s = bb.add(ToContiguousIndex(log_mu, log_d), mu=mu, nu=nu, s=s)
        theta = regs['theta']
        # 3. Load alt / keep values
        qroam = CirqGateAsBloq(
            SelectSwapQROM(
                *(self.theta, self.alt_theta, self.alt_mu, self.alt_nu, self.keep),
                target_bitsizes=(1, 1, alt_bitsize, alt_bitsize, self.keep_bitsize),
            )
        )
        s, theta, alt_theta, alt_mu, alt_nu, keep = bb.add(
            qroam,
            selection=s,
            target0=theta,
            target1=alt_theta,
            target2=alt_mu,
            target3=alt_nu,
            target4=keep,
        )
        sigma = bb.add(OnEach(self.keep_bitsize, Hadamard()), q=sigma)
        lte_gate = CirqGateAsBloq(LessThanEqualGate(self.keep_bitsize, self.keep_bitsize))
        keep, sigma, less_than = add_from_bloq_register_flat_qubits(
            bb, lte_gate, keep=keep, sigma=sigma, less_than=less_than
        )
        cz = CirqGateAsBloq(cirq.ControlledGate(cirq.Z))
        alt_theta, less_than = add_from_bloq_register_flat_qubits(
            bb, cz, alt_theta=alt_theta, less_than=less_than
        )
        cz = CirqGateAsBloq(cirq.ControlledGate(cirq.Z, control_values=(0,)))
        # negative control on the less_than register
        less_than, theta = add_from_bloq_register_flat_qubits(
            bb, cz, less_than=less_than, theta=theta
        )
        less_than, alt_mu, mu = bb.add(CSwapApprox(bitsize=log_mu), ctrl=less_than, x=alt_mu, y=mu)
        less_than, alt_nu, nu = bb.add(CSwapApprox(bitsize=log_mu), ctrl=less_than, x=alt_nu, y=nu)
        keep, sigma, less_than = add_from_bloq_register_flat_qubits(
            bb, lte_gate, keep=keep, sigma=sigma, less_than=less_than
        )
        # Select expects two plus states so set them up here.
        plus_a = bb.add(Hadamard(), q=regs['plus_a'])
        plus_b = bb.add(Hadamard(), q=regs['plus_b'])
        # Need a toffoli gate with one negative control. And it into an ancilla
        # and control off this for the swaps.
        junk = bb.allocate(1)
        ctrls = bb.join(np.array([eq_nu_mp1, plus_a]))
        ctrls, junk = bb.add(
            CirqGateAsBloq(MultiControlPauli(cvs=(0, 1), target_gate=cirq.X)),
            controls=ctrls,
            target=junk,
        )
        eq_nu_mp1, plus_a = bb.split(ctrls)
        junk, mu, nu = bb.add(CSwapApprox(bitsize=log_mu), ctrl=junk, x=mu, y=nu)
        bb.free(bb.join(np.array([less_than, alt_theta])))
        bb.free(junk)
        bb.free(s)
        bb.free(sigma)
        bb.free(keep)
        bb.free(alt_mu)
        bb.free(alt_nu)
        out_regs = {
            'mu': mu,
            'nu': nu,
            'theta': theta,
            'plus_a': plus_a,
            'plus_b': plus_b,
            'succ': succ,
            'eq_nu_mp1': eq_nu_mp1,
        }
        return out_regs
