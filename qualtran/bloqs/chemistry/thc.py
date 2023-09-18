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
"""SELECT and PREPARE for the tensor hypercontraction (THC) hamiltonian"""
from functools import cached_property
from typing import Dict

import cirq
import numpy as np
from attrs import frozen
from cirq_ft.algos.arithmetic_gates import LessThanEqualGate, LessThanGate
from cirq_ft.algos.multi_control_multi_target_pauli import MultiControlPauli
from cirq_ft.linalg.lcu_util import preprocess_lcu_coefficients_for_reversible_sampling

from qualtran import Bloq, BloqBuilder, Register, Signature, SoquetT
from qualtran.bloqs.arithmetic import EqualsAConstant, GreaterThanConstant
from qualtran.bloqs.basic_gates import Hadamard, Ry, Toffoli, XGate
from qualtran.bloqs.chemistry.contiguous_register import ToConitguousIndex
from qualtran.bloqs.on_each import OnEach
from qualtran.bloqs.swap_network import CSwapApprox
from qualtran.cirq_interop import CirqGateAsBloq


@frozen
class UniformSuperpositionTHC(Bloq):
    r"""Prepare uniform superposition state for THC.

    $$
        |0\rangle^{\otimes 2\log(M+1)} \rightarrow \sum_{\mu\le\nu}^{M} |\mu\rangle|\nu\rangle + \sum_{\mu}^{N/2}|\mu\rangle|\nu=M+1\rangle,
    $$

    where $M$ is the THC auxiliary dimension, and $N$ is the number of spin orbitals.

    The toffoli complexity of this gate should be 10 * log(M+1) + 2 b_r - 9.
    Currently it is a good deal larger due to:
        1. inverting inequality tests should not need more toffolis.
        2. We are not using phase-gradient gate toffoli cost for Ry rotations

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

    @staticmethod
    def add_cirq_comparator_gate(bb: 'BloqBuilder', cirq_bloq: Bloq, **regs: SoquetT):
        """Helper to split and join the registers appropriately"""
        flat_regs = []
        for k, v in regs.items():
            if v.reg.bitsize == 1:
                flat_regs.append([v])
            else:
                flat_regs.append(bb.split(v))
        print(flat_regs)
        qubits = bb.join(np.concatenate(flat_regs))
        bb.add(cirq_bloq, qubits=qubits)
        splt_qubits = bb.split(qubits)
        out_soqs = {}
        start = 0
        for k, v in regs.items():
            if v.reg.bitsize == 1:
                end = start + 1
                out_soqs[v] = splt_qubits[start:end]
                start += 1
            else:
                end = start + v.reg.bitsize
                out_soqs[v] = splt_qubits[start:end]
                start += v.reg.bitsize
        return list(out_soqs.items())

    def build_composite_bloq(self, bb: 'BloqBuilder', **regs: SoquetT) -> Dict[str, 'SoquetT']:
        mu = regs['mu']
        nu = regs['nu']
        succ = regs['succ']
        eq_nu_mp1 = regs['eq_nu_mp1']
        lte_mu_nu, lte_nu_mp1, gt_mu_n, junk, amp = bb.split(bb.allocate(5))
        num_bits_mu = self.num_mu.bit_length()
        mu = bb.add(OnEach(num_bits_mu, Hadamard()), q=mu)
        nu = bb.add(OnEach(num_bits_mu, Hadamard()), q=nu)
        data_size = self.num_mu * (self.num_mu + 1) // 2 + self.num_spin_orb // 2
        angle = np.arccos(1 - 2 ** np.floor(np.log2(data_size)) / data_size)
        amp = bb.add(Ry(angle), q=amp)
        # # lt_qubits = np.concatenate([bb.split(nu), lte_nu_mp1])
        # splt_nu = bb.split(nu)
        # self.add_cirq_arithmetic_gate(mu=mu, nu=nu, lte=lte_mu_nu)
        # lt_qubits = np.concatenate([splt_nu, [lte_nu_mp1]])
        nu, lte_nu_mp1 = self.add_cirq_comparator_gate(
            bb,
            CirqGateAsBloq(LessThanGate(num_bits_mu, self.num_mu + 1)),
            nu=nu,
            lte_nu_mp1=lte_nu_mp1,
        )
        # mu, nu, lte_mu_nu = bb.add(
        #     CirqGateAsBloq(LessThanEqualGate(num_bits_mu, num_bits_mu)), x=mu, y=nu, z=lte_mu_nu
        # )
        # nu, eq_nu_mp1 = bb.add(EqualsAConstant(num_bits_mu, self.num_mu + 1), x=nu, z=eq_nu_mp1)
        self.add_cirq_arithmetic_gate(mu=mu, nu=nu, lte=lte_mu_nu)
        # mu, gt_mu_n = bb.add(
        #     GreaterThanConstant(num_bits_mu, self.num_spin_orb // 2), x=mu, z=gt_mu_n
        # )
        # junk = bb.add(XGate(), q=junk)
        # (eq_nu_mp1, gt_mu_n), junk = bb.add(Toffoli(), ctrl=[eq_nu_mp1, gt_mu_n], target=junk)
        # amp, lte_nu_mp1, lte_mu_nu, junk = bb.add(
        #     CirqGateAsBloq(MultiControlPauli(((1,), (1,), (1,)), cirq.Z)),
        #     ctrl0=amp,
        #     ctrl1=lte_nu_mp1,
        #     ctrl2=lte_mu_nu,
        #     trgt=junk,
        # )
        # nu, lte_nu_mp1 = bb.add(
        #     CirqGateAsBloq(LessThanGate(num_bits_mu, self.num_mu + 1)), x=nu, z=lte_nu_mp1
        # )
        # mu, nu, lte_mu_nu = bb.add(
        #     CirqGateAsBloq(LessThanEqualGate(num_bits_mu, num_bits_mu)), x=mu, y=nu, z=lte_mu_nu
        # )
        # nu, eq_nu_mp1 = bb.add(EqualsAConstant(num_bits_mu, self.num_mu + 1), x=nu, z=eq_nu_mp1)
        # mu, gt_mu_n = bb.add(
        #     GreaterThanConstant(num_bits_mu, self.num_spin_orb // 2), x=mu, z=gt_mu_n
        # )
        # (eq_nu_mp1, gt_mu_n), junk = bb.add(Toffoli(), ctrl=[eq_nu_mp1, gt_mu_n], target=junk)
        # amp = bb.add(Ry(-angle), q=amp)
        # mu = bb.add(OnEach(num_bits_mu, Hadamard()), q=mu)
        # nu = bb.add(OnEach(num_bits_mu, Hadamard()), q=nu)
        # mu, nu, amp = bb.add(
        #     CirqGateAsBloq(MultiControlPauli(((1,) * num_bits_mu, (1,) * num_bits_mu), cirq.Z)),
        #     ctrl0=mu,
        #     ctrl1=nu,
        #     trgt=amp,
        # )
        # mu = bb.add(OnEach(num_bits_mu, Hadamard()), q=mu)
        # nu = bb.add(OnEach(num_bits_mu, Hadamard()), q=nu)
        # nu, lte_nu_mp1 = bb.add(
        #     CirqGateAsBloq(LessThanGate(num_bits_mu, self.num_mu + 1)), x=nu, z=lte_nu_mp1
        # )
        # mu, nu, lte_mu_nu = bb.add(
        #     CirqGateAsBloq(LessThanEqualGate(num_bits_mu, num_bits_mu)), x=mu, y=nu, z=lte_mu_nu
        # )
        # nu, eq_nu_mp1 = bb.add(EqualsAConstant(num_bits_mu, self.num_mu + 1), x=nu, z=eq_nu_mp1)
        # mu, gt_mu_n = bb.add(
        #     GreaterThanConstant(num_bits_mu, self.num_spin_orb // 2), x=mu, z=gt_mu_n
        # )
        # (eq_nu_mp1, gt_mu_n), junk = bb.add(Toffoli(), ctrl=[eq_nu_mp1, gt_mu_n], target=junk)
        # lte_nu_mp1, lte_mu_nu, junk, succ = bb.add(
        #     CirqGateAsBloq(MultiControlPauli(((1,), (1,), (1,)), cirq.X)),
        #     ctrl0=lte_nu_mp1,
        #     ctrl1=lte_mu_nu,
        #     ctrl2=junk,
        #     trgt=succ,
        # )
        # (eq_nu_mp1, gt_mu_n), junk = bb.add(Toffoli(), ctrl=[eq_nu_mp1, gt_mu_n], target=junk)
        # nu, lte_nu_mp1 = bb.add(
        #     CirqGateAsBloq(LessThanConstantGate(num_bits_mu, self.num_mu + 1)), x=nu, z=lte_nu_mp1
        # )
        # mu, nu, lte_mu_nu = bb.add(
        #     CirqGateAsBloq(LessThanEqualGate(num_bits_mu, num_bits_mu)), x=mu, y=nu, z=lte_mu_nu
        # )
        # mu, gt_mu_n = bb.add(
        #     GreaterThanConstant(num_bits_mu, self.num_spin_orb // 2), x=mu, z=gt_mu_n
        # )
        # junk = bb.add(XGate(), q=junk)
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

    Args:
        num_mu: THC auxiliary index dimension $M$
        num_spin_orb: number of spin orbitals $N$
        keep_bitsize: number of bits for keep register for coherent alias sampling.

    Registers:
     - mu: $\mu$ register.
     - nu: $\nu$ register.
     - theta: sign register.

    References:
        [Even more efficient quantum computations of chemistry through
            tensor hypercontraction](https://arxiv.org/pdf/2011.03494.pdf) Fig. 2 and Fig. 3.
    """

    num_mu: int
    num_spin_orb: int
    keep_bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("mu", bitsize=(self.num_mu).bit_length()),
                Register("nu", bitsize=(self.num_mu).bit_length()),
                Register("theta", bitsize=1),
            ]
        )

    @classmethod
    def build(cls, t_l, zeta, probability_epsilon=1e-8) -> 'PrepareTHC':
        assert len(zeta.shape) == 2
        num_spatial = len(t_l)
        num_mu = zeta.shape[0]
        triu_indices = np.triu_indices(num_mu)
        flat_data = np.concatenate(t_l, zeta[triu_indices])
        thetas = (1 - np.sign(flat_data)) // 2
        lambda_thc = np.sum(np.abs(t_l)) + 0.5 * np.sum(np.abs(zeta))
        alt, keep, mu = preprocess_lcu_coefficients_for_reversible_sampling(
            np.sqrt(flat_data / lambda_thc), epsilon=probability_epsilon
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', **regs: SoquetT) -> Dict[str, 'SoquetT']:
        data_size = self.num_spin_orb // 2 + self.num_mu * (self.num_mu + 1) // 2
        log_mu = (self.num_mu).bit_length()
        log_d = (data_size - 1).bit_length()
        # Allocate ancillae
        succ, eq_nu_mp1, flag_plus, less_than, alt_theta = bb.split(bb.allocate(5))
        s = bb.allocate(log_d)
        sigma = bb.allocate(self.keep_bitsize)
        keep = bb.allocate(self.keep_bitsize)
        alt_mu = bb.allocate(log_mu)
        alt_nu = bb.allocate(log_mu)
        # 1. Prepare THC uniform superposition over mu, nu. succ flags success.
        mu, nu, succ, eq_nu_mp1 = bb.add(
            UniformSuperpositionTHC(num_mu=self.num_mu, num_spin_orb=self.num_spin_orb),
            mu=regs['mu'],
            nu=regs['nu'],
            succ=succ,
            eq_nu_mp1=eq_nu_mp1,
        )
        # 2. Make contiguous register from mu and nu and store in register `s`.
        mu, nu, s = bb.add(ContiguousRegister(log_mu, log_d), mu=mu, nu=nu, s=s)
        theta = regs['theta']
        # 3. Load alt / keep values
        data = (
            tuple([1] * data_size),
            tuple([1] * data_size),
            tuple([10] * data_size),
            tuple([10] * data_size),
            tuple([10] * data_size),
        )
        qrom = SelectSwapQROM.build(*data, target_bitsizes=(1, 1, 4, 4, self.keep_bitsize))
        s, theta, alt_theta, alt_mu, alt_nu, keep = bb.add(
            qrom,
            selection=s,
            target0=regs['theta'],
            target1=alt_theta,
            target2=alt_mu,
            target3=alt_nu,
            target4=keep,
        )
        keep, sigma, less_than = bb.add(
            LessThanEqual(self.keep_bitsize, self.keep_bitsize), x=keep, y=sigma, z=less_than
        )
        # TODO: uncomment once controlled bloq decomposes correctly.
        # alt_theta, less_than = bb.add(ControlledBloq(ZGate()), control=alt_theta, q=less_than)
        # theta, less_than = bb.add(ControlledBloq(ZGate()), control=theta, q=less_than)
        less_than, alt_mu, mu = bb.add(CSwapApprox(bitsize=log_mu), ctrl=less_than, x=alt_mu, y=mu)
        less_than, alt_nu, nu = bb.add(CSwapApprox(bitsize=log_mu), ctrl=less_than, x=alt_nu, y=nu)
        keep, sigma, less_than = bb.add(
            LessThanEqual(self.keep_bitsize, self.keep_bitsize), x=keep, y=sigma, z=less_than
        )
        flag_plus = bb.add(Hadamard(), q=flag_plus)
        # TODO: uncomment once controlled bloq decomposes correctly.
        # eq_nu_mp1, flag_plus = bb.add(ControlledBloq(ZGate()), control=eq_nu_mp1, q=flag_plus)
        flag_plus, mu, nu = bb.add(CSwapApprox(bitsize=log_mu), ctrl=flag_plus, x=mu, y=nu)
        bb.free(bb.join(np.array([succ, eq_nu_mp1, flag_plus, less_than, alt_theta])))
        bb.free(s)
        bb.free(sigma)
        bb.free(keep)
        bb.free(alt_mu)
        bb.free(alt_nu)
        out_regs = {'mu': mu, 'nu': nu, 'theta': theta}
        return out_regs
