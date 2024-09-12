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
"""PREPARE for the molecular tensor hypercontraction (THC) hamiltonian"""
from functools import cached_property
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import cirq
import numpy as np
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QAny,
    QBit,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran._infra.data_types import BQUInt
from qualtran.bloqs.arithmetic import (
    EqualsAConstant,
    GreaterThanConstant,
    LessThanConstant,
    LessThanEqual,
    ToContiguousIndex,
)
from qualtran.bloqs.basic_gates import CSwap, Hadamard, Ry, Toffoli, XGate
from qualtran.bloqs.basic_gates.on_each import OnEach
from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM
from qualtran.bloqs.mcmt import MultiControlX
from qualtran.bloqs.reflections.reflection_using_prepare import ReflectionUsingPrepare
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.cirq_interop import CirqGateAsBloq
from qualtran.drawing import Text, WireSymbol
from qualtran.linalg.lcu_util import preprocess_probabilities_for_reversible_sampling
from qualtran.resource_counting.generalizers import ignore_cliffords, ignore_split_join
from qualtran.symbolics import SymbolicFloat

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@frozen
class UniformSuperpositionTHC(Bloq):
    r"""Prepare uniform superposition state for THC.

    $$
        |0\rangle^{\otimes 2\log(M+1)} \rightarrow \sum_{\mu\le\nu}^{M} |\mu\rangle|\nu\rangle
        + \sum_{\mu}^{N/2}|\mu\rangle|\nu=M+1\rangle,
    $$

    where $M$ is the THC auxiliary dimension, and $N$ is the number of spin orbitals.

    The toffoli complexity of this gate should be $10 \log(M+1) + 2 b_r - 9$.
    Currently it is a good deal larger due to:
     1. inverting inequality tests should not need more toffolis.
     2. We are not using phase-gradient gate toffoli cost for Ry rotations.
     3. Small differences in quoted vs implemented comparator costs.

    See: https://github.com/quantumlib/Qualtran/issues/390

    Args:
        num_mu: THC auxiliary index dimension $M$
        num_spin_orb: number of spin orbitals $N$

    Registers:
        mu: $\mu$ register.
        nu: $\nu$ register.
        succ: ancilla flagging success of amplitude amplification.
        nu_eq_mp1: ancillas for flagging if $\nu = M+1$.
        rot: The ancilla to be rotated for amplitude amplification.

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
                Register("mu", QAny(bitsize=self.num_mu.bit_length())),
                Register("nu", QAny(bitsize=self.num_mu.bit_length())),
                Register("nu_eq_mp1", QBit()),
                Register("succ", QBit()),
                Register("rot", QBit()),
            ]
        )

    def __str__(self) -> str:
        return r'$\sum_{\mu < \nu} |\mu\nu\rangle$'

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('Σ |μν>')
        return super().wire_symbol(reg, idx)

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        mu: SoquetT,
        nu: SoquetT,
        succ: SoquetT,
        nu_eq_mp1: SoquetT,
        rot: SoquetT,
    ) -> Dict[str, 'SoquetT']:
        # If we introduce comparators using out of place adders these will be left/right registers.
        # See: https://github.com/quantumlib/Qualtran/issues/390
        lte_mu_nu, lte_nu_mp1, gt_mu_n, junk = bb.split(bb.allocate(4))
        num_bits_mu = self.num_mu.bit_length()
        # 1. Prepare uniform superposition over all mu and nu
        mu = bb.add(OnEach(num_bits_mu, Hadamard()), q=mu)
        nu = bb.add(OnEach(num_bits_mu, Hadamard()), q=nu)
        # 2. Rotate an ancilla by `an angle`, appropriately chosen to be related
        # to the amount of data we actually want to load (upper triangle +
        # one-body)
        data_size = self.num_mu * (self.num_mu + 1) // 2 + self.num_spin_orb // 2
        angle = np.arccos(1 - 2 ** np.floor(np.log2(data_size)) / data_size)
        rot = bb.add(Ry(angle), q=rot)
        # 3. nu <= mu + 1 (zero injdexing we use mu)
        lt_gate = LessThanConstant(num_bits_mu, self.num_mu)
        nu, lte_nu_mp1 = bb.add(lt_gate, x=nu, target=lte_nu_mp1)
        # 4. mu <= nu (upper triangular)
        lte_gate = LessThanEqual(num_bits_mu, num_bits_mu)
        mu, nu, lte_mu_nu = bb.add(lte_gate, x=mu, y=nu, target=lte_mu_nu)
        # 5. nu == M (i.e. flag one-body contribution)
        nu, nu_eq_mp1 = bb.add(
            EqualsAConstant(num_bits_mu, self.num_mu + 1), x=nu, target=nu_eq_mp1
        )
        # 6. nu > N / 2 (flag out of range for one-body bits)
        mu, gt_mu_n = bb.add(
            GreaterThanConstant(num_bits_mu, self.num_spin_orb // 2), x=mu, target=gt_mu_n
        )
        # 7. Control off of 5 and 6 to not prepare if these conditions are met
        (nu_eq_mp1, gt_mu_n), junk = bb.add(Toffoli(), ctrl=[nu_eq_mp1, gt_mu_n], target=junk)
        # 6. Reflect on comparitors, rotated qubit and |+>.
        # ctrls = bb.join(np.array([rot, lte_nu_mp1, lte_mu_nu]))
        rot, lte_nu_mp1, lte_mu_nu, junk = bb.add(
            ReflectionUsingPrepare.reflection_around_zero(bitsizes=(1, 1, 1, 1), global_phase=1),
            reg0_=rot,
            reg1_=lte_nu_mp1,
            reg2_=lte_mu_nu,
            reg3_=junk,
        )
        # (rot, lte_nu_mp1, lte_mu_nu) = bb.split(ctrls)
        # We now undo comparitors and rotations and repeat the steps
        nu, lte_nu_mp1 = bb.add(lt_gate, x=nu, target=lte_nu_mp1)
        mu, nu, lte_mu_nu = bb.add(lte_gate, x=mu, y=nu, target=lte_mu_nu)
        nu, nu_eq_mp1 = bb.add(
            EqualsAConstant(num_bits_mu, self.num_mu + 1), x=nu, target=nu_eq_mp1
        )
        mu, gt_mu_n = bb.add(
            GreaterThanConstant(num_bits_mu, self.num_spin_orb // 2), x=mu, target=gt_mu_n
        )
        (nu_eq_mp1, gt_mu_n), junk = bb.add(Toffoli(), ctrl=[nu_eq_mp1, gt_mu_n], target=junk)
        rot = bb.add(Ry(-angle), q=rot)
        mu = bb.add(OnEach(num_bits_mu, Hadamard()), q=mu)
        nu = bb.add(OnEach(num_bits_mu, Hadamard()), q=nu)
        mu, nu, rot = bb.add(
            ReflectionUsingPrepare.reflection_around_zero(
                bitsizes=(num_bits_mu, num_bits_mu, 1), global_phase=1
            ),
            reg0_=mu,
            reg1_=nu,
            reg2_=rot,
        )
        # amp = trg[0]
        mu = bb.add(OnEach(num_bits_mu, Hadamard()), q=mu)
        nu = bb.add(OnEach(num_bits_mu, Hadamard()), q=nu)
        nu, lte_nu_mp1 = bb.add(lt_gate, x=nu, target=lte_nu_mp1)
        mu, nu, lte_mu_nu = bb.add(lte_gate, x=mu, y=nu, target=lte_mu_nu)
        nu, nu_eq_mp1 = bb.add(
            EqualsAConstant(num_bits_mu, self.num_mu + 1), x=nu, target=nu_eq_mp1
        )
        mu, gt_mu_n = bb.add(
            GreaterThanConstant(num_bits_mu, self.num_spin_orb // 2), x=mu, target=gt_mu_n
        )
        (nu_eq_mp1, gt_mu_n), junk = bb.add(Toffoli(), ctrl=[nu_eq_mp1, gt_mu_n], target=junk)
        (lte_nu_mp1, lte_mu_nu, junk), succ = bb.add(
            MultiControlX(cvs=(1, 1, 1)),
            controls=np.array([lte_nu_mp1, lte_mu_nu, junk]),
            target=succ,
        )
        (nu_eq_mp1, gt_mu_n), junk = bb.add(Toffoli(), ctrl=[nu_eq_mp1, gt_mu_n], target=junk)
        nu, lte_nu_mp1 = bb.add(lt_gate, x=nu, target=lte_nu_mp1)
        mu, nu, lte_mu_nu = bb.add(lte_gate, x=mu, y=nu, target=lte_mu_nu)
        mu, gt_mu_n = bb.add(
            GreaterThanConstant(num_bits_mu, self.num_spin_orb // 2), x=mu, target=gt_mu_n
        )
        junk = bb.add(XGate(), q=junk)
        bb.free(bb.join(np.array([lte_mu_nu, lte_nu_mp1, gt_mu_n, junk])))
        out_regs = {'mu': mu, 'nu': nu, 'succ': succ, 'nu_eq_mp1': nu_eq_mp1, 'rot': rot}
        return out_regs


@frozen
class PrepareTHC(PrepareOracle):
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
        mu: $\mu$ register.
        nu: $\nu$ register.
        plus_mn: plus state for controlled swaps on mu/nu.
        plus_a / plus_b: plus state for controlled swaps on spins.
        sigma: ancilla register for alias sampling.
        rot: ancilla register for rotation for uniform superposition state.
        succ: success flag qubit from uniform state preparation
        nu_eq_mp1: flag for if $nu = M+1$
        theta: sign register.
        s: Contiguous index register.
        alt_mn: Register to store alt mu and nu values.
        alt_theta: Register for alternate theta values.
        keep: keep_bitsize-sized register for the keep values from coherent alias sampling.
        less_than: Single qubit ancilla for alias sampling.
        extra_ctrl: An extra control register for producing a multi-controlled CSwap.

    References:
        [Even more efficient quantum computations of chemistry through
            tensor hypercontraction](https://arxiv.org/pdf/2011.03494.pdf) Fig. 2 and Fig. 3.
    """

    num_mu: int
    num_spin_orb: int
    alt_mu: Tuple[int, ...] = field(repr=False)
    alt_nu: Tuple[int, ...] = field(repr=False)
    alt_theta: Tuple[int, ...] = field(repr=False)
    theta: Tuple[int, ...] = field(repr=False)
    keep: Tuple[int, ...] = field(repr=False)
    keep_bitsize: int
    sum_of_l1_coeffs: SymbolicFloat

    @classmethod
    def from_hamiltonian_coeffs(
        cls,
        t_l: NDArray[np.float64],
        eta: NDArray[np.float64],
        zeta: NDArray[np.float64],
        num_bits_state_prep: int = 8,
    ) -> 'PrepareTHC':
        """Factory method to build PrepareTHC from Hamiltonian coefficients.

        Args:
            t_l: One body hamiltonian eigenvalues.
            eta: The THC leaf tensors.
            zeta: THC central tensor.
            num_bits_state_prep: The number of bits for the state prepared during alias sampling.

        Returns:
            Constructed PrepareTHC object.
        """
        assert len(t_l.shape) == 1
        assert len(eta.shape) == 2
        assert len(zeta.shape) == 2
        num_mu = zeta.shape[0]
        num_spat = t_l.shape[0]
        assert eta.shape == (num_mu, num_spat)
        triu_indices = np.triu_indices(num_mu)
        num_ut = len(triu_indices[0])
        flat_data = np.concatenate([zeta[triu_indices], t_l])
        thetas = [int(t) for t in (1 - np.sign(flat_data)) // 2]
        flat_data = np.abs(flat_data)
        alt, keep, mu = preprocess_probabilities_for_reversible_sampling(
            flat_data, sub_bit_precision=num_bits_state_prep
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
        # Compute the lambda value using the formula from the reference /
        # OpenFermion: resource_estimates.thc.compute_lambda_thc
        overlap = eta.dot(eta.T)
        norm_fac = np.diag(np.diag(overlap))
        zeta_normalized = norm_fac.dot(zeta).dot(norm_fac)  # Eq. 11 & 12
        lambda_t = np.sum(np.abs(t_l))  # Eq. 19
        lambda_z = 0.5 * np.sum(np.abs(zeta_normalized))  # Eq. 20
        return PrepareTHC(
            num_mu,
            2 * num_spat,
            alt_mu=tuple(alt_mu),
            alt_nu=tuple(alt_nu),
            alt_theta=tuple(alt_theta),
            theta=tuple(thetas),
            keep=tuple(keep),
            keep_bitsize=mu,
            sum_of_l1_coeffs=lambda_t + lambda_z,
        )

    @property
    def l1_norm_of_coeffs(self) -> SymbolicFloat:
        return self.sum_of_l1_coeffs

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (
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
            Register('succ', BQUInt(bitsize=1)),
            Register('nu_eq_mp1', BQUInt(bitsize=1)),
        )

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        data_size = self.num_spin_orb // 2 + self.num_mu * (self.num_mu + 1) // 2
        log_mu = self.num_mu.bit_length()
        return (
            Register('theta', QBit()),
            Register('s', QAny(bitsize=(data_size - 1).bit_length())),
            Register('alt_mn', QAny(bitsize=log_mu), shape=(2,)),
            Register('alt_theta', QBit()),
            Register('keep', QAny(bitsize=self.keep_bitsize)),
            Register('less_than', QBit()),
            Register('extra_ctrl', QBit()),
        )

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        mu: SoquetT,
        nu: SoquetT,
        plus_mn: SoquetT,
        plus_a: SoquetT,
        plus_b: SoquetT,
        sigma: SoquetT,
        rot: SoquetT,
        succ: SoquetT,
        nu_eq_mp1: SoquetT,
        theta: SoquetT,
        s: SoquetT,
        alt_mn: NDArray[Soquet],  # type: ignore[type-var]
        alt_theta: SoquetT,
        keep: SoquetT,
        less_than: SoquetT,
        extra_ctrl: SoquetT,
    ) -> Dict[str, 'SoquetT']:
        # 1. Prepare THC uniform superposition over mu, nu. succ flags success.
        mu, nu, succ, nu_eq_mp1, rot = bb.add(
            UniformSuperpositionTHC(num_mu=self.num_mu, num_spin_orb=self.num_spin_orb),
            mu=mu,
            nu=nu,
            succ=succ,
            nu_eq_mp1=nu_eq_mp1,
            rot=rot,
        )
        data_size = self.num_spin_orb // 2 + self.num_mu * (self.num_mu + 1) // 2
        log_mu = self.num_mu.bit_length()
        log_d = (data_size - 1).bit_length()
        # 2. Make contiguous register from mu and nu and store in register `s`.
        mu, nu, s = bb.add(ToContiguousIndex(log_mu, log_d), mu=mu, nu=nu, s=s)
        # 3. Load alt / keep values
        qroam = SelectSwapQROM.build_from_data(
            *(self.theta, self.alt_theta, self.alt_mu, self.alt_nu, self.keep),
            target_bitsizes=(1, 1, log_mu, log_mu, self.keep_bitsize),
            use_dirty_ancilla=False,
        )
        alt_mu, alt_nu = alt_mn
        s, theta, alt_theta, alt_mu, alt_nu, keep = bb.add(
            qroam,
            selection=s,
            target0_=theta,
            target1_=alt_theta,
            target2_=alt_mu,
            target3_=alt_nu,
            target4_=keep,
        )
        sigma = bb.add(OnEach(self.keep_bitsize, Hadamard()), q=sigma)
        lte_gate = LessThanEqual(self.keep_bitsize, self.keep_bitsize)
        keep, sigma, less_than = bb.add(lte_gate, x=keep, y=sigma, target=less_than)
        cz = CirqGateAsBloq(cirq.ControlledGate(cirq.Z))
        alt_theta, less_than = bb.add(cz, q=[alt_theta, less_than])
        cz = CirqGateAsBloq(cirq.ControlledGate(cirq.Z, control_values=(0,)))
        # negative control on the less_than register
        less_than, theta = bb.add(cz, q=[less_than, theta])
        less_than, alt_mu, mu = bb.add(CSwap(bitsize=log_mu), ctrl=less_than, x=alt_mu, y=mu)
        less_than, alt_nu, nu = bb.add(CSwap(bitsize=log_mu), ctrl=less_than, x=alt_nu, y=nu)
        keep, sigma, less_than = bb.add(lte_gate, x=keep, y=sigma, target=less_than)
        # delete the QROM
        # Select expects three plus states so set them up here.
        plus_a = bb.add(Hadamard(), q=plus_a)
        plus_b = bb.add(Hadamard(), q=plus_b)
        plus_mn = bb.add(Hadamard(), q=plus_mn)
        (nu_eq_mp1, plus_a), extra_ctrl = bb.add(
            MultiControlX(cvs=(0, 1)), controls=np.array([nu_eq_mp1, plus_a]), target=extra_ctrl
        )
        extra_ctrl, mu, nu = bb.add(CSwap(bitsize=log_mu), ctrl=extra_ctrl, x=mu, y=nu)
        out_regs = {
            'mu': mu,
            'nu': nu,
            'plus_mn': plus_mn,
            'plus_a': plus_a,
            'plus_b': plus_b,
            'sigma': sigma,
            'rot': rot,
            'succ': succ,
            'nu_eq_mp1': nu_eq_mp1,
            'theta': theta,
            's': s,
            'alt_mn': [alt_mu, alt_nu],
            'alt_theta': alt_theta,
            'keep': keep,
            'less_than': less_than,
            'extra_ctrl': extra_ctrl,
        }
        return out_regs

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        cost_1 = (UniformSuperpositionTHC(self.num_mu, self.num_spin_orb), 1)
        nmu = self.num_mu.bit_length()
        data_size = self.num_spin_orb // 2 + self.num_mu * (self.num_mu + 1) // 2
        nd = (data_size - 1).bit_length()
        cost_2 = (ToContiguousIndex(nmu, nd), 1)
        qroam = SelectSwapQROM.build_from_data(
            *(self.theta, self.alt_theta, self.alt_mu, self.alt_nu, self.keep),
            target_bitsizes=(1, 1, nmu, nmu, self.keep_bitsize),
            use_dirty_ancilla=False,
        )
        cost_3 = (qroam, 1)
        cost_4 = (OnEach(self.keep_bitsize, Hadamard()), 1)
        cost_5 = (LessThanEqual(self.keep_bitsize, self.keep_bitsize), 2)
        cost_6 = (CSwap(nmu), 3)
        cost_7 = (Toffoli(), 1)
        return dict([cost_1, cost_2, cost_3, cost_4, cost_5, cost_6, cost_7])


@bloq_example
def _thc_uni() -> UniformSuperpositionTHC:
    num_mu = 10
    num_spin_orb = 4
    thc_uni = UniformSuperpositionTHC(num_mu=num_mu, num_spin_orb=num_spin_orb)
    return thc_uni


@bloq_example(generalizer=[ignore_split_join, ignore_cliffords])
def _thc_prep() -> PrepareTHC:
    from qualtran.bloqs.chemistry.thc.prepare_test import build_random_test_integrals

    num_spat = 4
    num_mu = 8
    t_l, eta, zeta = build_random_test_integrals(num_mu, num_spat, seed=7)
    thc_prep = PrepareTHC.from_hamiltonian_coeffs(t_l, eta, zeta, num_bits_state_prep=8)
    return thc_prep


_THC_UNI_PREP = BloqDocSpec(bloq_cls=UniformSuperpositionTHC, examples=(_thc_uni,))

_THC_PREPARE = BloqDocSpec(bloq_cls=PrepareTHC, examples=(_thc_prep,))
