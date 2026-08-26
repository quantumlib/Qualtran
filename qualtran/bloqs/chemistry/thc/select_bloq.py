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
from typing import Dict, Optional, Tuple

import numpy as np
from attrs import evolve, field, frozen

from qualtran import (
    AddControlledT,
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    BQUInt,
    CtrlSpec,
    QAny,
    QBit,
    QFxp,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.basic_gates import CSwap, XGate
from qualtran.bloqs.chemistry.black_boxes import ApplyControlledZs
from qualtran.bloqs.chemistry.quad_fermion.givens_bloq import RealGivensRotationByPhaseGradient
from qualtran.bloqs.data_loading.qroam_clean import QROAMClean
from qualtran.bloqs.multiplexers.select_base import SelectOracle
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def _leaf_tensor_to_givens_rotations(eta: np.ndarray, num_bits_theta: int) -> np.ndarray:
    r"""Compute Givens rotation angles for transformation from orbital basis to THC

    Given the THC leaf tensor $\eta$ of shape $(M, N/2)$, this function computes the
    $N/2 - 1$ Givens rotation angles, $\theta$, that rotate from the standard basis to
    the THC basis for each $\mu \in [0, M-1]$.

    Args:
        eta: THC leaf tensor of shape (M, N/2).
        num_bits_theta: Precision $\beth$ (in bits) for the rotation angles.

    Returns:
        thetas: Integer array of shape (N/2 - 1,) containing angles $\theta$.

    References:
        [Quantum computing enhanced computational catalysis](https://arxiv.org/abs/2007.14460).
            Burg, Low, et al. 2021. Eq. 57
    """
    assert len(eta.shape) == 2
    num_mu, num_spatial = eta.shape
    num_angles = num_spatial - 1
    if num_angles <= 0:
        return np.zeros((0, num_mu), dtype=int)
    thetas = np.zeros((num_angles, num_mu), dtype=int)
    dtype = QFxp(num_bits_theta, num_bits_theta, signed=False)
    fpi = 4 * np.pi
    for mu in range(num_mu):
        eta_mu = eta[mu]
        norm = np.linalg.norm(eta_mu)
        if norm < 1e-12:
            continue
        u = eta_mu / norm

        # solve for first theta
        theta_first = (np.arctan2(u[num_angles], u[num_angles - 1]) + 2 * np.pi) % (2 * np.pi)
        thetas[num_angles - 1, mu] = dtype.to_fixed_width_int(theta_first / fpi)
        hypot = u[-1] ** 2 + u[-2] ** 2

        # solve for remaining thetas
        for i in range(num_angles - 2, -1, -1):
            theta_next = np.arctan2(np.sqrt(hypot), u[i])
            thetas[i, mu] = dtype.to_fixed_width_int(theta_next / fpi)
            hypot += u[i] ** 2
    return thetas


@frozen
class THCRotations(Bloq):
    r"""Bloq for rotating into THC basis through Givens rotation network.

    This is accounting for In-data:rot and In-R in Fig. 7 of the THC paper (Ref.
    1).

    The bloq loads $N/2 - 1$ Givens rotation angles from the selection register
    using QROAM, prepares a phase gradient state, applies $N/2 - 1$ real
    Givens rotations between spatial orbitals using `RealGivensRotationByPhaseGradient`,
    and uncomputes the phase gradient state.

    Args:
        num_mu: THC auxiliary index dimension $M$
        num_spin_orb: number of spin orbitals $N$ (number of spatial orbitals = $N/2$).
        num_bits_theta: Number of bits of precision for the rotations. Called
            $\beth$ in the reference.
        two_body_only: Whether to only apply the two body Hamiltonian. This reduces the QROM size.
        is_adjoint: Whether to dagger this bloq or not.
        angles_data: Optional tuple of tuples of integer-quantized angles.
            If None, default (zero) angle data is used.

    Registers:
        data: Register storing the loaded rotation angle data.
        sel: Selection register indexing $\mu$ (or one-body orbital).
        trg: Target spatial orbitals register of size $N/2$.

    References:
        [Even more efficient quantum computations of chemistry through
            tensor hypercontraction](https://arxiv.org/pdf/2011.03494.pdf) Fig. 7.
        [Quantum computing enhanced computational catalysis](https://arxiv.org/abs/2007.14460).
            Burg, Low, et al. 2021. Eq. 73
    """

    num_mu: int
    num_spin_orb: int
    num_bits_theta: int
    block_size: int = 1
    two_body_only: bool = False
    angles_data: Optional[Tuple[Tuple[int, ...], ...]] = field(
        default=None,
        repr=False,
        converter=lambda d: None if d is None else tuple(tuple(int(x) for x in row) for row in d),
    )

    @classmethod
    def from_thc_leaf_tensor(
        cls, eta: np.ndarray, num_bits_theta: int, block_size: int = 1, two_body_only: bool = False
    ) -> 'THCRotations':
        r"""Construct THCRotations from a THC leaf tensor $\eta$.

        Args:
            eta: THC vectors of shape (M, N/2).
            num_bits_theta: Number of bits of precision for the rotation angles ($\beth$).
            block_size: Block size for QROM loading.
            two_body_only: Whether to only apply the two body Hamiltonian.

        Returns:
            Constructed THCRotations object.
        """
        num_mu, num_spatial = eta.shape
        num_spin_orb = 2 * num_spatial
        angles_data = _leaf_tensor_to_givens_rotations(eta, num_bits_theta=num_bits_theta)
        return cls(
            num_mu=num_mu,
            num_spin_orb=num_spin_orb,
            num_bits_theta=num_bits_theta,
            block_size=block_size,
            two_body_only=two_body_only,
            angles_data=tuple(tuple(int(x) for x in row) for row in angles_data),
        )

    @property
    def num_spatial(self) -> int:
        return self.num_spin_orb // 2

    @property
    def num_angles(self) -> int:
        return self.num_spatial - 1

    @property
    def data_bitsize(self) -> int:
        return self.num_angles * self.num_bits_theta

    @property
    def num_terms(self) -> int:
        return self.num_mu

    @cached_property
    def selection_bitsize(self) -> int:
        return self.num_mu.bit_length()

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("data", QAny(bitsize=self.data_bitsize)),
                Register("sel", QAny(bitsize=self.selection_bitsize)),
                Register("trg", QAny(bitsize=self.num_spatial)),
            ]
        )

    def __str__(self) -> str:
        return "In_mu-R"

    def _build_qroam(self, num_angles: int, block_size: int) -> QROAMClean:
        if self.angles_data is None:
            # NOTE: using an array of zeros effectively skips QROAM so this will throw off the cost
            angles_data = tuple(tuple(0 for _ in range(self.num_terms)) for _ in range(num_angles))
        else:
            angles_data = self.angles_data
            # pad angles array if necessary for one electron terms
            angles_data = tuple(
                col + (0,) * max(0, self.num_terms - len(col)) for col in self.angles_data
            )

        target_bitsizes = (self.num_bits_theta,) * num_angles
        log_block_size = (int(block_size) - 1).bit_length()
        return QROAMClean(
            data_or_shape=[np.array(col) for col in angles_data],
            target_bitsizes=target_bitsizes,
            selection_bitsizes=(self.selection_bitsize,),
            log_block_sizes=(log_block_size,),
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        if self.num_angles <= 0:
            return {}
        qroam = self._build_qroam(self.num_angles, self.block_size)
        pg = PhaseGradientState(self.num_bits_theta)
        givens = RealGivensRotationByPhaseGradient(self.num_bits_theta)
        return {qroam: 1, pg: 1, givens: self.num_angles, pg.adjoint(): 1}

    def build_composite_bloq(
        self, bb: 'BloqBuilder', data: 'SoquetT', sel: 'SoquetT', trg: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        if self.num_angles <= 0:
            return {'data': data, 'sel': sel, 'trg': trg}

        bb.free(data)
        qroam = self._build_qroam(self.num_angles, self.block_size)
        sel_res = bb.add_d(qroam, selection=sel)
        sel = sel_res['selection']
        targets = [sel_res[f'target{i}_'] for i in range(self.num_angles)]
        for i in range(self.num_angles):
            junk = sel_res.get(f'junk_target{i}_')
            if junk is not None:
                for soq in np.asarray(junk).flat:
                    bb.free(soq)

        pg = bb.add(PhaseGradientState(self.num_bits_theta))
        q_trg = bb.split(trg)
        for k in range(self.num_angles):
            q_trg[k], q_trg[k + 1], targets[k], pg = bb.add(
                RealGivensRotationByPhaseGradient(self.num_bits_theta),
                target_i=q_trg[k],
                target_j=q_trg[k + 1],
                rom_data=targets[k],
                phase_gradient=pg,
            )

        bb.add(PhaseGradientState(self.num_bits_theta).adjoint(), phase_grad=pg)
        data = bb.join(np.concatenate([bb.split(tgt) for tgt in targets]))
        return {'data': data, 'sel': sel, 'trg': bb.join(q_trg)}


@frozen
class SelectTHC(SelectOracle):
    r"""SELECT for THC Hamiltonian.

    Args:
        num_mu: THC auxiliary index dimension $M$
        num_spin_orb: number of spin orbitals $N$
        num_bits_theta: Number of bits of precision for the rotations. Called
            $\beth$ in the reference.
        keep_bitsize: number of bits for keep register for coherent alias
            sampling. This can be determined from the PrepareTHC bloq. See
            https://github.com/quantumlib/Qualtran/issues/549
        kr1: block sizes for QROM erasure for outputting rotation angles. See Eq 34.
        kr2: block sizes for QROM erasure for outputting rotation angles. This
            is for the second QROM (eq 35)
        control_val: A control bit for the entire gate.
        eta: Optional THC leaf tensor of shape (M, N/2).

    Registers:
        succ: success flag qubit from uniform state preparation
        nu_eq_mp1: flag for if $\nu = M+1$
        mu: $\mu$ register.
        nu: $\nu$ register.
        plus_mn: Flag controlling swaps between mu and nu. Note that as per the
            Reference, the swaps are NOT performed as part of SELECT as they're
            accounted for during Prepare.
        plus_a / plus_b: plus state for controlled swaps on spins.
        sigma: ancilla register for alias sampling.
        rot: ancilla register for uniform superposition rotation.
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
    eta: Optional[np.ndarray] = field(
        default=None,
        eq=lambda x: None if x is None else (x.shape, x.dtype, x.tobytes()),
        repr=False,
    )

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
        n_spatial = self.num_spin_orb // 2
        n_angles = n_spatial - 1
        data_bitsize = n_angles * self.num_bits_theta
        data = bb.allocate(data_bitsize)
        if self.eta is not None:
            thc_rot = THCRotations.from_thc_leaf_tensor(
                self.eta, self.num_bits_theta, block_size=self.kr1, two_body_only=False
            )
        else:
            thc_rot = THCRotations(
                num_mu=self.num_mu,
                num_spin_orb=self.num_spin_orb,
                num_bits_theta=self.num_bits_theta,
                block_size=self.kr1,
                two_body_only=False,
            )
        data, mu, sys_a = bb.add(thc_rot, data=data, sel=mu, trg=sys_a)
        # Controlled Z_0
        (succ,), sys_b = bb.add(
            ApplyControlledZs(cvs=(1,), bitsize=self.num_spin_orb // 2),
            ctrls=np.asarray([succ]),
            system=sys_b,
        )
        # Undo rotations
        data, mu, sys_a = bb.add(thc_rot.adjoint(), data=data, sel=mu, trg=sys_a)
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
        if self.eta is not None:
            thc_rot_two_body = THCRotations.from_thc_leaf_tensor(
                self.eta, self.num_bits_theta, block_size=self.kr2, two_body_only=True
            )
        else:
            thc_rot_two_body = THCRotations(
                num_mu=self.num_mu,
                num_spin_orb=self.num_spin_orb,
                num_bits_theta=self.num_bits_theta,
                block_size=self.kr2,
                two_body_only=True,
            )
        data, mu, sys_a = bb.add(thc_rot_two_body, data=data, sel=mu, trg=sys_a)
        # Controlled Z_0
        (succ, nu_eq_mp1), sys_b = bb.add(
            ApplyControlledZs(cvs=(1, 0), bitsize=self.num_spin_orb // 2),
            ctrls=(succ, nu_eq_mp1),
            system=sys_b,
        )
        # Undo rotations
        data, mu, sys_a = bb.add(thc_rot_two_body.adjoint(), data=data, sel=mu, trg=sys_a)

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

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> Tuple['Bloq', 'AddControlledT']:
        from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv

        return get_ctrl_system_1bit_cv(
            self,
            ctrl_spec=ctrl_spec,
            current_ctrl_bit=self.control_val,
            get_ctrl_bloq_and_ctrl_reg_name=lambda cv: (evolve(self, control_val=cv), 'control'),
        )


@bloq_example
def _thc_rotations() -> THCRotations:
    thc_rotations = THCRotations(num_mu=10, num_spin_orb=8, num_bits_theta=12)
    return thc_rotations


_THC_ROTATIONS = BloqDocSpec(bloq_cls=THCRotations, examples=(_thc_rotations,))


@bloq_example
def _thc_sel() -> SelectTHC:
    num_mu = 10
    num_spin_orb = 2 * 4
    thc_sel = SelectTHC(
        num_mu=num_mu, num_spin_orb=num_spin_orb, num_bits_theta=12, keep_bitsize=10
    )
    return thc_sel


_THC_SELECT = BloqDocSpec(bloq_cls=SelectTHC, examples=(_thc_sel,))
