r"""Gates for qubitizing the second quantized Ab-initio chemistry Hamiltonian in the plane-wave dual basis.

This follows section V. of the [Linear T Paper](https://arxiv.org/abs/1805.03662).

Here we implement SelectChem (Fig. 14.), SubPrepareChem (Fig. 15) and
PrepareChem (Fig. 16), which are specializations of the ab-initio Hamiltonian
for the case of a plane wave (dual) basis.

Under the Jordan-Wigner transformation the Hamiltonian is given by

$$
\def\Zvec{\overrightarrow{Z}}
\def\hop#1{#1_{p,\sigma} \Zvec #1_{q,\sigma}}
H = \sum_{p\ne q} \frac{T(p-q)}{2}(\hop{X}+\hop{Y})
+ \sum_{(p\alpha)\ne (q\beta)} \frac{V(p-q)}{4} Z_{p\alpha}Z_{q\beta}
- \sum_{p\sigma} \frac{T(0) + U(p) + \sum_q V(p-q)}{2} Z_{p\sigma}
+ \sum_{p}\left(T(0) + U(p) + \sum_q \frac{V(p-q)}{2}\right)\mathbb{1},
$$

Note that in the above expression $p$ is really a three-dimensional vector of
integers with each component $p_i \in [0, M-1]$ where $M$ is the
related to the real-space grid resolution and is expected as input by the user.
In total there are $N=2M^3$ spin-orbitals.  In what follows we often label
registers with a single label $(e.g. |p\rangle)$, but this should be understood
to mean $|\vec{p}\rangle = |p_x\rangle|p_y\rangle|p_z\rangle$ with each register
of size $\log M$.  This model can be constructed using the functions provided in
`openfermion.hamiltonians.plane_wave_hamiltonian`.


This model consists of a PREPARE and SELECT operation where our selection operation has indices
for $p$, $\alpha$, $q$, and $\beta$ as well as two indicator bits $U$ and $V$. There are four cases
considered in both the PREPARE and SELECT operations corresponding to the terms in the Hamiltonian:

 - $U = 1, V = 0$ and $(p\alpha) = (q\beta)$, single-body $Z$
 - $U = 0, V=1$, $(p\alpha) \ne (q\beta)$, spin-spin $ZZ$ term
 - $U = 0, V = 0, p < q \wedge (\alpha = \beta)$, $XZX$ term
 - $U = 0, V = 0, p > q \wedge (\alpha = \beta)$, $YZY$ term

See the documentation for `PrepareHubbard` and `SelectHubbard` for more details.
"""
import math
from functools import cached_property
from typing import Collection, List, Optional, Sequence, Tuple, Union

import cirq
import numpy as np
from attrs import frozen
from numpy.typing import NDArray
from openfermion.circuits.lcu_util import preprocess_lcu_coefficients_for_reversible_sampling

from cirq_qubitization import cirq_infra
from cirq_qubitization.cirq_algos.apply_gate_to_lth_target import ApplyGateToLthQubit
from cirq_qubitization.cirq_algos.arithmetic_gates import AddMod, LessThanEqualGate
from cirq_qubitization.cirq_algos.prepare_uniform_superposition import PrepareUniformSuperposition
from cirq_qubitization.cirq_algos.qrom import QROM
from cirq_qubitization.cirq_algos.select_and_prepare import PrepareOracle, SelectOracle
from cirq_qubitization.cirq_algos.selected_majorana_fermion import SelectedMajoranaFermionGate
from cirq_qubitization.cirq_algos.state_preparation import PrepareUniformSuperposition
from cirq_qubitization.cirq_algos.swap_network import MultiTargetCSwap
from cirq_qubitization.cirq_algos.unary_iteration import UnaryIterationGate
from cirq_qubitization.cirq_infra import Registers, SelectionRegisters
from cirq_qubitization.cirq_infra.gate_with_registers import Registers, SelectionRegisters


class SelectChem(SelectOracle):
    r"""The SELECT operation optimized for the plane wave dual basis Hamiltonian

    Args:
        num_spin_orb: the number of spin-orbitals.
        control_val: Optional bit specifying the control value for constructing a controlled
            version of this gate. Defaults to None, which means no control.

    Parameters:
        control: A control bit for the entire gate.
        theta: Whether to apply a negative phase.
        U: Flag to apply parts of Hamiltonian. See module docstring.
        V: Flag to apply parts of Hamiltonian. See module docstring.
        p: Spatial orbital index.
        alpha: Spin index for orbital p.
        q: Spatial orbital index.
        beta: Spin index for orbital q.
        target: The system register to apply the select operation.

    References:
        Section V. Eq. 44 and Fig. 14 of https://arxiv.org/abs/1805.03662.
    """

    def __init__(self, M: int, control_val: Optional[int] = None):
        self.M = M
        self.control_val = control_val
        self._num_spat_orb = self.M**3

    @cached_property
    def control_registers(self) -> Registers:
        registers = [] if self.control_val is None else [cirq_infra.Register('control', 1)]
        return cirq_infra.Registers(registers)

    @cached_property
    def selection_registers(self) -> SelectionRegisters:
        return SelectionRegisters.build(
            theta=(1, 2),
            U=(1, 2),
            V=(1, 2),
            px=((self.M - 1).bit_length(), self.M),
            py=((self.M - 1).bit_length(), self.M),
            pz=((self.M - 1).bit_length(), self.M),
            alpha=(1, 2),
            qx=((self.M - 1).bit_length(), self.M),
            qy=((self.M - 1).bit_length(), self.M),
            qz=((self.M - 1).bit_length(), self.M),
            beta=(1, 2),
        )

    @cached_property
    def target_registers(self) -> Registers:
        # 2 is for spin
        return Registers.build(target=2 * self._num_spat_orb)

    @cached_property
    def registers(self) -> Registers:
        return Registers(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )

    def decompose_from_registers(
        self,
        theta: Sequence[cirq.Qid],
        U: Sequence[cirq.Qid],
        V: Sequence[cirq.Qid],
        px: Sequence[cirq.Qid],
        py: Sequence[cirq.Qid],
        pz: Sequence[cirq.Qid],
        alpha: Sequence[cirq.Qid],
        qx: Sequence[cirq.Qid],
        qy: Sequence[cirq.Qid],
        qz: Sequence[cirq.Qid],
        beta: Sequence[cirq.Qid],
        target: Sequence[cirq.Qid],
        control: Sequence[cirq.Qid] = (),
    ) -> cirq.OP_TREE:
        yield SelectedMajoranaFermionGate(
            selection_regs=SelectionRegisters.build(
                alpha=(1, 2),
                px=((self.M - 1).bit_length(), self.M),
                py=((self.M - 1).bit_length(), self.M),
                pz=((self.M - 1).bit_length(), self.M),
            ),
            control_regs=self.control_registers,
            target_gate=cirq.Y,
        ).on_registers(control=control, px=px, py=py, pz=pz, alpha=alpha, target=target)

        yield MultiTargetCSwap.make_on(control=V, target_x=px + py + pz, target_y=qx + qy + qz)
        yield MultiTargetCSwap.make_on(control=V, target_x=alpha, target_y=beta)

        q_selection_regs = SelectionRegisters.build(
            beta=(1, 2),
            qx=((self.M - 1).bit_length(), self.M),
            qy=((self.M - 1).bit_length(), self.M),
            qz=((self.M - 1).bit_length(), self.M),
        )
        yield SelectedMajoranaFermionGate(
            selection_regs=q_selection_regs, control_regs=self.control_registers, target_gate=cirq.X
        ).on_registers(control=control, qx=qx, qy=qy, qz=qz, beta=beta, target=target)

        yield MultiTargetCSwap.make_on(control=V, target_x=alpha, target_y=beta)
        yield MultiTargetCSwap.make_on(control=V, target_x=px + py + pz, target_y=qx + qy + qz)

        yield cirq.S(*control)
        yield cirq.Z(*(U)).controlled_by(*control)
        yield cirq.Z(*(V)).controlled_by(*control)
        yield cirq.Z(*(theta)).controlled_by(*control)

        target_qubits_for_apply_to_lth_gate = [
            target[q_selection_regs.to_flat_idx(s, qx, qy, qz)]
            for s in range(2)
            for qx in range(self.M)
            for qy in range(self.M)
            for qz in range(self.M)
        ]

        yield ApplyGateToLthQubit(
            selection_regs=SelectionRegisters.build(
                beta=(1, 2),
                qx=((self.M - 1).bit_length(), self.M),
                qy=((self.M - 1).bit_length(), self.M),
                qz=((self.M - 1).bit_length(), self.M),
            ),
            nth_gate=lambda *_: cirq.Z,
            control_regs=Registers.build(control=1 + self.control_registers.bitsize),
        ).on_registers(
            qx=qx,
            qy=qy,
            qz=qz,
            control=[*V, *control],
            beta=beta,
            target=target_qubits_for_apply_to_lth_gate,
        )

    def controlled(
        self,
        num_controls: int = None,
        control_values: Sequence[Union[int, Collection[int]]] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> 'SelectHubbard':
        if num_controls is None:
            num_controls = 1
        if control_values is None:
            control_values = [1] * num_controls
        if len(control_values) == 1 and self.control_val is None:
            return SelectChem(self.num_spin_orb, control_val=control_values[0])
        raise NotImplementedError(f'Cannot create a controlled version of {self}')


@cirq.value_equality()
@frozen
class SubPrepareChem(PrepareOracle):
    r"""Sub-prepare circuit for the plane wave dual Hamiltonian.

    This circuit acts on a state like:

    $$
        \mathrm{SUBPREPARE}|0\rangle^{\otimes{2 + \log N}} \rightarrow
        \sum_d^{N-1}\left(\tilde{U}(d)|\theta_d\rangle|1\rangle_U|0\rangle_V
        +\tilde{T}(d)|\theta_d^{(0)}\rangle|0\rangle_U|0\rangle_V
        +\tilde{V}(d)|\theta_d^{(1)}\rangle|0\rangle_U|1\rangle_V
        \right)|d\rangle
    $$

    where

    $$
        \tilde{U}(p) = \sqrt{\frac{Q(p)}{2\lambda}}\\
        \tilde{T}(p) = \sqrt{\frac{T(p)}{\lambda}}\\
        \tilde{V}(p) = \sqrt{\frac{V(p)}{4\lambda}}\\
        \theta_p = \frac{1-\mathrm{sign}(-Q(p))}{2}\\
        \theta_p^{(0)} = \frac{1-\mathrm{sign}(T(p))}{2}\\
        \theta_p^{(1)} = \frac{1-\mathrm{sign}(V(p))}{2}\\
    $$
    and
    $$
        Q(p) = |T(0)+U(p)+V_x(p)|\\
        V_x(p) = \sum_q V(p-q)
    $$

    Args:
        num_spin_orbs: number of spin orbitals (typically called N in literature.)
        theta: numpy array of theta values of shape [2, 2, M, M, M]
        altU: numpy array of alternate U values of shape [2, 2, M, M, M]
        altV: numpy array of alternate V values of shape [2, 2, M, M, M]
        altpxyz: numpy array of alternate indices p = (px,py,pz) values of shape [2, 2, M, M, M]
        keep: numpy array of keep values of shape [2, 2, M, M, M]
        mu: Exponent to divide the items in keep by in order to obtain a
        probability. See `preprocess_lcu_coefficients_for_reversible_alias_sampling`.

    Parameters:
        theta: Whether to apply a negative phase.
        U: Flag to apply parts of Hamiltonian. See module docstring.
        V: Flag to apply parts of Hamiltonian. See module docstring.
        p: (vector) Spatial orbital index, p = (px, py, pz)

    References:
        Section V. and Fig. 14 of https://arxiv.org/abs/1805.03662.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Section III.D. and Figure 15. Note there is an
        error in the circuit diagram in the paper (there should be no data:
        $\theta_{alt_l}$ gate in the QROM load.)
    """

    M: int
    theta_l: NDArray[np.int_]
    altU: NDArray[np.int_]
    altV: NDArray[np.int_]
    altp: List[NDArray[np.int_]]
    keep: NDArray[np.int_]
    mu: int

    @classmethod
    def build_from_coefficients(
        cls,
        T: NDArray[np.float64],
        U: NDArray[np.float64],
        V: NDArray[np.float64],
        Vx: NDArray[np.float64],
        *,
        ndim: int = 3,
        probability_epsilon: float = 1.0e-5,
    ) -> 'SubPrepareChem':
        r"""Build SUBPREPARE circuit from Hamiltonian coefficients.

        Args:
            T: kinetic energy matrix elements.
            U: external potential matrix elements.
            V: specifying electron-electron interaction matrix elements.
            Vx: "Exchage potential" matrix $\sum_q V(p-q)$.
            probability_epsilon: The epsilon that we use to set the precision of of the
                subprepare approximation. This parameter is called mu in the linear T paper.

        Returns:
            prepare: SUBPREPARE circuit with alt and keep values built from
                matrix elements via coherent alias sampling.
        """
        assert ndim == 3, "Only 3D systems are allowed."
        num_spatial_orbs = len(T)
        assert len(U) == num_spatial_orbs
        assert len(V) == num_spatial_orbs
        assert len(Vx) == num_spatial_orbs
        # Number of orbitals in each direction assuming a cubic box
        M = math.ceil(math.log(num_spatial_orbs, 3))
        assert M**3 == num_spatial_orbs, f"{M} vs {num_spatial_orbs}"
        # Eq. 52 in linear T paper
        lambda_H = np.sum(np.abs(T)) + np.sum(np.abs(U)) + np.sum(np.sum(Vx))
        # Stores the Tilde versions of T, U, and V defined in class docstring.
        coeffs = np.zeros((3, num_spatial_orbs), dtype=np.float64)
        # |00>_{UV}
        coeffs[0] = np.sqrt(np.abs(T) / lambda_H)
        # |01>_{UV}
        coeffs[1] = np.sqrt(np.abs(V) / (4 * lambda_H))
        # |10>_{UV}
        coeffs[2] = np.sqrt(np.abs(T[0] + U + Vx) / (2 * lambda_H))
        thetas = np.zeros((4, num_spatial_orbs), dtype=np.int8)
        # theta_p^0 |00>_{UV}
        thetas[0] = (1 - np.sign(T)) // 2
        # theta_p^1 |01>_{UV}
        thetas[1] = (1 - np.sign(V)) // 2
        # theta_p |10>_{UV}
        thetas[2] = (1 - np.sign(-(T[0] + U + Vx))) // 2
        thetas = thetas.reshape((2, 2, M, M, M))
        assert np.allclose(thetas[0, 0].ravel(), (1 - np.sign(T)) / 2)
        assert np.allclose(thetas[0, 1].ravel(), (1 - np.sign(V)) / 2)
        alt3, keep3, mu = preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=coeffs.ravel(), epsilon=probability_epsilon
        )
        # Map alt indices back to unravelled form alt_l -> alt_{(U, V, p)}
        altUV, altp3 = np.unravel_index(alt3, (3, num_spatial_orbs))
        assert np.all(altp3 < num_spatial_orbs)
        # Enlarged arrays with where U=1, V=1 will be all zeros
        altU = np.zeros(thetas.size, dtype=np.int_)
        altV = np.zeros(thetas.size, dtype=np.int_)
        altpx = np.zeros(thetas.size, dtype=np.int_)
        altpy = np.zeros(thetas.size, dtype=np.int_)
        altpz = np.zeros(thetas.size, dtype=np.int_)
        keep = np.zeros(thetas.size, dtype=np.int_)
        # Map altp indices back to unravelled form alt_p -> alt_{(U, V, px, py, pz)}
        _altpx, _altpy, _altpz = np.unravel_index(altp3, (M,) * 3)
        altpx[: 3 * num_spatial_orbs] = _altpx
        altpy[: 3 * num_spatial_orbs] = _altpy
        altpz[: 3 * num_spatial_orbs] = _altpz
        # indices 2 from alt correspond to U = |10>
        altU[np.where(altUV == 2)] = 1
        # indices 1 from alt correspond to U = |01>
        altV[np.where(altUV == 1)] = 1
        keep[: 3 * num_spatial_orbs] = keep3
        data_shape = (2, 2, M, M, M)
        return SubPrepareChem(
            M=M,
            theta_l=thetas,
            altU=altU.reshape(data_shape),
            altV=altV.reshape(data_shape),
            altp=[altpx.reshape(data_shape), altpy.reshape(data_shape), altpz.reshape(data_shape)],
            keep=np.array(keep).reshape(data_shape),
            mu=mu,
        )

    @cached_property
    def selection_registers(self) -> SelectionRegisters:
        M = self.altU.shape[-1]
        regs = SelectionRegisters.build(
            U=(1, 2),
            V=(1, 2),
            px=((M - 1).bit_length(), M),
            py=((M - 1).bit_length(), M),
            pz=((M - 1).bit_length(), M),
        )
        return regs

    @cached_property
    def sigma_mu_bitsize(self) -> int:
        return self.mu

    @cached_property
    def alternates_bitsize(self) -> int:
        return sum(reg.bitsize for reg in self.selection_registers) + 1

    @cached_property
    def keep_bitsize(self) -> int:
        return self.mu

    @cached_property
    def junk_registers(self) -> Registers:
        M = self.altU.shape[-1]
        return Registers.build(
            sigma_mu=self.sigma_mu_bitsize,
            altU=1,
            altV=1,
            altpx=(M - 1).bit_length(),
            altpy=(M - 1).bit_length(),
            altpz=(M - 1).bit_length(),
            keep=self.keep_bitsize,
            less_than_equal=1,
        )

    @cached_property
    def theta_register(self) -> Registers:
        return Registers.build(theta=1)

    @cached_property
    def registers(self) -> Registers:
        return Registers([*self.theta_register, *self.selection_registers, *self.junk_registers])

    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        selU = qubit_regs["U"]
        selV = qubit_regs["V"]
        selpx = qubit_regs["px"]
        selpy = qubit_regs["py"]
        selpz = qubit_regs["pz"]
        altU = qubit_regs["altU"]
        altV = qubit_regs["altV"]
        altpx = qubit_regs["altpx"]
        altpy = qubit_regs["altpy"]
        altpz = qubit_regs["altpz"]
        theta = qubit_regs["theta"]
        keep = qubit_regs["keep"]
        less_than_equal = qubit_regs["less_than_equal"]
        sigma_mu = qubit_regs["sigma_mu"]
        yield PrepareUniformSuperposition(n=3).on_registers(controls=[], target=selU + selV)
        yield PrepareUniformSuperposition(n=self.M**3).on_registers(
            controls=[], target=selpx + selpy + selpz
        )
        yield cirq.H.on_each(*sigma_mu)
        qrom = QROM.build(self.altU, self.altV, *self.altp, self.keep, self.theta_l)
        yield qrom.on_registers(
            selection0=selU,
            selection1=selV,
            selection2=selpx,
            selection3=selpy,
            selection4=selpz,
            target0=altU,
            target1=altV,
            target2=altpx,
            target3=altpy,
            target4=altpz,
            target5=keep,
            target6=theta,
        )
        yield LessThanEqualGate([2] * self.mu, [2] * self.mu).on(*keep, *sigma_mu, *less_than_equal)
        yield MultiTargetCSwap.make_on(control=less_than_equal, target_x=altU, target_y=selU)
        yield MultiTargetCSwap.make_on(control=less_than_equal, target_x=altV, target_y=selV)
        yield MultiTargetCSwap.make_on(
            control=less_than_equal, target_x=altpx + altpy + altpz, target_y=selpx + selpy + selpz
        )
        yield LessThanEqualGate([2] * self.mu, [2] * self.mu).on(*keep, *sigma_mu, *less_than_equal)

    def _value_equality_values_(self):
        return self.M


@frozen
class IndexedAddMod(UnaryIterationGate):
    selection_regs: SelectionRegisters
    mod: int
    control_regs: Registers = Registers([])

    @cached_property
    def control_registers(self) -> Registers:
        return self.control_regs

    @cached_property
    def selection_registers(self) -> SelectionRegisters:
        return self.selection_regs

    @cached_property
    def selection_bitsizes(self) -> List[int]:
        return [reg.bitsize for reg in self.selection_regs]

    @cached_property
    def target_registers(self) -> Registers:
        return Registers.build(**{f'target{i}': sb for i, sb in enumerate(self.selection_bitsizes)})

    def nth_operation(self, control: cirq.Qid, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        selection_idx = tuple(qubit_regs[reg.name] for reg in self.selection_registers)
        target_regs = tuple(v for k, v in qubit_regs.items() if k in self.target_registers)
        yield AddMod(self.selection_bitsizes[0], self.mod, add_val=selection_idx[0]).on(
            *target_regs[0]
        )
        yield AddMod(self.selection_bitsizes[1], self.mod, add_val=selection_idx[1]).on(
            *target_regs[1]
        )
        yield AddMod(self.selection_bitsizes[2], self.mod, add_val=selection_idx[2]).on(
            *target_regs[2]
        )


@cirq.value_equality()
@frozen
class PrepareChem(PrepareOracle):
    r"""Prepare circuit for the plane wave dual Hamiltonian.

    This circuit acts on a state like:

    $$
        \mathrm{PREPARE}|0\rangle^{\otimes{3 + 2log N}} \rightarrow
        \sum_{p\sigma} \left(
        \tilde{U}(p)|\theta_p|1\rangle_U|0\rangle_V|p\alpha q\beta\rangle
        +\tilde{T}(p-q)|\theta_{p-q}^{(0)}\rangle|0\rangle_U|0\rangle_V |p\alpha q\beta\rangle
        +\tilde{V}(p-q)|\theta_{p-q}^{(1)}\rangle|0\rangle_U|1\rangle_V |p\alpha q\beta\rangle
        \right)
    $$

    See SubPrepareChem docstring for definitions of terms.

    Args:
        M : number of spin orbitals (typically called N in literature.)
        T: kinetic energy matrix elements of size (M**3).
        U: external potential matrix elements of size (M**3).
        V: specifying electron-electron interaction matrix elements of size (M**3).
        Vx: "Exchage potential" matrix $\sum_q V(p-q)$ of size (M**3).
        probability_epsilon: The epsilon that we use to set the precision of of the
            subprepare approximation. This parameter is called mu in the linear T paper.

    Parameters:
        theta: Whether to apply a negative phase.
        U: Flag to apply parts of Hamiltonian. See module docstring.
        V: Flag to apply parts of Hamiltonian. See module docstring.
        p: Spatial orbital index.
        alpha: Spin index for orbital p.
        q: Spatial orbital index.
        beta: Spin index for orbital q.

    References:
        Section V. and Fig. 14 of https://arxiv.org/abs/1805.03662.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Section III.D. and Figure 15. Note there is an
        error in the circuit diagram in the paper (there should be no data:
        $\theta_{alt_l}$ gate in the QROM load.)
    """

    M: int
    T: NDArray
    U: NDArray
    V: NDArray
    Vx: NDArray
    ndim: int = 3

    @cached_property
    def registers(self) -> Registers:
        return Registers([*self.selection_registers, *self.junk_registers])

    @cached_property
    def selection_registers(self) -> SelectionRegisters:
        return SelectionRegisters.build(
            theta=(1, 2),
            U=(1, 2),
            V=(1, 2),
            px=((self.M - 1).bit_length(), self.M),
            py=((self.M - 1).bit_length(), self.M),
            pz=((self.M - 1).bit_length(), self.M),
            alpha=(1, 2),
            qx=((self.M - 1).bit_length(), self.M),
            qy=((self.M - 1).bit_length(), self.M),
            qz=((self.M - 1).bit_length(), self.M),
            beta=(1, 2),
        )

    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        selU = qubit_regs["U"]
        selV = qubit_regs["V"]
        selpx = qubit_regs["px"]
        selpy = qubit_regs["py"]
        selpz = qubit_regs["pz"]
        selqx = qubit_regs["qx"]
        selqy = qubit_regs["qy"]
        selqz = qubit_regs["qz"]
        theta = qubit_regs["theta"]
        alpha = qubit_regs["alpha"]
        beta = qubit_regs["beta"]
        spc = SubPrepareChem.build_from_coefficients(T=self.T, U=self.U, V=self.V, Vx=self.Vx)
        sub_prep_ancilla = {reg.name: cirq_infra.qalloc(reg.bitsize) for reg in spc.junk_registers}
        yield spc.on_registers(**qubit_regs, **sub_prep_ancilla)
        yield cirq.H(*alpha)
        num_spat_orb = self.M**self.ndim
        yield PrepareUniformSuperposition(n=num_spat_orb, cvs=[0]).on_registers(
            controls=[*selU], target=[*selqx, *selqy, *selqz]
        )
        yield cirq.H(*beta).controlled_by(*selV),
        p_bitsize = self.ndim * self.selection_registers["px"].bitsize
        yield cirq.H.controlled(
            num_controls=p_bitsize + 1, control_values=[1] + [0] * p_bitsize
        ).on(*selV, *selpx, *selpy, *selpz, *beta)
        yield cirq.X.controlled(
            num_controls=p_bitsize + 1, control_values=[1] + [0] * p_bitsize
        ).on(*selV, *selpx, *selpy, *selpz, *beta)
        yield cirq.CNOT(*alpha, *beta)
        yield MultiTargetCSwap.make_on(
            control=selU, target_x=selpx + selpy + selpz, target_y=selqx + selqy + selqz
        )
        yield IndexedAddMod(
            selection_regs=SelectionRegisters.build(
                qx=((self.M - 1).bit_length(), self.M),
                qy=((self.M - 1).bit_length(), self.M),
                qz=((self.M - 1).bit_length(), self.M),
            ),
            mod=self.M,
        ).on(*selqx, *selqy, *selqz, *selpx, *selpy, *selpz)

    def _value_equality_values_(self):
        return (self.M, self.ndim)
