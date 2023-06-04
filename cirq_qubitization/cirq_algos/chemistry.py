r"""Gates for Qubitizing the Chemistry Model.

This follows section V. of the [Linear T Paper](https://arxiv.org/abs/1805.03662).

Here we implement SelectChem (Fig. 14.), SubPrepareChem (Fig. 15) and
PrepareChem (Fig. 16), which are specializations of the ab-initio Hamiltonian
for the case of a plane wave (dual) basis.

Under the Jordan-Wigner transformation the Hamiltonian is given by

$$
\def\Zvec{\overrightarrow{Z}}
\def\hop#1{#1_{p,\sigma} \Zvec #1_{q,\sigma}}
H = \sum_{p\ne q} \frac{T(p-q)}{2}\hop{X}+\hop{Y}
+ \sum_{(p\alpha)\ne (q\beta)} \frac{V(p-q)}{4} Z_{p\alpha}Z_{q\beta}
- \sum_{p\sigma} \frac{T(0) + U(p) + \sum_q V(p-q)}{2} Z_{p\sigma}
+ \sum_{p}(T(0) + U(p) + \sum_q \frac{V(p-q)}{2})\mathbb{1}
$$


This model consists of a PREPARE and SELECT operation where our selection operation has indices
for $p$, $\alpha$, $q$, and $\beta$ as well as two indicator bits $U$ and $V$. There are four cases
considered in both the PREPARE and SELECT operations corresponding to the terms in the Hamiltonian:

 - $U = 1, V = 0$ and $(p\alpha) = (q\beta)$, single-body Z
 - $U = 0, V=1$, $(p\alpha) \ne (q\beta)$, spin-spin ZZ term
 - $U = 0, V = 0, p < q ^ (\alpha = \beta)$, XZX term
 - $U = 0, V = 0, p > q ^ (\alpha = \beta)$, YZY term

See the documentation for `PrepareHubbard` and `SelectHubbard` for details.
"""
from functools import cached_property
from typing import Collection, Optional, Sequence, Tuple, Union

import cirq
import numpy as np
from attrs import frozen
from numpy.typing import NDArray
from openfermion.circuits.lcu_util import preprocess_lcu_coefficients_for_reversible_sampling

from cirq_qubitization import cirq_infra
from cirq_qubitization.cirq_algos.arithmetic_gates import LessThanEqualGate
from cirq_qubitization.cirq_algos.prepare_uniform_superposition import PrepareUniformSuperposition
from cirq_qubitization.cirq_algos.qrom import QROM
from cirq_qubitization.cirq_algos.select_and_prepare import PrepareOracle, SelectOracle
from cirq_qubitization.cirq_algos.state_preparation import PrepareUniformSuperposition
from cirq_qubitization.cirq_algos.swap_network import MultiTargetCSwap
from cirq_qubitization.cirq_infra import Registers, SelectionRegisters
from cirq_qubitization.cirq_infra.gate_with_registers import Registers, SelectionRegisters


class SelectChem(SelectOracle):
    r"""The SELECT operation optimized for the plane wave dual basis Hamiltonian

    Args:
        num_spin_orb: the number of spin-orbitals.
        control_val: Optional bit specifying the control value for constructing a controlled
            version of this gate. Defaults to None, which means no control.

    Registers:
        control: A control bit for the entire gate.
        theta: Whether to apply a negative phase.
        U: Whether we're applying the single-site part of the potential.
        V: Whether we're applying the pairwise part of the potential.
        p: First set of site indices, x component.
        alpha: First set of sites' spin indicator.
        q: Second set of site indices, x component.
        beta: Second set of sites' spin indicator.
        target: The system register to apply the select operation.

    References:
        Section V. and Fig. 14 of https://arxiv.org/abs/1805.03662.
    """

    def __init__(self, num_spin_orb: int, control_val: Optional[int] = None):
        self.num_spin_orb = num_spin_orb
        self.control_val = control_val
        if self.num_spin_orb % 2 != 0:
            raise NotImplementedError(f"num_spin_orb must be even: {self.num_spin_orb}.")
        self._num_spat_orb = self.num_spin_orb // 2

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
            p=(self._num_spat_orb.bit_length(), self._num_spat_orb),
            alpha=(1, 2),
            q=(self._num_spat_orb.bit_length(), self._num_spat_orb),
            beta=(1, 2),
        )

    @cached_property
    def target_registers(self) -> Registers:
        # 2 is for spin
        return Registers.build(target=self.num_spin_orb)

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
        p: Sequence[cirq.Qid],
        alpha: Sequence[cirq.Qid],
        q: Sequence[cirq.Qid],
        beta: Sequence[cirq.Qid],
        target: Sequence[cirq.Qid],
        control: Sequence[cirq.Qid] = (),
    ) -> cirq.OP_TREE:
        yield SelectedMajoranaFermionGate(
            selection_regs=SelectionRegisters.build(
                alpha=(1, 2), p=(self.registers['p'].bitsize, self._num_spat_orb)
            ),
            control_regs=self.control_registers,
            target_gate=cirq.Y,
        ).on_registers(control=control, p=p, alpha=alpha, target=target)

        yield MultiTargetCSwap.make_on(control=V, target_x=p, target_y=q)
        yield MultiTargetCSwap.make_on(control=V, target_x=alpha, target_y=beta)

        q_selection_regs = SelectionRegisters.build(
            beta=(1, 2), q=(self.registers['q'].bitsize, self._num_spat_orb)
        )
        yield SelectedMajoranaFermionGate(
            selection_regs=q_selection_regs, control_regs=self.control_registers, target_gate=cirq.X
        ).on_registers(control=control, q=q, beta=beta, target=target)

        yield MultiTargetCSwap.make_on(control=V, target_x=alpha, target_y=beta)
        yield MultiTargetCSwap.make_on(control=V, target_x=p, target_y=q)

        yield cirq.S(*control)
        yield cirq.Z(*(U)).controlled_by(*control)
        yield cirq.Z(*(V)).controlled_by(*control)
        yield cirq.Z(*(theta)).controlled_by(*control)

        target_qubits_for_apply_to_lth_gate = [
            target[q_selection_regs.to_flat_idx(s, q)]
            for s in range(2)
            for q in range(self._num_spat_orb)
        ]

        yield ApplyGateToLthQubit(
            selection_regs=SelectionRegisters.build(
                beta=(1, 2), q=(self.registers['p'].bitsize, self._num_spat_orb)
            ),
            nth_gate=lambda *_: cirq.Z,
            control_regs=Registers.build(control=1 + self.control_registers.bitsize),
        ).on_registers(
            q=q, control=[*V, *control], beta=beta, target=target_qubits_for_apply_to_lth_gate
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


@frozen
class SubPrepareChem(PrepareOracle):
    r"""Sub-prepare circuit for the plane wave dual Hamiltonian.

    This circuit acts like:

    $$
        \mathrm{SUBPREPARE}|0\rangle^{\otimes{2 + log N}} \rightarrow
        \sum_d^{N-1}\left(\tilde{U}(d)|\theta_d|1\rangle_U|0\rangle_V
        +\tilde{T}(d)|\theta_d^{(0)}\rangle|0\rangle_U|0\rangle_V
        +\tilde{V}(d)|\theta_d^{(1)}\rangle|0\rangle_U|1\rangle_V
        \right)|d\rangle
    $$

    where

    $$
        \tilde{U}(p) = \sqrt{\frac{Q(p)}{2\lambda}}
        \tilde{T}(p) = \sqrt{\frac{T(p)}{\lambda}}
        \tilde{V}(p) = \sqrt{\frac{V(p)}{4\lambda}}
        \theta_p = \frac{1-\sign(-Q(p))}{2}
        \theta_p^{(0)} = \frac{1-\sign(T(p))}{2}
        \theta_p^{(1)} = \frac{1-\sign(V(p))}{2}
    $$
    and
    $$
        Q(p) = |T(0)+U(p)+Vx(p)|.
        Vx(p) = \sum_q V(p-q)
    $$

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Section III.D. and Figure 15. Note there is an
        error in the circuit diagram in the paper (there should be no data:
        $\theta_{alt_l}$ gate in the QROM load.)
    """

    num_spin_orb: int
    theta: NDArray[np.int_]
    altU: NDArray[np.int_]
    altV: NDArray[np.int_]
    altp: NDArray[np.int_]
    keep: NDArray[np.int_]
    mu: int

    @classmethod
    def build_from_coefficients(
        cls,
        num_spin_orb: int,
        T: NDArray[np.float64],
        U: NDArray[np.float64],
        V: NDArray[np.float64],
        Vx: NDArray[np.float64],
        lambda_H,
        *,
        probability_epsilon: float = 1.0e-5,
    ) -> 'SubPrepareChem':
        # Inputs are $\tilde{U}(p)$, $\tilde{V}(p)$, $\tilde{T}(p)$ and corresponding $\theta_p$ values
        # We flatten the [U, V, p] data to find the LCU coefficients for alias sampling.
        # The values in alt are flattened indicies referring to the original data array.
        # We need to iterate through alt and extract (U, V, p) indicies
        num_spatial = num_spin_orb // 2
        assert len(T) == num_spatial
        assert len(U) == num_spatial
        assert len(V) == num_spatial
        assert len(Vx) == num_spatial
        # Stores the Tilde versions of T, U, and V defined in class docstring.
        coeffs = np.zeros((3, num_spatial), dtype=np.float64)
        # |00>_{UV}
        coeffs[0] = np.sqrt(np.abs(T) / lambda_H)
        # |01>_{UV}
        coeffs[1] = np.sqrt(np.abs(V) / (4 * lambda_H))
        # |10>_{UV}
        coeffs[2] = np.sqrt(np.abs(T[0] + U + Vx) / (2 * lambda_H))
        thetas = np.zeros((4, num_spatial), dtype=np.int8)
        # theta_p^0 |00>_{UV}
        thetas[0] = (1 - np.sign(T)) / 2
        # theta_p^1 |01>_{UV}
        thetas[1] = (1 - np.sign(V)) / 2
        # theta_p |10>_{UV}
        thetas[2] = (1 - np.sign(-(T[0] + U + Vx))) / 2
        thetas = thetas.reshape((2, 2, -1))
        assert np.allclose(thetas[0, 0], (1 - np.sign(T)) / 2)
        alt3, keep3, mu = preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=coeffs[:3].ravel(), epsilon=probability_epsilon
        )
        # Map alt indices back to unravelled form alt_l -> alt_{(U, V, p)}
        altUV, altp3 = np.unravel_index(alt3, (3, num_spatial))
        # Enlarged arrays with where U=1, V=1 will be all zeros
        altU = np.zeros(thetas.size, dtype=np.int_)
        altV = np.zeros(thetas.size, dtype=np.int_)
        altp = np.zeros(thetas.size, dtype=np.int_)
        keep = np.zeros(thetas.size, dtype=np.int_)
        # indices 2 from alt correspond to U = |10>
        altU[np.where(altUV == 2)] = 1
        # indices 1 from alt correspond to U = |01>
        altV[np.where(altUV == 1)] = 1
        altp[: 3 * num_spatial] = altp3
        keep[: 3 * num_spatial] = keep3
        assert np.all(altp < num_spin_orb)
        return SubPrepareChem(
            num_spin_orb=num_spin_orb,
            theta=thetas,
            altU=altU.reshape((2, 2, num_spatial)),
            altV=altV.reshape((2, 2, num_spatial)),
            altp=altp.reshape((2, 2, num_spatial)),
            keep=np.array(keep, dtype=np.int_).reshape((2, 2, num_spatial)),
            mu=mu,
        )

    @classmethod
    def build_from_plane_wave_dual(
        cls, num_spin_orb: int, *, probability_epsilon: float = 1.0e-5
    ) -> 'SubPrepareChem':
        pass

    def _attrs_post_init__(self):
        if self.num_spin_orb % 2 != 0:
            raise NotImplementedError(f"num_spin_orb must be even: {self.num_spin_orb}.")

    def __hash__(self):
        return hash(
            (self.num_spin_orb,)
            + tuple(self.altp.ravel())
            + tuple(self.altU.ravel())
            + tuple(self.altV.ravel())
            + tuple(self.keep.ravel())
            + (self.mu,)
        )

    @cached_property
    def selection_registers(self) -> SelectionRegisters:
        num_spatial = self.num_spin_orb // 2
        regs = SelectionRegisters.build(
            U=(1, 2), V=(1, 2), p=(num_spatial.bit_length() - 1, num_spatial)
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
        return Registers.build(
            sigma_mu=self.sigma_mu_bitsize,
            altU=1,
            altV=1,
            altp=self.num_spin_orb.bit_length() - 2,
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
        num_spat_orb = self.num_spin_orb // 2
        selU = qubit_regs["U"]
        selV = qubit_regs["V"]
        selp = qubit_regs["p"]
        altU = qubit_regs["altU"]
        altV = qubit_regs["altV"]
        altp = qubit_regs["altp"]
        theta = qubit_regs["theta"]
        keep = qubit_regs["keep"]
        less_than_equal = qubit_regs["less_than_equal"]
        sigma_mu = qubit_regs["sigma_mu"]
        yield PrepareUniformSuperposition(n=3).on_registers(controls=[], target=selU + selV)
        yield PrepareUniformSuperposition(n=num_spat_orb).on_registers(controls=[], target=selp)
        yield cirq.H.on_each(*sigma_mu)
        qrom = QROM.build([self.altU, self.altV, self.altp, self.keep, self.theta])
        yield qrom.on_registers(
            selection0=selU,
            selection1=selV,
            selection2=selp,
            target0=altU,
            target1=altV,
            target2=altp,
            target3=keep,
            target4=theta,
        )
        yield LessThanEqualGate([2] * self.mu, [2] * self.mu).on(*keep, *sigma_mu, *less_than_equal)
        yield MultiTargetCSwap.make_on(control=less_than_equal, target_x=altU, target_y=selU)
        yield MultiTargetCSwap.make_on(control=less_than_equal, target_x=altV, target_y=selV)
        yield MultiTargetCSwap.make_on(control=less_than_equal, target_x=altp, target_y=selp)
        yield LessThanEqualGate([2] * self.mu, [2] * self.mu).on(*keep, *sigma_mu, *less_than_equal)


class PrepareChem(PrepareOracle):
    r"""The PREPARE operation optimized for the 2D Hubbard model.

    In contrast to the arbitrary chemistry Hamiltonian, we:
     - explicitly consider the two dimensions of indices to permit optimization of the circuits.
     - dispense with the `theta` parameter.

    The circuit for implementing $\textit{PREPARE}_{Hubbard}$ has a T-cost of $O(log(N)$
    and uses $O(1)$ single qubit rotations.

    Args:
        x_dim: the number of sites along the x axis.
        y_dim: the number of sites along the y axis.
        t: coefficient for hopping terms in the Hubbard model hamiltonian.
        mu: coefficient for single body Z term and two-body ZZ terms in the Hubbard model
            hamiltonian.

    Registers:
        control: A control bit for the entire gate.
        U: Whether we're applying the single-site part of the potential.
        V: Whether we're applying the pairwise part of the potential.
        p_x: First set of site indices, x component.
        p_y: First set of site indices, y component.
        alpha: First set of sites' spin indicator.
        q_x: Second set of site indices, x component.
        q_y: Second set of site indices, y component.
        beta: Second set of sites' spin indicator.
        target: The system register to apply the select operation.
        junk: Temporary Work space.

    References:
        Section V. and Fig. 20 of https://arxiv.org/abs/1805.03662.
    """

    def __init__(self, num_spin_orb: int, control_val: Optional[int] = None):
        self.num_spin_orb = num_spin_orb
        self.control_val = control_val
        if self.num_spin_orb % 2 != 0:
            raise NotImplementedError(f"num_spin_orb must be even: {self.num_spin_orb}.")
        self._num_spat_orb = self.num_spin_orb // 2

    # @cached_property
    # def junk_registers(self) -> Registers:
    #     return Registers.build(temp=2)

    # @cached_property
    # def registers(self) -> Registers:
    #     return Registers([*self.selection_registers, *self.junk_registers])

    # @cached_property
    # def selection_registers(self) -> SelectionRegisters:
    #     return SelectionRegisters.build(
    #         theta=(1, 2),
    #         U=(1, 2),
    #         V=(1, 2),
    #         p=(self._num_spat_orb.bit_length(), self._num_spat_orb),
    #         alpha=(1, 2),
    #         q=(self._num_spat_orb.bit_length(), self._num_spat_orb),
    #         beta=(1, 2),
    #     )

    # def decompose_from_registers(
    #     self,
    #     theta: Sequence[cirq.Qid],
    #     U: Sequence[cirq.Qid],
    #     V: Sequence[cirq.Qid],
    #     p: Sequence[cirq.Qid],
    #     alpha: Sequence[cirq.Qid],
    #     q: Sequence[cirq.Qid],
    #     beta: Sequence[cirq.Qid],
    #     target: Sequence[cirq.Qid],
    #     control: Sequence[cirq.Qid] = (),
    # ) -> cirq.OP_TREE:
    # yield cirq.Ry(rads=2 * np.arccos(np.sqrt(self.t * N / qlambda))).on(*V)
    # yield cirq.Ry(rads=2 * np.arccos(np.sqrt(1 / 5))).on(*U).controlled_by(*V)
    # yield PrepareUniformSuperposition(self.x_dim).on_registers(controls=[], target=p_x)
    # yield PrepareUniformSuperposition(self.y_dim).on_registers(controls=[], target=p_y)
    # yield cirq.H.on_each(*temp)
    # yield cirq.CNOT(*U, *V)
    # yield cirq.X(*beta)
    # yield from [cirq.X(*V), cirq.H(*alpha).controlled_by(*V), cirq.CX(*V, *beta), cirq.X(*V)]
    # yield cirq.Circuit(cirq.CNOT.on_each([*zip([*p_x, *p_y, *alpha], [*q_x, *q_y, *beta])]))
    # yield MultiTargetCSwap.make_on(control=temp[:1], target_x=q_x, target_y=q_y)
    # yield AddMod(len(q_x), self.x_dim, add_val=1, cv=[0, 0]).on(*U, *V, *q_x)
    # yield MultiTargetCSwap.make_on(control=temp[:1], target_x=q_x, target_y=q_y)

    # and_target = cirq_infra.qalloc(1)
    # and_anc = cirq_infra.qalloc(1)
    # yield And(cv=(0, 0, 1)).on_registers(
    #     control=[*U, *V, temp[-1]], ancilla=and_anc, target=and_target
    # )
    # yield MultiTargetCSwap.make_on(
    #     control=and_target, target_x=[*p_x, *p_y, *alpha], target_y=[*q_x, *q_y, *beta]
    # )
    # yield And(cv=(0, 0, 1), adjoint=True).on_registers(
    #     control=[*U, *V, temp[-1]], ancilla=and_anc, target=and_target
    # )
    # cirq_infra.qfree([*and_anc, *and_target])

    # def _t_complexity_(self) -> t_complexity_protocol.TComplexity:
    #     return t_complexity_protocol.TComplexity(clifford=6 * self.num_spin_orb)
