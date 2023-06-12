r"""Gates for Qubitizing the Hubbard Model.

This follows section V. of the [Linear T Paper](https://arxiv.org/abs/1805.03662).

The 2D Hubbard model is a special case of the electronic structure Hamiltonian
restricted to spins on a planar grid.

$$
H = -t \sum_{\langle p,q \rangle, \sigma} a_{p,\sigma}^\dagger a_{q,\sigma}
    + \frac{u}{2} \sum_{p,\alpha\ne\beta} n_{p, \alpha} n_{p, \beta}
$$

Under the Jordan-Wigner transformation this is

$$
\def\Zvec{\overrightarrow{Z}}
\def\hop#1{#1_{p,\sigma} \Zvec #1_{q,\sigma}}
H = -\frac{t}{2} \sum_{\langle p,q \rangle, \sigma} (\hop{X} + \hop{Y})
  + \frac{u}{8} \sum_{p,\alpha\ne\beta} Z_{p,\alpha}Z_{p,\beta}
  - \frac{u}{4} \sum_{p,\sigma} Z_{p,\sigma} + \frac{uN}{4}\mathbb{1}
$$


This model consists of a PREPARE and SELECT operation where our selection operation has indices
for $p$, $\alpha$, $q$, and $\beta$ as well as two indicator bits $U$ and $V$. There are four cases
considered in both the PREPARE and SELECT operations corresponding to the terms in the Hamiltonian:

 - $U=1$, single-body Z
 - $V=1$, spin-spin ZZ term
 - $p<q$, XZX term
 - $p>q$, YZY term.

See the documentation for `PrepareHubbard` and `SelectHubbard` for details.
"""
from functools import cached_property
from typing import Collection, Optional, Sequence, Tuple, Union

import cirq
import numpy as np
from attrs import frozen

from cirq_qubitization import cirq_infra
from cirq_qubitization.cirq_algos import (
    AddMod,
    And,
    ApplyGateToLthQubit,
    MultiTargetCSwap,
    PrepareOracle,
    PrepareUniformSuperposition,
    QubitizationWalkOperator,
    SelectedMajoranaFermionGate,
    SelectOracle,
)
from cirq_qubitization.cirq_infra import Registers, SelectionRegisters


@frozen(cache_hash=True)
class SelectHubbard(SelectOracle):
    r"""The SELECT operation optimized for the 2D Hubbard model.

    In contrast to the arbitrary chemistry Hamiltonian, we:
     - explicitly consider the two dimensions of indices to permit optimization of the circuits.
     - dispense with the `theta` parameter.

    If neither $U$ nor $V$ is set we apply the kinetic terms of the Hamiltonian:

    $$
    -\hop{X} \quad p < q \\
    -\hop{Y} \quad p > q
    $$

    If $U$ is set we know $(p,\alpha)=(q,\beta)$ and apply the single-body term: $-Z_{p,\alpha}$.
    If $V$ is set we know $p=q, \alpha=0$, and $\beta=1$ and apply the spin term:
    $Z_{p,\alpha}Z_{p,\beta}$

    The circuit for implementing $\textit{C-SELECT}_{Hubbard}$ has a T-cost of $10 * N + log(N)$
    and $0$ rotations.


    Args:
        x_dim: the number of sites along the x axis.
        y_dim: the number of sites along the y axis.
        control_val: Optional bit specifying the control value for constructing a controlled
            version of this gate. Defaults to None, which means no control.

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

    References:
        Section V. and Fig. 19 of https://arxiv.org/abs/1805.03662.
    """

    x_dim: int
    y_dim: int
    control_val: Optional[int] = None

    def __attrs_post_init__(self):
        if self.x_dim != self.y_dim:
            raise NotImplementedError("Currently only supports the case where x_dim=y_dim.")

    @cached_property
    def control_registers(self) -> Registers:
        registers = [] if self.control_val is None else [cirq_infra.Register('control', 1)]
        return cirq_infra.Registers(registers)

    @cached_property
    def selection_registers(self) -> SelectionRegisters:
        return SelectionRegisters.build(
            U=(1, 2),
            V=(1, 2),
            p_x=((self.x_dim - 1).bit_length(), self.x_dim),
            p_y=((self.y_dim - 1).bit_length(), self.y_dim),
            alpha=(1, 2),
            q_x=((self.x_dim - 1).bit_length(), self.x_dim),
            q_y=((self.y_dim - 1).bit_length(), self.y_dim),
            beta=(1, 2),
        )

    @cached_property
    def target_registers(self) -> Registers:
        return Registers.build(target=self.x_dim * self.y_dim * 2)

    @cached_property
    def registers(self) -> Registers:
        return Registers(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )

    def decompose_from_registers(
        self,
        context: cirq.DecompositionContext,
        U: Sequence[cirq.Qid],
        V: Sequence[cirq.Qid],
        p_x: Sequence[cirq.Qid],
        p_y: Sequence[cirq.Qid],
        alpha: Sequence[cirq.Qid],
        q_x: Sequence[cirq.Qid],
        q_y: Sequence[cirq.Qid],
        beta: Sequence[cirq.Qid],
        target: Sequence[cirq.Qid],
        control: Sequence[cirq.Qid] = (),
    ) -> cirq.OP_TREE:
        yield SelectedMajoranaFermionGate(
            selection_regs=SelectionRegisters.build(
                alpha=(1, 2),
                p_y=(self.registers['p_y'].bitsize, self.y_dim),
                p_x=(self.registers['p_x'].bitsize, self.x_dim),
            ),
            control_regs=self.control_registers,
            target_gate=cirq.Y,
        ).on_registers(control=control, p_x=p_x, p_y=p_y, alpha=alpha, target=target)

        yield MultiTargetCSwap.make_on(control=V, target_x=p_x, target_y=q_x)
        yield MultiTargetCSwap.make_on(control=V, target_x=p_y, target_y=q_y)
        yield MultiTargetCSwap.make_on(control=V, target_x=alpha, target_y=beta)

        q_selection_regs = SelectionRegisters.build(
            beta=(1, 2),
            q_y=(self.registers['q_y'].bitsize, self.y_dim),
            q_x=(self.registers['q_x'].bitsize, self.x_dim),
        )
        yield SelectedMajoranaFermionGate(
            selection_regs=q_selection_regs, control_regs=self.control_registers, target_gate=cirq.X
        ).on_registers(control=control, q_x=q_x, q_y=q_y, beta=beta, target=target)

        yield MultiTargetCSwap.make_on(control=V, target_x=alpha, target_y=beta)
        yield MultiTargetCSwap.make_on(control=V, target_x=p_y, target_y=q_y)
        yield MultiTargetCSwap.make_on(control=V, target_x=p_x, target_y=q_x)

        yield cirq.S(*control) ** -1 if control else cirq.global_phase_operation(
            -1j
        )  # Fix errant i from XY=iZ
        yield cirq.Z(*U).controlled_by(*control)  # Fix errant -1 from multiple pauli applications

        target_qubits_for_apply_to_lth_gate = [
            target[q_selection_regs.to_flat_idx(1, qy, qx)]
            for qx in range(self.x_dim)
            for qy in range(self.y_dim)
        ]

        yield ApplyGateToLthQubit(
            selection_regs=SelectionRegisters.build(
                q_y=(self.registers['q_y'].bitsize, self.y_dim),
                q_x=(self.registers['q_x'].bitsize, self.x_dim),
            ),
            nth_gate=lambda *_: cirq.Z,
            control_regs=Registers.build(control=1 + self.control_registers.bitsize),
        ).on_registers(
            q_x=q_x, q_y=q_y, control=[*V, *control], target=target_qubits_for_apply_to_lth_gate
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
            return SelectHubbard(self.x_dim, self.y_dim, control_val=control_values[0])
        raise NotImplementedError(f'Cannot create a controlled version of {self}')

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        info = super()._circuit_diagram_info_(args)
        if self.control_val is None:
            return info
        ctrl = ('@' if self.control_val else '@(0)',)
        return info.with_wire_symbols(ctrl + info.wire_symbols[0:1] + info.wire_symbols[2:])


@frozen(cache_hash=True)
class PrepareHubbard(PrepareOracle):
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

    x_dim: int
    y_dim: int
    t: int
    mu: int

    def __attrs_post_init__(self):
        if self.x_dim != self.y_dim:
            raise NotImplementedError("Currently only supports the case where x_dim=y_dim.")

    @cached_property
    def selection_registers(self) -> SelectionRegisters:
        return SelectionRegisters.build(
            U=(1, 2),
            V=(1, 2),
            p_x=((self.x_dim - 1).bit_length(), self.x_dim),
            p_y=((self.y_dim - 1).bit_length(), self.y_dim),
            alpha=(1, 2),
            q_x=((self.x_dim - 1).bit_length(), self.x_dim),
            q_y=((self.y_dim - 1).bit_length(), self.y_dim),
            beta=(1, 2),
        )

    @cached_property
    def junk_registers(self) -> Registers:
        return Registers.build(temp=2)

    @cached_property
    def registers(self) -> Registers:
        return Registers([*self.selection_registers, *self.junk_registers])

    def decompose_from_registers(
        self,
        context: cirq.DecompositionContext,
        U: Sequence[cirq.Qid],
        V: Sequence[cirq.Qid],
        p_x: Sequence[cirq.Qid],
        p_y: Sequence[cirq.Qid],
        alpha: Sequence[cirq.Qid],
        q_x: Sequence[cirq.Qid],
        q_y: Sequence[cirq.Qid],
        beta: Sequence[cirq.Qid],
        temp: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        N = self.x_dim * self.y_dim * 2
        qlambda = 2 * N * self.t + (N * self.mu) // 2
        yield cirq.Ry(rads=2 * np.arccos(np.sqrt(self.t * N / qlambda))).on(*V)
        yield cirq.Ry(rads=2 * np.arccos(np.sqrt(1 / 5))).on(*U).controlled_by(*V)
        yield PrepareUniformSuperposition(self.x_dim).on_registers(controls=[], target=p_x)
        yield PrepareUniformSuperposition(self.y_dim).on_registers(controls=[], target=p_y)
        yield cirq.H.on_each(*temp)
        yield cirq.CNOT(*U, *V)
        yield cirq.X(*beta)
        yield from [cirq.X(*V), cirq.H(*alpha).controlled_by(*V), cirq.CX(*V, *beta), cirq.X(*V)]
        yield cirq.Circuit(cirq.CNOT.on_each([*zip([*p_x, *p_y, *alpha], [*q_x, *q_y, *beta])]))
        yield MultiTargetCSwap.make_on(control=temp[:1], target_x=q_x, target_y=q_y)
        yield AddMod(len(q_x), self.x_dim, add_val=1, cv=[0, 0]).on(*U, *V, *q_x)
        yield MultiTargetCSwap.make_on(control=temp[:1], target_x=q_x, target_y=q_y)

        and_target = context.qubit_manager.qalloc(1)
        and_anc = context.qubit_manager.qalloc(1)
        yield And(cv=(0, 0, 1)).on_registers(
            control=[*U, *V, temp[-1]], ancilla=and_anc, target=and_target
        )
        yield MultiTargetCSwap.make_on(
            control=and_target, target_x=[*p_x, *p_y, *alpha], target_y=[*q_x, *q_y, *beta]
        )
        yield And(cv=(0, 0, 1), adjoint=True).on_registers(
            control=[*U, *V, temp[-1]], ancilla=and_anc, target=and_target
        )
        context.qubit_manager.qfree([*and_anc, *and_target])


def get_walk_operator_for_hubbard_model(
    x_dim: int, y_dim: int, t: int, mu: int
) -> 'QubitizationWalkOperator':
    select = SelectHubbard(x_dim, y_dim)
    prepare = PrepareHubbard(x_dim, y_dim, t, mu)
    return QubitizationWalkOperator(select=select, prepare=prepare)
