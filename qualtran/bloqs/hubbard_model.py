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

r"""Gates for Qubitizing the Hubbard Model.

This follows section V. of the [Linear T Paper](https://arxiv.org/abs/1805.03662).

The 2D Hubbard model is a special case of the electronic structure Hamiltonian
restricted to spins on a planar grid.

$$
H = -t \sum_{\langle p,q \rangle, \sigma} a_{p,\sigma}^\dagger a_{q,\sigma}
    + \frac{u}{2} \sum_{p,\alpha\ne\beta} n_{p, \alpha} n_{p, \beta}
$$

Under the Jordan-Wigner transformation, this is

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
from typing import Collection, Optional, Sequence, Tuple, Union

import attrs
import cirq
import numpy as np
from cirq._compat import cached_property
from numpy.typing import NDArray

from qualtran import Register, SelectionRegister, Signature
from qualtran._infra.gate_with_registers import total_bits
from qualtran.bloqs.and_bloq import MultiAnd
from qualtran.bloqs.apply_gate_to_lth_target import ApplyGateToLthQubit
from qualtran.bloqs.arithmetic import AddConstantMod
from qualtran.bloqs.basic_gates import CSwap
from qualtran.bloqs.prepare_uniform_superposition import PrepareUniformSuperposition
from qualtran.bloqs.qubitization_walk_operator import QubitizationWalkOperator
from qualtran.bloqs.select_and_prepare import PrepareOracle, SelectOracle
from qualtran.bloqs.selected_majorana_fermion import SelectedMajoranaFermion


@attrs.frozen
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

    Signature:
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
    def control_registers(self) -> Tuple[Register, ...]:
        return () if self.control_val is None else (Register('control', 1),)

    @cached_property
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        return (
            SelectionRegister('U', 1, 2),
            SelectionRegister('V', 1, 2),
            SelectionRegister('p_x', (self.x_dim - 1).bit_length(), self.x_dim),
            SelectionRegister('p_y', (self.y_dim - 1).bit_length(), self.y_dim),
            SelectionRegister('alpha', 1, 2),
            SelectionRegister('q_x', (self.x_dim - 1).bit_length(), self.x_dim),
            SelectionRegister('q_y', (self.y_dim - 1).bit_length(), self.y_dim),
            SelectionRegister('beta', 1, 2),
        )

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register('target', self.x_dim * self.y_dim * 2),)

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        p_x, p_y, q_x, q_y = quregs['p_x'], quregs['p_y'], quregs['q_x'], quregs['q_y']
        U, V, alpha, beta = quregs['U'], quregs['V'], quregs['alpha'], quregs['beta']
        control, target = quregs.get('control', ()), quregs['target']

        yield SelectedMajoranaFermion(
            selection_regs=(
                SelectionRegister('alpha', 1, 2),
                SelectionRegister('p_y', self.signature.get_left('p_y').total_bits(), self.y_dim),
                SelectionRegister('p_x', self.signature.get_left('p_x').total_bits(), self.x_dim),
            ),
            control_regs=self.control_registers,
            target_gate=cirq.Y,
        ).on_registers(control=control, p_x=p_x, p_y=p_y, alpha=alpha, target=target)

        yield CSwap.make_on(ctrl=V, x=p_x, y=q_x)
        yield CSwap.make_on(ctrl=V, x=p_y, y=q_y)
        yield CSwap.make_on(ctrl=V, x=alpha, y=beta)

        q_selection_regs = (
            SelectionRegister('beta', 1, 2),
            SelectionRegister('q_y', self.signature.get_left('q_y').total_bits(), self.y_dim),
            SelectionRegister('q_x', self.signature.get_left('q_x').total_bits(), self.x_dim),
        )
        yield SelectedMajoranaFermion(
            selection_regs=q_selection_regs, control_regs=self.control_registers, target_gate=cirq.X
        ).on_registers(control=control, q_x=q_x, q_y=q_y, beta=beta, target=target)

        yield CSwap.make_on(ctrl=V, x=alpha, y=beta)
        yield CSwap.make_on(ctrl=V, x=p_y, y=q_y)
        yield CSwap.make_on(ctrl=V, x=p_x, y=q_x)

        yield cirq.S(*control) ** -1 if control else cirq.global_phase_operation(
            -1j
        )  # Fix errant i from XY=iZ
        yield cirq.Z(*U).controlled_by(*control)  # Fix errant -1 from multiple pauli applications

        target_qubits_for_apply_to_lth_gate = [
            target[np.ravel_multi_index((1, qy, qx), (2, self.y_dim, self.x_dim))]
            for qx in range(self.x_dim)
            for qy in range(self.y_dim)
        ]

        yield ApplyGateToLthQubit(
            selection_regs=(
                SelectionRegister('q_y', self.signature.get_left('q_y').total_bits(), self.y_dim),
                SelectionRegister('q_x', self.signature.get_left('q_x').total_bits(), self.x_dim),
            ),
            nth_gate=lambda *_: cirq.Z,
            control_regs=Register('control', 1 + total_bits(self.control_registers)),
        ).on_registers(
            q_x=q_x, q_y=q_y, control=[*V, *control], target=target_qubits_for_apply_to_lth_gate
        )

    def controlled(
        self,
        num_controls: Optional[int] = None,
        control_values: Optional[
            Union[cirq.ops.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> 'SelectHubbard':
        if num_controls is None:
            num_controls = 1
        if control_values is None:
            control_values = [1] * num_controls
        if (
            isinstance(control_values, Sequence)
            and isinstance(control_values[0], int)
            and len(control_values) == 1
            and self.control_val is None
        ):
            return SelectHubbard(self.x_dim, self.y_dim, control_val=control_values[0])
        raise NotImplementedError(
            f'Cannot create a controlled version of {self} with control_values={control_values}.'
        )

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        info = super(SelectHubbard, self)._circuit_diagram_info_(args)
        if self.control_val is None:
            return info
        ctrl = ('@' if self.control_val else '@(0)',)
        return info.with_wire_symbols(ctrl + info.wire_symbols[0:1] + info.wire_symbols[2:])


@attrs.frozen
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

    Signature:
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
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        return (
            SelectionRegister('U', 1, 2),
            SelectionRegister('V', 1, 2),
            SelectionRegister('p_x', (self.x_dim - 1).bit_length(), self.x_dim),
            SelectionRegister('p_y', (self.y_dim - 1).bit_length(), self.y_dim),
            SelectionRegister('alpha', 1, 2),
            SelectionRegister('q_x', (self.x_dim - 1).bit_length(), self.x_dim),
            SelectionRegister('q_y', (self.y_dim - 1).bit_length(), self.y_dim),
            SelectionRegister('beta', 1, 2),
        )

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return (Register('temp', 2),)

    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.selection_registers, *self.junk_registers])

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        p_x, p_y, q_x, q_y = quregs['p_x'], quregs['p_y'], quregs['q_x'], quregs['q_y']
        U, V, alpha, beta = quregs['U'], quregs['V'], quregs['alpha'], quregs['beta']
        temp = quregs['temp']

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
        yield CSwap.make_on(ctrl=temp[:1], x=q_x, y=q_y)
        yield AddConstantMod(len(q_x), self.x_dim, add_val=1, cvs=[0, 0]).on(*U, *V, *q_x)
        yield CSwap.make_on(ctrl=temp[:1], x=q_x, y=q_y)

        and_target = context.qubit_manager.qalloc(1)
        and_anc = context.qubit_manager.qalloc(1)
        yield MultiAnd(cvs=(0, 0, 1)).on_registers(
            ctrl=np.array([U, V, temp[-1:]]), junk=np.array([and_anc]), target=and_target
        )
        yield CSwap.make_on(ctrl=and_target, x=[*p_x, *p_y, *alpha], y=[*q_x, *q_y, *beta])
        yield MultiAnd(cvs=(0, 0, 1)).adjoint().on_registers(
            ctrl=np.array([U, V, temp[-1:]]), junk=np.array([and_anc]), target=and_target
        )
        context.qubit_manager.qfree([*and_anc, *and_target])


def get_walk_operator_for_hubbard_model(
    x_dim: int, y_dim: int, t: int, mu: int
) -> 'QubitizationWalkOperator':
    select = SelectHubbard(x_dim, y_dim)
    prepare = PrepareHubbard(x_dim, y_dim, t, mu)
    return QubitizationWalkOperator(select=select, prepare=prepare)
