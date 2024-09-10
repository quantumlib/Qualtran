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

from functools import cached_property
from typing import Iterator, Optional, Tuple

import attrs
import cirq
import numpy as np
from numpy.typing import NDArray

from qualtran import bloq_example, BloqDocSpec, BQUInt, QAny, QBit, Register, Signature
from qualtran._infra.gate_with_registers import total_bits
from qualtran._infra.single_qubit_controlled import SpecializedSingleQubitControlledExtension
from qualtran.bloqs.basic_gates import CSwap
from qualtran.bloqs.multiplexers.apply_gate_to_lth_target import ApplyGateToLthQubit
from qualtran.bloqs.multiplexers.select_base import SelectOracle
from qualtran.bloqs.multiplexers.selected_majorana_fermion import SelectedMajoranaFermion


@attrs.frozen
class SelectHubbard(SelectOracle, SpecializedSingleQubitControlledExtension):  # type: ignore[misc]
    r"""The SELECT operation optimized for the 2D Hubbard model.

    In contrast to SELECT for an arbitrary chemistry Hamiltonian, we:
     - explicitly consider the two dimensions of indices to permit optimization of the circuits.
     - dispense with the `theta` index for phases.

    If neither $U$ nor $V$ is set we apply the kinetic terms of the Hamiltonian:

    $$
    -\hop{X} \quad p < q \\
    -\hop{Y} \quad p > q
    $$

    If $U$ is set we know $(p,\alpha)=(q,\beta)$ and apply the single-body term: $-Z_{p,\alpha}$.
    If $V$ is set we know $p=q, \alpha=0$, and $\beta=1$ and apply the spin term:
    $Z_{p,\alpha}Z_{p,\beta}$

    `SelectHubbard`'s construction uses $10 * N + log(N)$ T-gates.

    Args:
        x_dim: the number of sites along the x axis.
        y_dim: the number of sites along the y axis.
        control_val: Optional bit specifying the control value for constructing a controlled
            version of this gate. Defaults to None, which means un-controlled.

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
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Section V. and Fig. 19.
    """

    x_dim: int
    y_dim: int
    control_val: Optional[int] = None

    def __attrs_post_init__(self):
        if self.x_dim != self.y_dim:
            raise NotImplementedError("Currently only supports the case where x_dim=y_dim.")

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return () if self.control_val is None else (Register('control', QBit()),)

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (
            Register('U', BQUInt(1, 2)),
            Register('V', BQUInt(1, 2)),
            Register('p_x', BQUInt((self.x_dim - 1).bit_length(), self.x_dim)),
            Register('p_y', BQUInt((self.y_dim - 1).bit_length(), self.y_dim)),
            Register('alpha', BQUInt(1, 2)),
            Register('q_x', BQUInt((self.x_dim - 1).bit_length(), self.x_dim)),
            Register('q_y', BQUInt((self.y_dim - 1).bit_length(), self.y_dim)),
            Register('beta', BQUInt(1, 2)),
        )

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register('target', QAny(self.x_dim * self.y_dim * 2)),)

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
    ) -> Iterator[cirq.OP_TREE]:
        p_x, p_y, q_x, q_y = quregs['p_x'], quregs['p_y'], quregs['q_x'], quregs['q_y']
        U, V, alpha, beta = quregs['U'], quregs['V'], quregs['alpha'], quregs['beta']
        control, target = quregs.get('control', ()), quregs['target']

        yield SelectedMajoranaFermion(
            selection_regs=(
                Register('alpha', BQUInt(1, 2)),
                Register('p_y', BQUInt(self.signature.get_left('p_y').total_bits(), self.y_dim)),
                Register('p_x', BQUInt(self.signature.get_left('p_x').total_bits(), self.x_dim)),
            ),
            control_regs=self.control_registers,
            target_gate=cirq.Y,
        ).on_registers(control=control, p_x=p_x, p_y=p_y, alpha=alpha, target=target)

        yield CSwap.make_on(ctrl=V, x=p_x, y=q_x)
        yield CSwap.make_on(ctrl=V, x=p_y, y=q_y)
        yield CSwap.make_on(ctrl=V, x=alpha, y=beta)

        q_selection_regs = (
            Register('beta', BQUInt(1, 2)),
            Register('q_y', BQUInt(self.signature.get_left('q_y').total_bits(), self.y_dim)),
            Register('q_x', BQUInt(self.signature.get_left('q_x').total_bits(), self.x_dim)),
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
                Register('q_y', BQUInt(self.signature.get_left('q_y').total_bits(), self.y_dim)),
                Register('q_x', BQUInt(self.signature.get_left('q_x').total_bits(), self.x_dim)),
            ),
            nth_gate=lambda *_: cirq.Z,
            control_regs=Register('control', QAny(1 + total_bits(self.control_registers))),
        ).on_registers(
            q_x=q_x, q_y=q_y, control=[*V, *control], target=target_qubits_for_apply_to_lth_gate
        )

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        info = super(SelectHubbard, self)._circuit_diagram_info_(args)
        if self.control_val is None:
            return info
        ctrl = ('@' if self.control_val else '@(0)',)
        return info.with_wire_symbols(ctrl + info.wire_symbols[0:1] + info.wire_symbols[2:])

    def __str__(self):
        s = f'SelectHubbard({self.x_dim}, {self.y_dim})'
        if self.control_val is not None:
            return f'C{s}'
        return s


@bloq_example
def _sel_hubb() -> SelectHubbard:
    x_dim = 4
    y_dim = 4
    sel_hubb = SelectHubbard(x_dim, y_dim)
    return sel_hubb


_SELECT_HUBBARD: BloqDocSpec = BloqDocSpec(bloq_cls=SelectHubbard, examples=(_sel_hubb,))
