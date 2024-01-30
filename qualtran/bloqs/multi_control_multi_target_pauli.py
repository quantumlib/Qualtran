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
from typing import Dict, Tuple

import cirq
import numpy as np
import sympy
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    GateWithRegisters,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.and_bloq import And, MultiAnd
from qualtran.bloqs.basic_gates import CNOT, Toffoli, XGate
from qualtran.cirq_interop.t_complexity_protocol import t_complexity, TComplexity


@frozen
class MultiTargetCNOT(GateWithRegisters):
    r"""Implements single control, multi-target $C[X^{\otimes n}]$ gate.

    Implements $|0><0| I + |1><1| X^{\otimes n}$ using a circuit of depth $2\log(n) + 1$
    containing only CNOT gates.

    References:
        Appendix B.1 of https://arxiv.org/abs/1812.00954.
    """

    bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(control=1, targets=self.bitsize)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        control: NDArray[cirq.Qid],
        targets: NDArray[cirq.Qid],
    ):
        def cnots_for_depth_i(i: int, q: NDArray[cirq.Qid]) -> cirq.OP_TREE:
            for c, t in zip(q[: 2**i], q[2**i : min(len(q), 2 ** (i + 1))]):
                yield cirq.CNOT(c, t)

        depth = len(targets).bit_length()
        for i in range(depth):
            yield cirq.Moment(cnots_for_depth_i(depth - i - 1, targets))
        yield cirq.CNOT(*control, targets[0])
        for i in range(depth):
            yield cirq.Moment(cnots_for_depth_i(i, targets))

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=["@"] + ["X"] * self.bitsize)


@bloq_example
def _c_multi_not_symb() -> MultiTargetCNOT:
    n = sympy.Symbol('n')
    c_multi_not_symb = MultiTargetCNOT(bitsize=n)
    return c_multi_not_symb


@bloq_example
def _c_multi_not() -> MultiTargetCNOT:
    c_multi_not = MultiTargetCNOT(bitsize=5)
    return c_multi_not


_C_MULTI_NOT_DOC = BloqDocSpec(
    bloq_cls=MultiTargetCNOT,
    import_line='from qualtran.bloqs.multi_control_multi_target_pauli import MultiTargetCNOT',
    examples=(_c_multi_not_symb, _c_multi_not),
)


@frozen
class MultiControlPauli(GateWithRegisters):
    """Implements multi-control, single-target C^{n}P gate.

    Implements $C^{n}P = (1 - |1^{n}><1^{n}|) I + |1^{n}><1^{n}| P^{n}$ using $n-1$
    clean ancillas using a multi-controlled `AND` gate.

    References:
        [Constructing Large Controlled Nots](https://algassert.com/circuits/2015/06/05/Constructing-Large-Controlled-Nots.html)
    """

    cvs: Tuple[int, ...] = field(converter=lambda v: (v,) if isinstance(v, int) else tuple(v))
    target_gate: cirq.Pauli = cirq.X

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(controls=len(self.cvs), target=1)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray['cirq.Qid']
    ) -> cirq.OP_TREE:
        controls, target = quregs['controls'], quregs['target']
        qm = context.qubit_manager
        and_ancilla, and_target = np.array(qm.qalloc(len(self.cvs) - 2)), qm.qalloc(1)
        ctrl, junk = controls[:, np.newaxis], and_ancilla[:, np.newaxis]
        if len(self.cvs) == 2:
            and_op = And(*self.cvs).on_registers(ctrl=ctrl, target=and_target)
        else:
            and_op = MultiAnd(self.cvs).on_registers(ctrl=ctrl, junk=junk, target=and_target)
        yield and_op
        yield self.target_gate.on(*target).controlled_by(*and_target)
        yield and_op**-1
        qm.qfree([*and_ancilla, *and_target])

    def short_name(self) -> str:
        return r'$C^{n}(P)$'

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@" if b else "@(0)" for b in self.cvs]
        wire_symbols += [str(self.target_gate)]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _t_complexity_(self) -> TComplexity:
        and_gate = And(*self.cvs) if len(self.cvs) == 2 else MultiAnd(self.cvs)
        and_cost = t_complexity(and_gate)
        controlled_pauli_cost = t_complexity(self.target_gate.controlled(1))
        and_inv_cost = t_complexity(and_gate**-1)
        return and_cost + controlled_pauli_cost + and_inv_cost

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs') -> np.ndarray:
        return cirq.apply_unitary(self.target_gate.controlled(control_values=self.cvs), args)

    def _has_unitary_(self) -> bool:
        return True


@bloq_example
def _ccpauli() -> MultiControlPauli:
    ccpauli = MultiControlPauli(cvs=(1, 0, 1, 0, 1), target_gate=cirq.X)
    return ccpauli


_CC_PAULI_DOC = BloqDocSpec(
    bloq_cls=MultiControlPauli,
    import_line='from qualtran.bloqs.multi_control_multi_target_pauli import MultiControlPauli',
    examples=(_ccpauli,),
)


@frozen
class MultiControlX(Bloq):
    r"""Implements multi-control, single-target X gate as a bloq using $n-2$ clean ancillas.

    Args:
        cvs: A tuple of control values. Each entry specifies whether that control line is a
            "positive" control (`cv[i]=1`) or a "negative" control (`cv[i]=0`).

    Registers:
        ctrls: An input register with n 1-bit controls corresponding to the size of the control
            variable settings above.
        x: A 1-bit input register bit-flipped based on the values in the ctrls register.

    References:
        [Constructing Large CNOTS]
        (https://algassert.com/circuits/2015/06/05/Constructing-Large-Controlled-Nots.html).
        Section title "$n−2$ Ancilla Bits", Figure titled $C^n$NOT from $n-2$ zeroed bits.
    """

    cvs: Tuple[int, ...] = field(converter=lambda v: (v,) if isinstance(v, int) else tuple(v))

    @cached_property
    def signature(self) -> 'Signature':
        assert len(self.cvs) > 0
        return Signature(
            [Register('ctrls', bitsize=1, shape=(len(self.cvs),)), Register('x', bitsize=1)]
        )

    def on_classical_vals(
        self, ctrls: 'ClassicalValT', x: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        if (self.cvs == ctrls).all():
            x = (x + 1) % 2

        return {'ctrls': ctrls, 'x': x}

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrls: SoquetT, x: SoquetT
    ) -> Dict[str, 'SoquetT']:

        # n = number of controls in the bloq.
        n = len(self.cvs)

        # Base case 1: CNOT()
        if n == 1:
            # Allows for 0-controlled implementations.
            if self.cvs[0] == 0:
                ctrls[0] = bb.add(XGate(), q=ctrls[0])
            ctrls[0], x = bb.add(CNOT(), ctrl=ctrls[0], target=x)
            if self.cvs[0] == 0:
                ctrls[0] = bb.add(XGate(), q=ctrls[0])
            return {'ctrls': ctrls, 'x': x}

        # Base case 2: Toffoli()
        if n == 2:
            # Allows for 0-controlled implementations.
            for i in range(len(self.cvs)):
                if self.cvs[i] == 0:
                    ctrls[i] = bb.add(XGate(), q=ctrls[i])

            ctrls, x = bb.add(Toffoli(), ctrl=ctrls, target=x)

            for i in range(len(self.cvs)):
                if self.cvs[i] == 0:
                    ctrls[i] = bb.add(XGate(), q=ctrls[i])

            return {'ctrls': ctrls, 'x': x}

        # Iterative case: MultiControlledX
        # Allocate necessary ancilla bits.
        ancillas = bb.allocate(n=(n - 2))

        # Split the ancilla bits for bloq decomposition connections.
        ancillas_split = bb.split(ancillas)

        # Initialize a list to store the grouped Toffoli gate controls.
        toffoli_ctrls = []

        # Allows for 0-controlled implementations.
        for i in range(len(self.cvs)):
            if self.cvs[i] == 0:
                ctrls[i] = bb.add(XGate(), q=ctrls[i])

        # Iterative case 0: The first Toffoli gate is controlled by the first two controls.
        toffoli_ctrl = [ctrls[0], ctrls[1]]
        toffoli_ctrl, ancillas_split[0] = bb.add(
            Toffoli(), ctrl=toffoli_ctrl, target=ancillas_split[0]
        )
        # Save the Toffoli controls for later uncomputation.
        toffoli_ctrls.append(toffoli_ctrl)

        # Iterative case i: The middle Toffoli gates with controls ancilla and control.
        for i in range(n - 3):
            toffoli_ctrl = [ancillas_split[i], ctrls[i + 2]]
            toffoli_ctrl, ancillas_split[i + 1] = bb.add(
                Toffoli(), ctrl=toffoli_ctrl, target=ancillas_split[i + 1]
            )
            toffoli_ctrls.append(toffoli_ctrl)

        # Iteritave case n - 1: The final Toffoli gate which is not uncomputed.
        toffoli_ctrl = [ancillas_split[n - 3], ctrls[n - 1]]
        toffoli_ctrl, x = bb.add(Toffoli(), ctrl=toffoli_ctrl, target=x)

        # Start storing end states back into ancilla and control qubits.
        ancillas_split[n - 3] = toffoli_ctrl[0]
        ctrls[n - 1] = toffoli_ctrl[1]

        # Iterative case i: Uncomputation of middle Toffoli gates.
        for i in range(n - 3):
            toffoli_ctrl = toffoli_ctrls.pop()
            toffoli_ctrl, ancillas_split[n - 3 - i] = bb.add(
                Toffoli(), ctrl=toffoli_ctrl, target=ancillas_split[n - 3 - i]
            )
            ancillas_split[n - 4 - i] = toffoli_ctrl[0]
            ctrls[n - 2 - i] = toffoli_ctrl[1]

        # Iterative case 0: Uncomputation of first Toffoli gate.
        toffoli_ctrl = toffoli_ctrls.pop()
        toffoli_ctrl, ancillas_split[0] = bb.add(
            Toffoli(), ctrl=toffoli_ctrl, target=ancillas_split[0]
        )
        ctrls[0:2] = toffoli_ctrl

        # Uncompute 0-controlled qubits.
        for i in range(len(self.cvs)):
            if self.cvs[i] == 0:
                ctrls[i] = bb.add(XGate(), q=ctrls[i])

        # Join and free ancilla qubits.
        ancillas = bb.join(ancillas_split)
        bb.free(ancillas)

        # Return the output registers.
        return {'ctrls': ctrls, 'x': x}

    def short_name(self) -> str:
        return f'C^{len(self.cvs)}-NOT'
