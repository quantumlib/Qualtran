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

from qualtran import bloq_example, BloqDocSpec, GateWithRegisters, QBit, Register, Signature
from qualtran.bloqs.basic_gates import XGate
from qualtran.bloqs.mcmt.and_bloq import MultiAnd
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
    target_gate: cirq.Pauli

    @cached_property
    def signature(self) -> 'Signature':
        ctrl = Register('controls', QBit(), shape=(len(self.cvs),))
        target = Register('target', QBit())
        return Signature([ctrl, target] if len(self.cvs) > 0 else [target])

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray['cirq.Qid']
    ) -> cirq.OP_TREE:
        controls, target = quregs.get('controls', np.array([])), quregs['target']
        if len(self.cvs) <= 2:
            controls = controls.flatten()
            yield [cirq.X(q) for cv, q in zip(self.cvs, controls) if cv == 0]
            yield self.target_gate.on(*target).controlled_by(*controls)
            yield [cirq.X(q) for cv, q in zip(self.cvs, controls) if cv == 0]
            return
        qm = context.qubit_manager
        and_ancilla, and_target = np.array(qm.qalloc(len(self.cvs) - 2)), qm.qalloc(1)
        and_op = MultiAnd(self.cvs).on_registers(
            ctrl=controls, junk=and_ancilla[:, np.newaxis], target=and_target
        )
        yield and_op
        yield self.target_gate.on(*target).controlled_by(*and_target)
        yield and_op**-1
        qm.qfree([*and_ancilla, *and_target])

    def short_name(self) -> str:
        n = len(self.cvs)
        ctrl = f'C^{n}' if n > 2 else ['', 'C', 'CC'][n]
        return f'{ctrl}{self.target_gate!s}'

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@" if b else "@(0)" for b in self.cvs]
        wire_symbols += [str(self.target_gate)]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def on_classical_vals(self, **vals: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        controls, target = vals.get('controls', np.array([])), vals.get('target')
        if self.target_gate not in (cirq.X, XGate()):
            raise NotImplementedError(f"{self} is not classically simulatable.")

        if (self.cvs == controls).all():
            target = (target + 1) % 2

        return {'controls': controls, 'target': target}

    def _t_complexity_(self) -> TComplexity:
        n = len(self.cvs)
        if n <= 2:
            pre_post_clifford = TComplexity(clifford=2 * (len(self.cvs) - sum(self.cvs)))
            target_cost = t_complexity(self.target_gate.controlled(n))
            return pre_post_clifford + target_cost
        and_cost = t_complexity(MultiAnd(self.cvs))
        controlled_pauli_cost = t_complexity(self.target_gate.controlled(1))
        and_inv_cost = t_complexity(MultiAnd(self.cvs).adjoint())
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
