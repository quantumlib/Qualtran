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
from typing import Dict, Tuple, TYPE_CHECKING, Union

import cirq
import numpy as np
import sympy
from attrs import field, frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    CtrlSpec,
    DecomposeTypeError,
    GateWithRegisters,
    QBit,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.basic_gates import XGate
from qualtran.bloqs.mcmt.and_bloq import _to_tuple_or_has_length, is_symbolic
from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
from qualtran.symbolics import HasLength, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class MultiControlPauli(GateWithRegisters):
    r"""Implements multi-control, single-target C^{n}P gate.

    Implements $C^{n}P = (1 - |1^{n}><1^{n}|) I + |1^{n}><1^{n}| P^{n}$ using $n-1$
    clean ancillas using a multi-controlled `AND` gate. Uses the Toffoli ladder
    construction described in "n−2 Ancilla Bits" section of Ref[1] but uses an
    $\text{AND} / \text{AND}^\dagger$ ladder instead for computing / uncomputing
    using clean ancillas instead of the Toffoli ladder. The measurement based
    uncomputation of $\text{AND}$ does not consume any magic states and thus has
    better constant factors.

    References:
        [Constructing Large Controlled Nots](https://algassert.com/circuits/2015/06/05/Constructing-Large-Controlled-Nots.html)
    """

    cvs: Union[HasLength, Tuple[int, ...]] = field(converter=_to_tuple_or_has_length)
    target_gate: cirq.Pauli

    @cached_property
    def signature(self) -> 'Signature':
        ctrl = Register('controls', QBit(), shape=(self.n_ctrls,))
        target = Register('target', QBit())
        return Signature(
            [ctrl, target] if is_symbolic(self.n_ctrls) or self.n_ctrls > 0 else [target]
        )

    @property
    def n_ctrls(self) -> SymbolicInt:
        return self.cvs.n if isinstance(self.cvs, HasLength) else len(self.cvs)

    @property
    def concrete_cvs(self) -> Tuple[int, ...]:
        if isinstance(self.cvs, HasLength):
            raise ValueError(f"{self.cvs} is symbolic")
        return self.cvs

    @property
    def target_bloq(self) -> Bloq:
        from qualtran.cirq_interop import cirq_gate_to_bloq

        return cirq_gate_to_bloq(self.target_gate)

    @property
    def _multi_ctrl_bloq(self) -> ControlledViaAnd:
        ctrl_spec = CtrlSpec(cvs=(np.array(self.concrete_cvs),))
        return ControlledViaAnd(self.target_bloq, ctrl_spec)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        if is_symbolic(self.cvs):
            raise DecomposeTypeError(f"cannot decompose {self} with symbolic {self.cvs=}")

        target = soqs.pop('target')
        (target_reg_name,) = [reg.name for reg in self.target_bloq.signature]
        if self.n_ctrls == 0:
            # TODO discuss if we should remove support for this case.
            target = bb.add(self.target_bloq, **{target_reg_name: target})
            return {'target': target}

        (ctrl_reg_name,) = self._multi_ctrl_bloq.ctrl_reg_names

        ctrl = soqs.pop('controls')
        out_soqs = bb.add_d(self._multi_ctrl_bloq, **{ctrl_reg_name: ctrl, target_reg_name: target})

        return {'controls': out_soqs[ctrl_reg_name], 'target': out_soqs[target_reg_name]}

    def __str__(self) -> str:
        n = self.n_ctrls
        ctrl = f'C^{n}' if is_symbolic(n) or n > 2 else ['', 'C', 'CC'][int(n)]
        return f'{ctrl}{self.target_gate!s}'

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@" if b else "@(0)" for b in self.concrete_cvs]
        wire_symbols += [str(self.target_gate)]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def on_classical_vals(self, **vals: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        controls, target = vals.get('controls', np.array([])), vals.get('target', 0)
        if self.target_gate not in (cirq.X, XGate()):
            raise NotImplementedError(f"{self} is not classically simulatable.")

        if np.all(self.concrete_cvs == controls):
            target = (target + 1) % 2

        return {'controls': controls, 'target': target}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        if self.n_ctrls == 0:
            return {self.target_bloq: 1}

        if is_symbolic(self.cvs):
            # TODO CtrlSpec does not support symbolic cvs yet.
            #      remove this case once support is added.
            #      https://github.com/quantumlib/Qualtran/issues/1168
            from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd

            if self.n_ctrls == 1:
                return {self.target_bloq.controlled(): 1}
            elif self.n_ctrls == 2:
                and_bloq = And(ssa.new_symbol('cv1'), ssa.new_symbol('cv2'))
                return {self.target_bloq.controlled(): 1, and_bloq: 1, and_bloq.adjoint(): 1}
            else:
                m_and_bloq = MultiAnd(self.cvs)
                return {self.target_bloq.controlled(): 1, m_and_bloq: 1, m_and_bloq.adjoint(): 1}

        return {self._multi_ctrl_bloq: 1}

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs') -> np.ndarray:
        cpauli = (
            self.target_gate.controlled(control_values=self.concrete_cvs)
            if self.n_ctrls
            else self.target_gate
        )
        return cirq.apply_unitary(cpauli, args)

    def _has_unitary_(self) -> bool:
        return not is_symbolic(self.n_ctrls)


@bloq_example
def _ccpauli() -> MultiControlPauli:
    ccpauli = MultiControlPauli(cvs=(1, 0, 1, 0, 1), target_gate=cirq.X)
    return ccpauli


@bloq_example
def _ccpauli_symb() -> MultiControlPauli:
    from qualtran.symbolics import HasLength

    ccpauli_symb = MultiControlPauli(cvs=HasLength(sympy.Symbol("n")), target_gate=cirq.X)
    return ccpauli_symb


_CC_PAULI_DOC = BloqDocSpec(bloq_cls=MultiControlPauli, examples=(_ccpauli,))


@frozen
class MultiControlX(MultiControlPauli):
    r"""Implements multi-control, single-target X gate.

    See :class:`MultiControlPauli` for implementation and costs.
    """
    target_gate: cirq.Pauli = field(init=False)

    @target_gate.default
    def _X(self):
        return cirq.X


@frozen
class MultiControlZ(MultiControlPauli):
    r"""Implements multi-control, single-target Z gate.

    See :class:`MultiControlPauli` for implementation and costs.
    """
    target_gate: cirq.Pauli = field(init=False)

    @target_gate.default
    def _Z(self):
        return cirq.Z
