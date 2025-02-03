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
import warnings
from functools import cached_property
from typing import Dict, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
import sympy
from attrs import field, frozen
from numpy.typing import NDArray

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
from qualtran.bloqs.basic_gates import XGate, ZGate
from qualtran.bloqs.mcmt.and_bloq import _to_tuple_or_has_length
from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
from qualtran.drawing import Circle, TextBox, WireSymbol
from qualtran.symbolics import HasLength, is_symbolic, Shaped, slen, SymbolicInt

if TYPE_CHECKING:
    import cirq

    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


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
    target_bloq: Bloq

    def __attrs_post_init__(self):
        warnings.warn(
            "`MultiControlPauli` is deprecated. Use `bloq.controlled(...)` which now defaults"
            "to reducing controls using an `And` ladder."
            "For the same signature as `MultiControlPauli(cvs, target_bloq)`,"
            "use `target_bloq.controlled(CtrlSpec(cvs=cvs))`.",
            DeprecationWarning,
        )

    @cached_property
    def signature(self) -> 'Signature':
        ctrl = Register('controls', QBit(), shape=(self.n_ctrls,))
        target = Register('target', QBit())
        return Signature(
            [ctrl, target] if is_symbolic(self.n_ctrls) or self.n_ctrls > 0 else [target]
        )

    @property
    def n_ctrls(self) -> SymbolicInt:
        return slen(self.cvs)

    @property
    def concrete_cvs(self) -> Tuple[int, ...]:
        if isinstance(self.cvs, HasLength):
            raise ValueError(f"{self.cvs} is symbolic")
        return self.cvs

    @property
    def _multi_ctrl_bloq(self) -> ControlledViaAnd:
        cvs: Union[NDArray[np.integer], Shaped] = (
            Shaped((self.n_ctrls,)) if is_symbolic(self.n_ctrls) else np.array(self.concrete_cvs)
        )
        ctrl_spec = CtrlSpec(cvs=(cvs,))
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

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        if self.n_ctrls == 0:
            return {self.target_bloq: 1}
        return {self._multi_ctrl_bloq: 1}

    def __str__(self) -> str:
        n = self.n_ctrls
        ctrl = f'C^{n}' if is_symbolic(n) or n > 2 else ['', 'C', 'CC'][int(n)]
        return f'{ctrl}{self.target_bloq!s}'

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return TextBox(str(self))
        if reg.name == 'target':
            (target_reg,) = tuple(self.target_bloq.signature)
            return self.target_bloq.wire_symbol(target_reg)

        (i,) = idx
        cv = self.concrete_cvs[i]
        return Circle(filled=(cv == 1))

    def _circuit_diagram_info_(self, args) -> 'cirq.CircuitDiagramInfo':
        from qualtran.cirq_interop._bloq_to_cirq import _wire_symbol_to_cirq_diagram_info

        return _wire_symbol_to_cirq_diagram_info(self, args)

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs') -> np.ndarray:
        import cirq

        from qualtran.cirq_interop import BloqAsCirqGate

        target_gate = (
            self.target_bloq
            if isinstance(self.target_bloq, cirq.Gate)
            else BloqAsCirqGate(self.target_bloq)
        )
        cpauli = (
            target_gate.controlled(control_values=self.concrete_cvs)
            if self.n_ctrls
            else target_gate
        )
        return cirq.apply_unitary(cpauli, args)

    def _has_unitary_(self) -> bool:
        return not is_symbolic(self.n_ctrls)


@frozen
class MultiControlX(MultiControlPauli):
    r"""Implements multi-control, single-target X gate.

    Reduces multiple controls to a single control using an `And` ladder.
    See class `ControlledViaAnd` for details on construction.

    Alternatively, one can directly use `XGate().controlled(CtrlSpec(cvs=cvs))`

    Args:
        cvs: a tuple of `n` control bits, or a `HasLength(n)` to control by `n` 1s.

    Registers:
        controls: control register of type `QBit` and shape `(n,)`.
        target: single qubit target register.
    """

    target_bloq: Bloq = field(init=False)

    @target_bloq.default
    def _X(self):
        return XGate()

    def __attrs_post_init__(self):
        pass

    def adjoint(self) -> 'Bloq':
        return self


@bloq_example
def _ccpauli() -> MultiControlX:
    ccpauli = MultiControlX(cvs=(1, 0, 1, 0, 1))
    return ccpauli


@bloq_example
def _ccpauli_symb() -> MultiControlX:
    from qualtran.symbolics import HasLength

    ccpauli_symb = MultiControlX(cvs=HasLength(sympy.Symbol("n")))
    return ccpauli_symb


_CC_PAULI_DOC = BloqDocSpec(bloq_cls=MultiControlX, examples=(_ccpauli,))


@frozen
class MultiControlZ(MultiControlPauli):
    r"""Implements multi-control, single-target Z gate.

    Reduces multiple controls to a single control using an `And` ladder.
    See class `ControlledViaAnd` for details on construction.

    Alternatively, one can directly use `ZGate().controlled(CtrlSpec(cvs=cvs))`

    Args:
        cvs: a tuple of `n` control bits, or a `HasLength(n)` to control by `n` 1s.

    Registers:
        controls: control register of type `QBit` and shape `(n,)`.
        target: single qubit target register.
    """

    target_bloq: Bloq = field(init=False)

    @target_bloq.default
    def _Z(self):
        return ZGate()

    def __attrs_post_init__(self):
        pass

    def adjoint(self) -> 'Bloq':
        return self
