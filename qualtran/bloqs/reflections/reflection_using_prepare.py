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
from typing import Dict, Iterator, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union

import attrs
import cirq
import numpy as np
from numpy.typing import NDArray

from qualtran import Bloq, bloq_example, BloqDocSpec, CtrlSpec, QBit, Register, Side, Signature
from qualtran._infra.gate_with_registers import GateWithRegisters, merge_qubits, total_bits
from qualtran._infra.single_qubit_controlled import SpecializedSingleQubitControlledExtension
from qualtran.bloqs.basic_gates import GlobalPhase, XGate, ZPowGate
from qualtran.bloqs.mcmt import MultiControlZ
from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
from qualtran.drawing import Circle, ModPlus, TextBox
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.symbolics import HasLength, is_symbolic, pi, sarg, SymbolicFloat, SymbolicInt

if TYPE_CHECKING:
    from qualtran import BloqBuilder, SoquetT
    from qualtran.bloqs.block_encoding.lcu_block_encoding import BlackBoxPrepare
    from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@attrs.frozen(cache_hash=True)
class ReflectionUsingPrepare(GateWithRegisters, SpecializedSingleQubitControlledExtension):  # type: ignore[misc]
    r"""Applies reflection around a state prepared by `prepare_gate`

    Applies $R_{s, g=1} = g (I - 2|s\rangle\langle s|)$ using $R_{s} =
    P(I - 2|0\rangle\langle0|)P^{\dagger}$ s.t. $P|0\rangle = |s\rangle$.

    Here:
    - $|s\rangle$: The state along which we want to reflect.
    - $P$: Unitary that prepares that state $|s\rangle $ from the zero state $|0\rangle$
    - $R_{s}$: Reflection operator that adds a `-1` phase to all states in the subspace
        spanned by $|s\rangle$.
    - $g$: The global phase to control the behavior of the reflection. For example:
        We often use $g=-1$ in literature to denote the reflection operator as
        $R_{s} = -1 (I - 2|s\rangle\langle s|) = 2|s\rangle\langle s| - I$

    The composite gate corresponds to implementing the following circuit:

    ```
    |control> ------------------ Z -------------------
                                 |
    |L>       ---- PREPARE^† --- o --- PREPARE -------
    ```


    Args:
        prepare_gate: An implementation of `PREPARE` for state preparation.
        control_val: If 0/1, a controlled version of the reflection operator is constructed.
            Defaults to None, in which case the resulting reflection operator is not controlled.
        global_phase: The global phase to apply in front of the reflection operator. When building a
            controlled reflection operator, the global phase translates into a relative phase.
        eps: precision for implementation of rotation. Only relevant if
            global_phase is arbitrary angle and gate is not controlled.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
            Babbush et. al. (2018). Figure 1.
    """

    prepare_gate: Union['PrepareOracle', 'BlackBoxPrepare']
    control_val: Optional[int] = None
    global_phase: complex = 1
    eps: SymbolicFloat = 1e-11

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return () if self.control_val is None else (Register('control', QBit()),)

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return self.prepare_gate.selection_registers

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return self.prepare_gate.junk_registers

    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.control_registers, *self.selection_registers, *self.junk_registers])

    @classmethod
    def reflection_around_zero(
        cls,
        bitsizes: Sequence[SymbolicInt],
        control_val: Optional[int] = None,
        global_phase: complex = 1,
        eps: float = 1e-11,
    ) -> 'ReflectionUsingPrepare':
        """Build a reflection around zero bloq.

        Args:
            bitsizes: A list of bitsizes for the selection registers.
            control_val: If 0/1, a controlled version of the reflection operator is constructed.
                Defaults to None, in which case the resulting reflection operator is not controlled.
            global_phase: The global phase to apply in front of the reflection operator. When
                building a controlled reflection operator, the global phase translates into a
                relative phase.
            eps: precision for implementation of rotation. Only relevant if
                global_phase is arbitrary angle and gate is not controlled.
        """
        prepare_gate = PrepareIdentity.from_bitsizes(bitsizes=bitsizes)
        return ReflectionUsingPrepare(
            prepare_gate, control_val=control_val, global_phase=global_phase, eps=eps
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        # 0. Allocate new ancillas, if needed.
        phase_target = (
            bb.allocate(dtype=QBit()) if self.control_val is None else soqs.pop('control')
        )
        # prep_soqs =
        # state_prep_junk = {
        #     reg.name: soqs.pop(reg.name)
        #     for reg in self.prepare_gate.junk_registers
        #     if reg.side & Side.LEFT
        # }
        # state_prep_sel = soqs
        # state_prep_ancilla = {
        #     reg.name: np.array(bb.qalloc(reg.total_bits())).reshape(reg.shape + (reg.bitsize,))
        #     for reg in self.prepare_gate.junk_registers
        # }
        # state_prep_selection_regs = quregs
        # prepare_op = self.prepare_gate.on_registers(
        #     **state_prep_selection_regs, **state_prep_ancilla
        # )
        # 1. PREPARE†
        soqs = bb.add_d(self.prepare_gate.adjoint(), **soqs)
        # yield cirq.inverse(prepare_op)
        # 2. MultiControlled Z, controlled on |000..00> state.
        # phase_control = np.array(
        #     merge_qubits(self.selection_registers, **state_prep_selection_regs)
        # )
        if self.control_val is None:
            phase_target = bb.add(XGate(), q=phase_target)
        multi_ctrl_soqs = {
            ctrl_reg.name: soqs.pop(sel_reg.name)
            for ctrl_reg, sel_reg in zip(self._multi_ctrl_z.ctrl_regs, self.selection_registers)
        }
        multi_ctrl_soqs = bb.add_d(self._multi_ctrl_z, **multi_ctrl_soqs, target=phase_target)
        for ctrl_reg, sel_reg in zip(self._multi_ctrl_z.ctrl_regs, self.selection_registers):
            soqs[sel_reg.name] = multi_ctrl_soqs.pop(ctrl_reg.name)
        phase_target = multi_ctrl_soqs.pop('target')
        assert not multi_ctrl_soqs
        if self.global_phase != 1:
            exponent = sarg(self.global_phase) / pi(self.global_phase)
            if self.control_val is None:
                bb.add(GlobalPhase(exponent=exponent, eps=self.eps))
            else:
                phase_target = bb.add(ZPowGate(exponent=exponent, eps=self.eps), q=phase_target)
        # 3. PREPARE
        soqs = bb.add_d(self.prepare_gate, **soqs)
        # 4. Deallocate phase_target.
        if self.control_val is None:
            phase_target = bb.add(XGate(), q=phase_target)
            bb.free(phase_target)
        else:
            soqs['control'] = phase_target
        return soqs

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg.name == 'control':
            return Circle(filled=True)
        if reg.name in {reg.name for reg in self.selection_registers}:
            return f'R_L[{",".join(str(i) for i in idx)}]'
        return self.prepare_gate.wire_symbol(reg, idx)

    # def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
    #     wire_symbols = ['@' if self.control_val else '@(0)'] * total_bits(self.control_registers)
    #     wire_symbols += ['R_L'] * total_bits(self.selection_registers)
    #     return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # n_phase_control = sum(reg.total_bits() for reg in self.selection_registers)
        # cvs = HasLength(n_phase_control) if is_symbolic(n_phase_control) else [0] * n_phase_control
        costs: Set['BloqCountT'] = {
            (self.prepare_gate, 1),
            (self.prepare_gate.adjoint(), 1),
            (self._multi_ctrl_z, 1),
        }
        if self.control_val is None:
            costs.add((XGate(), 2))
        if self.global_phase != 1:
            phase_op: Bloq = GlobalPhase.from_coefficient(self.global_phase, eps=self.eps)
            if self.control_val is not None:
                phase_op = phase_op.controlled(ctrl_spec=CtrlSpec(cvs=self.control_val))
            costs.add((phase_op, 1))
        return costs

    @cached_property
    def _multi_ctrl_z(self) -> MultiControlZ:
        ctrl_soqs, dtypes, cvs = {}, [], []
        for reg in self.selection_registers:
            dtypes.append(reg.dtype)
            cvs.append(np.zeros(reg.shape, dtype=int))
        return MultiControlZ(CtrlSpec(tuple(dtypes), tuple(cvs)))


@bloq_example(generalizer=ignore_split_join)
def _refl_using_prep() -> ReflectionUsingPrepare:
    from qualtran.bloqs.state_preparation import StatePreparationAliasSampling

    data = [1] * 5
    eps = 1e-2
    prepare_gate = StatePreparationAliasSampling.from_probabilities(data, precision=eps)

    refl_using_prep = ReflectionUsingPrepare(prepare_gate)
    return refl_using_prep


@bloq_example(generalizer=ignore_split_join)
def _refl_around_zero() -> ReflectionUsingPrepare:
    refl_around_zero = ReflectionUsingPrepare.reflection_around_zero(
        bitsizes=(1, 2, 3), global_phase=-1, control_val=1
    )
    return refl_around_zero


_REFL_USING_PREP_DOC = BloqDocSpec(
    bloq_cls=ReflectionUsingPrepare,
    import_line='from qualtran.bloqs.reflections.reflection_using_prepare import ReflectionUsingPrepare',
    examples=(_refl_using_prep, _refl_around_zero),
)
