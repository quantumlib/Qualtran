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
from typing import Any, Dict, Iterator, Set, Tuple, TYPE_CHECKING, Union

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
    QBit,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.basic_gates import CNOT, Toffoli, XGate
from qualtran.bloqs.mcmt.and_bloq import _to_tuple_or_has_length, And, is_symbolic, MultiAnd
from qualtran.symbolics import HasLength, SymbolicInt

if TYPE_CHECKING:
    import quimb.tensor as qtn

    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class MultiTargetCNOT(GateWithRegisters):
    r"""Implements single control, multi-target $C[X^{\otimes n}]$ gate.

    Implements $|0><0| I + |1><1| X^{\otimes n}$ using a circuit of depth $2\log(n) + 1$
    containing only CNOT gates.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis](https://arxiv.org/abs/1812.00954).
        Appendix B.1.
    """

    bitsize: 'SymbolicInt'

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(control=1, targets=self.bitsize)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        control: NDArray[cirq.Qid],  # type: ignore[type-var]
        targets: NDArray[cirq.Qid],  # type: ignore[type-var]
    ):
        def cnots_for_depth_i(i: int, q: NDArray[cirq.Qid]) -> Iterator[cirq.OP_TREE]:  # type: ignore[type-var]
            for c, t in zip(q[: 2**i], q[2**i : min(len(q), 2 ** (i + 1))]):
                yield cirq.CNOT(c, t)

        depth = len(targets).bit_length()
        for i in range(depth):
            yield cirq.Moment(cnots_for_depth_i(depth - i - 1, targets))
        yield cirq.CNOT(*control, targets[0])
        for i in range(depth):
            yield cirq.Moment(cnots_for_depth_i(i, targets))

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        if isinstance(self.bitsize, sympy.Expr):
            raise ValueError(f'Symbolic bitsize {self.bitsize} not supported')
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
    import_line='from qualtran.bloqs.mcmt import MultiTargetCNOT',
    examples=(_c_multi_not_symb, _c_multi_not),
)


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

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray['cirq.Qid']
    ) -> Iterator[cirq.OP_TREE]:
        controls, target = quregs.get('controls', np.array([])), quregs['target']
        if len(self.concrete_cvs) < 2:
            controls = controls.flatten()
            yield [cirq.X(q) for cv, q in zip(self.concrete_cvs, controls) if cv == 0]
            yield self.target_gate.on(*target).controlled_by(*controls)
            yield [cirq.X(q) for cv, q in zip(self.concrete_cvs, controls) if cv == 0]
            return
        qm = context.qubit_manager
        and_ancilla, and_target = np.array(qm.qalloc(len(self.concrete_cvs) - 2)), qm.qalloc(1)
        if len(self.concrete_cvs) == 2:
            and_op = And(*self.concrete_cvs).on(*controls.flatten(), *and_target)
            and_op_inv = And(*self.concrete_cvs).adjoint()(*controls.flatten(), *and_target)
        else:
            and_op = MultiAnd(self.concrete_cvs).on_registers(
                ctrl=controls, junk=and_ancilla[:, np.newaxis], target=and_target
            )
            and_op_inv = and_op**-1  # type: ignore[operator]
        yield and_op
        yield self.target_gate.on(*target).controlled_by(*and_target)
        yield and_op_inv
        qm.qfree([*and_ancilla, *and_target])

    def pretty_name(self) -> str:
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

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        from qualtran.cirq_interop._cirq_to_bloq import _cirq_gate_to_bloq

        ret = {(_cirq_gate_to_bloq(self.target_gate.controlled(1)), 1)}

        if is_symbolic(self.n_ctrls):
            return ret | {(MultiAnd(self.cvs), 1), (MultiAnd(self.cvs).adjoint(), 1)}

        n = int(self.n_ctrls)
        if n >= 2:
            and_gate = (
                And(self.concrete_cvs[0], self.concrete_cvs[1])
                if n == 2
                else MultiAnd(self.concrete_cvs)
            )
            return ret | {(and_gate, 1), (and_gate.adjoint(), 1)}
        n_pre_post_x = 2 * (len(self.concrete_cvs) - sum(self.concrete_cvs))
        pre_post_graph = {(XGate(), n_pre_post_x)} if n_pre_post_x else set({})
        return {(_cirq_gate_to_bloq(self.target_gate.controlled(n)), 1)} | pre_post_graph

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs') -> np.ndarray:
        cpauli = (
            self.target_gate.controlled(control_values=self.concrete_cvs)
            if self.n_ctrls
            else self.target_gate
        )
        return cirq.apply_unitary(cpauli, args)

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        from qualtran.cirq_interop._cirq_to_bloq import _add_my_tensors_from_gate

        _add_my_tensors_from_gate(
            self, self.signature, self.pretty_name(), tn, tag, incoming=incoming, outgoing=outgoing
        )

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


_CC_PAULI_DOC = BloqDocSpec(
    bloq_cls=MultiControlPauli,
    import_line='from qualtran.bloqs.mcmt import MultiControlPauli',
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
        [Constructing Large CNOTS](https://algassert.com/circuits/2015/06/05/Constructing-Large-Controlled-Nots.html).
        Section title "$n−2$ Ancilla Bits", Figure titled $C^n$NOT from $n-2$ zeroed bits.
    """

    cvs: Tuple[int, ...] = field(converter=lambda v: (v,) if isinstance(v, int) else tuple(v))

    @cached_property
    def signature(self) -> 'Signature':
        assert len(self.cvs) > 0
        return Signature([Register('ctrls', QBit(), shape=(len(self.cvs),)), Register('x', QBit())])

    def on_classical_vals(
        self, ctrls: 'ClassicalValT', x: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        if np.all(self.cvs == ctrls):
            x = (x + 1) % 2

        return {'ctrls': ctrls, 'x': x}

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrls: NDArray[Soquet], x: SoquetT  # type: ignore[type-var]
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

    def pretty_name(self) -> str:
        return f'C^{len(self.cvs)}-NOT'
