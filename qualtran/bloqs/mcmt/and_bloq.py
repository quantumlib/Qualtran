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

"""Bloqs for doing "AND" logical operations.

The behavior is modified by the 'control variable' attributes. A traditional value of '1'
means that a bit value of '1' is logical true for the and operation. A control value of
'0' means that a bit value of '0' is the logical true.

The `Toffoli` bloq is similar to the `And` bloq. Toffoli will flip the target bit according
to the and of its control registers. `And` will output the result into a fresh register.
"""

import itertools
from functools import cached_property
from typing import Any, Dict, Optional, Set, Tuple

import attrs
import cirq
import numpy as np
import quimb.tensor as qtn
import sympy
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    CompositeBloq,
    GateWithRegisters,
    QBit,
    Register,
    Side,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.util_bloqs import ArbitraryClifford
from qualtran.cirq_interop import decompose_from_cirq_style_method
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.drawing import Circle, directional_text_box, WireSymbol
from qualtran.resource_counting import big_O, BloqCountT, SympySymbolAllocator
from qualtran.resource_counting.generalizers import (
    cirq_to_bloqs,
    generalize_cvs,
    generalize_rotation_angle,
    ignore_alloc_free,
    ignore_cliffords,
)


@frozen
class And(GateWithRegisters):
    """A two-bit 'and' operation optimized for T-count.

    Args:
        cv1: Whether the first bit is a positive control.
        cv2: Whether the second bit is a positive control.

    Registers:
        ctrl: A two-bit control register.
        target [right]: The output bit.

    References:
     - [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
            Babbush et. al. 2018. Section III.A. and Fig. 4.
     - [Verifying Measurement Based Uncomputation](https://algassert.com/post/1903). Gidney, C. 2019.
    """

    cv1: int = 1
    cv2: int = 1
    uncompute: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('ctrl', QBit(), shape=(2,)),
                Register('target', QBit(), side=Side.RIGHT if not self.uncompute else Side.LEFT),
            ]
        )

    def adjoint(self) -> 'And':
        return attrs.evolve(self, uncompute=not self.uncompute)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if isinstance(self.cv1, sympy.Expr) or isinstance(self.cv2, sympy.Expr):
            pre_post_cliffords = big_O(1)
        else:
            pre_post_cliffords = 2 - self.cv1 - self.cv2
        if self.uncompute:
            return {(ArbitraryClifford(n=2), 4 + 2 * pre_post_cliffords)}

        return {(ArbitraryClifford(n=2), 9 + 2 * pre_post_cliffords), (TGate(), 4)}

    def pretty_name(self) -> str:
        dag = '†' if self.uncompute else ''
        return f'And{dag}'

    def on_classical_vals(
        self, *, ctrl: NDArray[np.uint8], target: Optional[int] = None
    ) -> Dict[str, NDArray[np.uint8]]:
        out = 1 if tuple(ctrl) == (self.cv1, self.cv2) else 0
        if not self.uncompute:
            return {'ctrl': ctrl, 'target': out}

        # Uncompute
        assert target == out
        return {'ctrl': ctrl}

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag: Any,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        # Fill in our tensor using "and" logic.
        data = np.zeros((2, 2, 2, 2, 2), dtype=np.complex128)
        for c1, c2 in itertools.product((0, 1), repeat=2):
            if c1 == self.cv1 and c2 == self.cv2:
                data[c1, c2, c1, c2, 1] = 1
            else:
                data[c1, c2, c1, c2, 0] = 1

        # Here: uncompute just switches the direction of the target index.
        if self.uncompute:
            trg = incoming['target']
        else:
            trg = outgoing['target']

        tn.add(
            qtn.Tensor(
                data=data,
                inds=(
                    incoming['ctrl'][0],
                    incoming['ctrl'][1],
                    outgoing['ctrl'][0],
                    outgoing['ctrl'][1],
                    trg,
                ),
                tags=['And', tag],
            )
        )

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name == 'target':
            return directional_text_box('∧', side=soq.reg.side)

        (c_idx,) = soq.idx
        filled = bool(self.cv1 if c_idx == 0 else self.cv2)
        return Circle(filled)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        """Decomposes a single `And` gate on 2 controls and 1 target in terms of Clifford+T gates.

        * And(cv).on(c1, c2, target) uses 4 T-gates and assumes target is in |0> state.
        * And(cv, adjoint=True).on(c1, c2, target) uses measurement based un-computation
            (0 T-gates) and will always leave the target in |0> state.
        """
        (c1, c2), (target,) = (quregs['ctrl'].flatten(), quregs['target'].flatten())
        pre_post_ops = [cirq.X(q) for (q, v) in zip([c1, c2], [self.cv1, self.cv2]) if v == 0]
        yield pre_post_ops
        if self.uncompute:
            yield cirq.H(target)
            yield cirq.measure(target, key=f"{target}")
            yield cirq.CZ(c1, c2).with_classical_controls(f"{target}")
            yield cirq.reset(target)
        else:
            yield [cirq.H(target), cirq.T(target)]
            yield [cirq.CNOT(c1, target), cirq.CNOT(c2, target)]
            yield [cirq.CNOT(target, c1), cirq.CNOT(target, c2)]
            yield [cirq.T(c1) ** -1, cirq.T(c2) ** -1, cirq.T(target)]
            yield [cirq.CNOT(target, c1), cirq.CNOT(target, c2)]
            yield [cirq.H(target), cirq.S(target)]
        yield pre_post_ops

    def __pow__(self, power: int) -> 'And':
        if power == 1:
            return self
        if power == -1:
            return self.adjoint()
        return NotImplemented  # pragma: no cover

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        controls = ["(0)", "@"]
        target = "And†" if self.uncompute else "And"
        wire_symbols = [controls[self.cv1], controls[self.cv2], target]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _has_unitary_(self) -> bool:
        return not self.uncompute

    def _t_complexity_(self) -> TComplexity:
        pre_post_cliffords = 2 - self.cv1 - self.cv2  # number of zeros in self.cv
        if self.uncompute:
            return TComplexity(clifford=4 + 2 * pre_post_cliffords)
        else:
            return TComplexity(t=4 * 1, clifford=9 + 2 * pre_post_cliffords)


@bloq_example(
    generalizer=[cirq_to_bloqs, ignore_cliffords, ignore_alloc_free, generalize_rotation_angle]
)
def _and_bloq() -> And:
    and_bloq = And()
    return and_bloq


_AND_DOC = BloqDocSpec(
    bloq_cls=And, import_line='from qualtran.bloqs.mcmt import And', examples=(_and_bloq,)
)


@frozen
class MultiAnd(Bloq):
    """A many-bit (multi-control) 'and' operation.

    Args:
        cvs: A tuple of control variable settings. Each entry specifies whether that
            control line is a "positive" control (`cv[i]=1`) or a "negative" control `0`.

    Registers:
        ctrl: An n-bit control register.
        junk [right]: An `n-2` bit junk register to be cleaned up by the inverse operation.
        target [right]: The output bit.
    """

    cvs: Tuple[int, ...] = field(converter=tuple)

    @cvs.validator
    def _validate_cvs(self, field, val):
        if len(val) < 3:
            raise ValueError("MultiAnd must have at least 3 control values `cvs`.")

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('ctrl', QBit(), shape=(len(self.cvs),)),
                Register('junk', QBit(), shape=(len(self.cvs) - 2,), side=Side.RIGHT),
                Register('target', QBit(), side=Side.RIGHT),
            ]
        )

    def on_classical_vals(self, ctrl: NDArray[np.uint8]) -> Dict[str, NDArray[np.uint8]]:
        accumulate_and = np.bitwise_and.accumulate(np.equal(ctrl, self.cvs).astype(np.uint8))
        junk, target = accumulate_and[1:-1], accumulate_and[-1]
        return {'ctrl': ctrl, 'junk': junk, 'target': target}

    def __pow__(self, power: int) -> "MultiAnd":
        if power == 1:
            return self
        if power == -1:
            return self.adjoint()
        return NotImplemented  # pragma: no cover

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        controls = ["(0)", "@"]
        target = "And"
        wire_symbols = [controls[c] for c in self.cvs]
        wire_symbols += ["Anc"] * (len(self.cvs) - 2)
        wire_symbols += [target]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _decompose_via_tree(
        self,
        controls: NDArray[cirq.Qid],
        control_values: Tuple[int, ...],
        ancillas: NDArray[cirq.Qid],
        target: cirq.Qid,
    ) -> cirq.ops.op_tree.OpTree:
        """Decomposes multi-controlled `And` in-terms of an `And` ladder of size #controls- 2."""

        if len(controls) == 2:
            yield And(*control_values).on(*controls, target)
            return
        new_controls = np.concatenate([ancillas[0:1], controls[2:]])
        new_control_values = (1, *control_values[2:])
        and_op = And(*control_values[:2]).on(*controls[:2], ancillas[0])

        yield and_op
        yield from self._decompose_via_tree(new_controls, new_control_values, ancillas[1:], target)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        control, ancilla, target = (
            quregs['ctrl'].flatten(),
            quregs.get('junk', np.array([])).flatten(),
            quregs['target'].flatten(),
        )
        yield self._decompose_via_tree(control, self.cvs, ancilla, *target)

    def decompose_bloq(self) -> 'CompositeBloq':
        return decompose_from_cirq_style_method(self)

    def _t_complexity_(self, adjoint: bool = False) -> TComplexity:
        pre_post_cliffords = len(self.cvs) - sum(self.cvs)  # number of zeros in self.cv
        num_single_and = len(self.cvs) - 1
        if adjoint:
            return TComplexity(clifford=4 * num_single_and + 2 * pre_post_cliffords)
        return TComplexity(
            t=4 * num_single_and, clifford=9 * num_single_and + 2 * pre_post_cliffords
        )

    def short_name(self) -> str:
        return 'And'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(And(), len(self.cvs) - 1)}


@bloq_example(generalizer=(ignore_cliffords, generalize_cvs))
def _multi_and() -> MultiAnd:
    multi_and = MultiAnd(cvs=(1, 0, 1, 0, 1, 0))
    return multi_and


_MULTI_AND_DOC = BloqDocSpec(
    bloq_cls=MultiAnd,
    import_line='from qualtran.bloqs.mcmt import MultiAnd',
    examples=(_multi_and,),
)
