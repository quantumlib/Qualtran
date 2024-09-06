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
from typing import cast, Dict, Iterable, Iterator, List, Optional, Tuple, TYPE_CHECKING, Union

import attrs
import cirq
import numpy as np
import sympy
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    CompositeBloq,
    ConnectionT,
    DecomposeTypeError,
    GateWithRegisters,
    QBit,
    Register,
    Side,
    Signature,
)
from qualtran.bloqs.basic_gates import TGate, XGate
from qualtran.bloqs.bookkeeping import ArbitraryClifford
from qualtran.cirq_interop import decompose_from_cirq_style_method
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.drawing import Circle, directional_text_box, Text, WireSymbol
from qualtran.resource_counting import (
    big_O,
    BloqCountDictT,
    MutableBloqCountDictT,
    SympySymbolAllocator,
)
from qualtran.resource_counting.generalizers import (
    cirq_to_bloqs,
    generalize_cvs,
    generalize_rotation_angle,
    ignore_alloc_free,
    ignore_cliffords,
)
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.symbolics import HasLength, is_symbolic, SymbolicInt

if TYPE_CHECKING:
    import quimb.tensor as qtn

# TODO: https://github.com/quantumlib/Qualtran/issues/1346
FLAG_AND_AS_LEAF = False


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
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
            Babbush et. al. 2018. Section III.A. and Fig. 4.

        [Verifying Measurement Based Uncomputation](https://algassert.com/post/1903). Gidney, C. 2019.
    """

    cv1: Union[int, sympy.Expr] = 1
    cv2: Union[int, sympy.Expr] = 1
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

    def decompose_bloq(self) -> 'CompositeBloq':
        if FLAG_AND_AS_LEAF:
            raise DecomposeTypeError(f"{self} is atomic.")
        return decompose_from_cirq_style_method(self)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        if FLAG_AND_AS_LEAF:
            raise DecomposeTypeError(f"{self} is atomic.")

        if isinstance(self.cv1, sympy.Expr) or isinstance(self.cv2, sympy.Expr):
            pre_post_cliffords: Union[sympy.Order, int] = big_O(1)
        else:
            pre_post_cliffords = 2 - self.cv1 - self.cv2
        if self.uncompute:
            return {ArbitraryClifford(n=2): 4 + 2 * pre_post_cliffords}

        return {ArbitraryClifford(n=2): 9 + 2 * pre_post_cliffords, TGate(): 4}

    def _t_complexity_(self) -> 'TComplexity':
        if not FLAG_AND_AS_LEAF:
            return NotImplemented

        if isinstance(self.cv1, sympy.Expr) or isinstance(self.cv2, sympy.Expr):
            pre_post_cliffords: Union[sympy.Order, int] = 0
        else:
            pre_post_cliffords = 2 - self.cv1 - self.cv2
        if self.uncompute:
            return TComplexity(clifford=4 + 2 * pre_post_cliffords)

        return TComplexity(t=4, clifford=9 + 2 * pre_post_cliffords)

    def on_classical_vals(
        self, *, ctrl: NDArray[np.uint8], target: Optional[int] = None
    ) -> Dict[str, ClassicalValT]:
        out = 1 if tuple(ctrl) == (self.cv1, self.cv2) else 0
        if not self.uncompute:
            return {'ctrl': ctrl, 'target': out}

        # Uncompute
        assert target == out
        return {'ctrl': ctrl}

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        # Fill in our tensor using "and" logic.
        data = np.zeros((2, 2, 2, 2, 2), dtype=np.complex128)
        for c1, c2 in itertools.product((0, 1), repeat=2):
            if c1 == self.cv1 and c2 == self.cv2:
                data[c1, c2, c1, c2, 1] = 1
            else:
                data[c1, c2, c1, c2, 0] = 1

        # uncompute just switches the direction of the target index.
        trg = incoming['target'] if self.uncompute else outgoing['target']

        in_ctrls = cast(NDArray, incoming['ctrl'])
        out_ctrls = cast(NDArray, outgoing['ctrl'])
        return [
            qtn.Tensor(
                data=data,
                inds=[
                    (in_ctrls[0], 0),
                    (in_ctrls[1], 0),
                    (out_ctrls[0], 0),
                    (out_ctrls[1], 0),
                    (trg, 0),
                ],
                tags=[str(self)],
            )
        ]

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.name == 'target':
            return directional_text_box('∧', side=reg.side)

        (c_idx,) = idx
        filled = bool(self.cv1 if c_idx == 0 else self.cv2)
        return Circle(filled)

    def __str__(self):
        dag = '†' if self.uncompute else ''
        return f'And{dag}'

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]  # type: ignore[type-var]
    ) -> Iterator[cirq.OP_TREE]:
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

    def to_clifford_t_circuit(self) -> 'cirq.FrozenCircuit':
        """Decomposes a single `And` gate on 2 controls and 1 target in terms of Clifford+T gates.

        * And(cv).on(c1, c2, target) uses 4 T-gates and assumes target is in |0> state.
        * And(cv, adjoint=True).on(c1, c2, target) uses measurement based un-computation
            (0 T-gates) and will always leave the target in |0> state.
        """
        c1 = cirq.NamedQubit('ctrl_0')
        c2 = cirq.NamedQubit('ctrl_1')
        target = cirq.NamedQubit('target')
        pre_post_ops = [cirq.X(q) for (q, v) in zip([c1, c2], [self.cv1, self.cv2]) if v == 0]
        circuit = cirq.Circuit(pre_post_ops)
        if self.uncompute:
            circuit += cirq.Circuit(
                [
                    cirq.H(target),
                    cirq.measure(target, key=f"{target}"),
                    cirq.CZ(c1, c2).with_classical_controls(f"{target}"),
                    cirq.reset(target),
                ]
            )
        else:
            circuit += cirq.Circuit(
                [
                    [cirq.H(target), cirq.T(target)],
                    [cirq.CNOT(c1, target), cirq.CNOT(c2, target)],
                    [cirq.CNOT(target, c1), cirq.CNOT(target, c2)],
                    [cirq.T(c1) ** -1, cirq.T(c2) ** -1, cirq.T(target)],
                    [cirq.CNOT(target, c1), cirq.CNOT(target, c2)],
                    [cirq.H(target), cirq.S(target)],
                ]
            )
        circuit += pre_post_ops
        return circuit.freeze()

    def __pow__(self, power: int) -> 'And':
        if power == 1:
            return self
        if power == -1:
            return self.adjoint()
        return NotImplemented  # pragma: no cover

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        controls = ["(0)", "@"]
        target = "And†" if self.uncompute else "And"
        if isinstance(self.cv1, sympy.Expr) or isinstance(self.cv2, sympy.Expr):
            wire_symbols = [str(self.cv1), str(self.cv2), target]
        else:
            wire_symbols = [controls[self.cv1], controls[self.cv2], target]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _has_unitary_(self) -> bool:
        return not self.uncompute


@bloq_example(
    generalizer=[cirq_to_bloqs, ignore_cliffords, ignore_alloc_free, generalize_rotation_angle]
)
def _and_bloq() -> And:
    and_bloq = And()
    return and_bloq


_AND_DOC = BloqDocSpec(bloq_cls=And, examples=(_and_bloq,))


def _to_tuple_or_has_length(
    x: Union[HasLength, Iterable[SymbolicInt]]
) -> Union[HasLength, Tuple[SymbolicInt, ...]]:
    if isinstance(x, HasLength):
        if is_symbolic(x.n):
            return x
        else:
            return (1,) * x.n
    return tuple(x)


@frozen
class MultiAnd(Bloq):
    """A many-bit (multi-control) 'and' operation.

    Args:
        cvs: A tuple of control variable settings. Each entry specifies whether that
            control line is a "positive" control (`cv[i]=1`) or a "negative" control `0`.
            If a HasLength object is passed, assumes the control values to be all 1's.


    Registers:
        ctrl: An n-bit control register.
        junk [right]: An `n-2` bit junk register to be cleaned up by the inverse operation.
        target [right]: The output bit.
    """

    cvs: Union[HasLength, Tuple[SymbolicInt, ...]] = field(converter=_to_tuple_or_has_length)

    @cvs.validator
    def _validate_cvs(self, field, val):
        if not is_symbolic(val) and len(val) < 3:
            raise ValueError("MultiAnd must have at least 3 control values `cvs`.")

    @property
    def n_ctrls(self) -> SymbolicInt:
        return self.cvs.n if isinstance(self.cvs, HasLength) else len(self.cvs)

    @property
    def concrete_cvs(self) -> Tuple[SymbolicInt, ...]:
        if isinstance(self.cvs, HasLength):
            raise ValueError(f"{self.cvs} is symbolic")
        return self.cvs

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('ctrl', QBit(), shape=(self.n_ctrls,)),
                Register('junk', QBit(), shape=(self.n_ctrls - 2,), side=Side.RIGHT),
                Register('target', QBit(), side=Side.RIGHT),
            ]
        )

    def on_classical_vals(self, ctrl: NDArray[np.uint8]) -> Dict[str, NDArray[np.uint8]]:
        accumulate_and = np.bitwise_and.accumulate(
            np.equal(ctrl, np.asarray(self.cvs)).astype(np.uint8)
        )
        junk, target = accumulate_and[1:-1], accumulate_and[-1]
        return {'ctrl': ctrl, 'junk': junk, 'target': target}

    def __pow__(self, power: int) -> "Bloq":
        if power == 1:
            return self
        if power == -1:
            return self.adjoint()
        return NotImplemented  # pragma: no cover

    def _decompose_via_tree(
        self,
        controls: NDArray[cirq.Qid],
        control_values: Tuple[SymbolicInt, ...],
        ancillas: NDArray[cirq.Qid],
        target: cirq.Qid,
    ) -> cirq.ops.op_tree.OpTree:
        """Decomposes multi-controlled `And` in-terms of an `And` ladder of size #controls- 2."""

        if len(controls) == 2:
            yield And(control_values[0], control_values[1]).on(*controls, target)
            return
        new_controls = np.concatenate([ancillas[0:1], controls[2:]])
        new_control_values = (1, *control_values[2:])
        and_op = And(control_values[0], control_values[1]).on(*controls[:2], ancillas[0])

        yield and_op
        yield from self._decompose_via_tree(new_controls, new_control_values, ancillas[1:], target)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> Iterator[cirq.OP_TREE]:
        control, ancilla, target = (
            quregs['ctrl'].flatten(),
            quregs.get('junk', np.array([])).flatten(),
            quregs['target'].flatten(),
        )
        yield self._decompose_via_tree(control, self.concrete_cvs, ancilla, *target)

    def decompose_bloq(self) -> 'CompositeBloq':
        return decompose_from_cirq_style_method(self)

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.name == 'ctrl':
            return Circle(filled=self.concrete_cvs[idx[0]] == 1)
        if reg.name == 'target':
            return directional_text_box('∧', side=reg.side)
        if len(idx) > 0:
            pretty_text = f'{reg.name}[{", ".join(str(i) for i in idx)}]'
        else:
            pretty_text = reg.name
        return directional_text_box(text=pretty_text, side=reg.side)

    def __str__(self):
        return f'MultiAnd(n={self.n_ctrls})'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        cost: 'MutableBloqCountDictT' = {And(): self.n_ctrls - 1}
        if not (
            is_symbolic(self.cvs)
            or is_symbolic(*self.concrete_cvs)
            or (self.n_ctrls == sum(self.concrete_cvs))
        ):
            cost[XGate()] = 2 * (self.n_ctrls - sum(self.concrete_cvs))

        return cost


@bloq_example(generalizer=(ignore_cliffords, generalize_cvs))
def _multi_and() -> MultiAnd:
    multi_and = MultiAnd(cvs=(1, 0, 1, 0, 1, 0))
    return multi_and


@bloq_example
def _multi_and_symb() -> MultiAnd:
    import sympy

    from qualtran.symbolics.types import HasLength

    multi_and_symb = MultiAnd(cvs=HasLength(sympy.Symbol("n")))
    return multi_and_symb


_MULTI_AND_DOC = BloqDocSpec(bloq_cls=MultiAnd, examples=(_multi_and,))
