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

from collections import defaultdict
from functools import cached_property
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import attrs
import cirq
import numpy as np
import sympy
from attrs import frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    DecomposeTypeError,
    GateWithRegisters,
    QAny,
    QBit,
    QInt,
    QMontgomeryUInt,
    QUInt,
    Register,
    Side,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic.addition import OutOfPlaceAdder
from qualtran.bloqs.arithmetic.bitwise import BitwiseNot
from qualtran.bloqs.arithmetic.conversions.sign_extension import SignExtend
from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.bloqs.bookkeeping import Cast
from qualtran.bloqs.mcmt import MultiControlX
from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd
from qualtran.drawing import WireSymbol
from qualtran.drawing.musical_score import Circle, Text, TextBox
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.symbolics import HasLength, is_symbolic, SymbolicInt

if TYPE_CHECKING:
    from qualtran import BloqBuilder
    from qualtran.resource_counting import (
        BloqCountDictT,
        MutableBloqCountDictT,
        SympySymbolAllocator,
    )
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class LessThanConstant(GateWithRegisters, cirq.ArithmeticGate):  # type: ignore[misc]
    """Applies U_a|x>|z> = |x> |z ^ (x < a)>"""

    bitsize: SymbolicInt
    less_than_val: SymbolicInt

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(x=QUInt(self.bitsize), target=QBit())

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text("")
        if reg.name == 'x':
            return TextBox("x")
        if reg.name == 'target':
            return TextBox("z∧(x<a)")
        raise ValueError(f'Unknown register name {reg.name}')

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [2] * int(self.bitsize), int(self.less_than_val), [2]

    def with_registers(self, *new_registers) -> "LessThanConstant":
        return LessThanConstant(len(new_registers[0]), new_registers[1])

    def apply(self, *register_vals: int) -> Union[int, Iterable[int]]:
        input_val, less_than_val, target_register_val = register_vals
        return input_val, less_than_val, target_register_val ^ (input_val < less_than_val)

    def on_classical_vals(self, *, x: int, target: int) -> Dict[str, 'ClassicalValT']:
        return {'x': x, 'target': target ^ (x < self.less_than_val)}

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["In(x)"] * int(self.bitsize)
        wire_symbols += [f'⨁(x < {self.less_than_val})']
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __pow__(self, power: int):
        if power in [1, -1]:
            return self
        return NotImplemented  # pragma: no cover

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]  # type: ignore[type-var]
    ) -> Iterator[cirq.OP_TREE]:
        """Decomposes the gate into N And & And† operations for a T complexity of 4N.

        The decomposition proceeds from the most significant qubit -bit 0- to the least significant
        qubit while maintaining whether the qubit sequence is equal to the current prefix of the
        `_val` or not.

        The bare-bone logic is:
        1. if ith bit of `_val` is 1 then:
            - qubit sequence < `_val` iff they are equal so far and the current qubit is 0.
        2. update `are_equal`: `are_equal := are_equal and (ith bit == ith qubit).`

        This logic is implemented using $n$ `And` & `And†` operations and n+1 clean ancilla where
            - one ancilla `are_equal` contains the equality informaiton
            - ancilla[i] contain whether the qubits[:i+1] != (i+1)th prefix of `_val`
        """
        qubits, (target,) = quregs['x'], quregs['target']
        # Trivial case, self._val is larger than any value the registers could represent
        if self.less_than_val >= 2**self.bitsize:
            yield cirq.X(target)
            return
        adjoint = []

        (are_equal,) = context.qubit_manager.qalloc(1)

        # Initially our belief is that the numbers are equal.
        yield cirq.X(are_equal)
        adjoint.append(cirq.X(are_equal))

        # Scan from left to right.
        # `are_equal` contains whether the numbers are equal so far.
        ancilla = context.qubit_manager.qalloc(int(self.bitsize))
        for b, q, a in zip(
            QUInt(int(self.bitsize)).to_bits(int(self.less_than_val)), qubits, ancilla
        ):
            if b:
                yield cirq.X(q)
                adjoint.append(cirq.X(q))

                # ancilla[i] = are_equal so far and (q_i != _val[i]).
                #            = equivalent to: Is the current prefix of qubits < prefix of `_val`?
                yield And().on(q, are_equal, a)
                adjoint.append(And().adjoint().on(q, are_equal, a))

                # target ^= is the current prefix of the qubit sequence < current prefix of `_val`
                yield cirq.CNOT(a, target)

                # If `a=1` (i.e. the current prefixes aren't equal) this means that
                # `are_equal` is currently = 1 and q[i] != _val[i] so we need to flip `are_equal`.
                yield cirq.CNOT(a, are_equal)
                adjoint.append(cirq.CNOT(a, are_equal))
            else:
                # ancilla[i] = are_equal so far and (q = 1).
                yield And().on(q, are_equal, a)
                adjoint.append(And().adjoint().on(q, are_equal, a))

                # if `a=1` then we need to flip `are_equal` since this means that are_equal=1,
                # b_i=0, q_i=1 => current prefixes are not equal so we need to flip `are_equal`.
                yield cirq.CNOT(a, are_equal)
                adjoint.append(cirq.CNOT(a, are_equal))
        yield from reversed(adjoint)
        context.qubit_manager.qfree(ancilla)
        context.qubit_manager.qfree([are_equal])

    def _has_unitary_(self):
        return True

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        if (
            not is_symbolic(self.less_than_val, self.bitsize)
            and self.less_than_val >= 2**self.bitsize
        ):
            return {XGate(): 1}
        num_set_bits = (
            int(self.less_than_val).bit_count()
            if not is_symbolic(self.less_than_val)
            else self.bitsize
        )
        return {
            And(): self.bitsize,
            And().adjoint(): self.bitsize,
            CNOT(): num_set_bits + 2 * self.bitsize,
            XGate(): 2 * (1 + num_set_bits),
        }


@bloq_example
def _lt_k() -> LessThanConstant:
    lt_k = LessThanConstant(bitsize=8, less_than_val=5)
    return lt_k


@bloq_example
def _lt_k_symb() -> LessThanConstant:
    n, k = sympy.symbols("n k")
    lt_k_symb = LessThanConstant(bitsize=n, less_than_val=k)
    return lt_k_symb


_LT_K_DOC = BloqDocSpec(
    bloq_cls=LessThanConstant, examples=[_lt_k, _lt_k_symb]  # TODO: support symbolic call graph
)


@frozen
class BiQubitsMixer(GateWithRegisters):
    """Implements the COMPARE2 subroutine from the reference (Fig. 1)

    This gates mixes the values in a way that preserves the result of comparison.
    The signature being compared are 2-qubit signature where

        x = 2*x_msb + x_lsb
        y = 2*y_msb + y_lsb

    The Gate mixes the 4 qubits so that sign(x - y) = sign(x_lsb' - y_lsb') where x_lsb' and y_lsb'
    are the final values of x_lsb' and y_lsb'.

    Note that the ancilla qubits are used to reduce the T-count and the user
    should clean the qubits at a later point in time with the adjoint gate.
    See: https://github.com/quantumlib/Cirq/pull/6313 and
    https://github.com/quantumlib/Qualtran/issues/389

    References:
        Supplementary Materials: Improved Techniques for Preparing Eigenstates of Fermionic Hamiltonians
        https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-018-0071-5/MediaObjects/41534_2018_71_MOESM1_ESM.pdf
    """

    is_adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        one_side = Side.RIGHT if not self.is_adjoint else Side.LEFT
        return Signature(
            [
                Register('x', QUInt(2)),
                Register('y', QUInt(2)),
                Register('ancilla', QAny(3), side=one_side),
            ]
        )

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> Iterator[cirq.OP_TREE]:
        x, y, ancilla = quregs['x'], quregs['y'], quregs['ancilla']
        x_msb, x_lsb = x
        y_msb, y_lsb = y

        def _cswap(
            control: cirq.Qid, a: cirq.Qid, b: cirq.Qid, aux: cirq.Qid
        ) -> Iterator[cirq.Operation]:
            """A CSWAP with 4T complexity and whose adjoint has 0T complexity.

                A controlled SWAP that swaps `a` and `b` based on `control`.
            It uses an extra qubit `aux` so that its adjoint would have
            a T complexity of zero.
            """
            yield cirq.CNOT(a, b)
            yield And(uncompute=self.is_adjoint).on(control, b, aux)
            yield cirq.CNOT(aux, a)
            yield cirq.CNOT(a, b)

        def _decomposition() -> Iterator[cirq.Operation]:
            # computes the difference of x - y where
            #   x = 2*x_msb + x_lsb
            #   y = 2*y_msb + y_lsb
            # And stores the result in x_lsb and y_lsb such that
            #   sign(x - y) = sign(x_lsb - y_lsb)
            # This decomposition uses 3 ancilla qubits in order to have a
            # T complexity of 8.
            yield cirq.X(ancilla[0])
            yield cirq.CNOT(y_msb, x_msb)
            yield cirq.CNOT(y_lsb, x_lsb)
            yield from _cswap(x_msb, x_lsb, ancilla[0], ancilla[1])
            yield from _cswap(x_msb, y_msb, y_lsb, ancilla[2])
            yield cirq.CNOT(y_lsb, x_lsb)

        if self.is_adjoint:
            yield from reversed(tuple(cirq.flatten_to_ops(_decomposition())))
        else:
            yield from _decomposition()

    def adjoint(self) -> 'BiQubitsMixer':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)

    def __pow__(self, power: int) -> 'BiQubitsMixer':
        if power == 1:
            return self
        if power == -1:
            return self.adjoint()
        return NotImplemented  # pragma: no cover

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {XGate(): 1, CNOT(): 9, And(uncompute=self.is_adjoint): 2}

    def _has_unitary_(self):
        return not self.is_adjoint


@bloq_example
def _bi_qubits_mixer() -> BiQubitsMixer:
    bi_qubits_mixer = BiQubitsMixer()
    return bi_qubits_mixer


_BI_QUBITS_MIXER_DOC = BloqDocSpec(bloq_cls=BiQubitsMixer, examples=[_bi_qubits_mixer])


@frozen
class SingleQubitCompare(GateWithRegisters):
    """Applies U|a>|b>|0>|0> = |a> |a=b> |(a<b)> |(a>b)>

    References:
        Supplementary Materials: Improved Techniques for Preparing Eigenstates of Fermionic Hamiltonians.
        Figure 3.
        https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-018-0071-5/MediaObjects/41534_2018_71_MOESM1_ESM.pdf
    """

    is_adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        one_side = Side.RIGHT if not self.is_adjoint else Side.LEFT
        return Signature(
            [
                Register('a', QBit()),
                Register('b', QBit()),
                Register('less_than', QBit(), side=one_side),
                Register('greater_than', QBit(), side=one_side),
            ]
        )

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> Iterator[cirq.OP_TREE]:
        a = quregs['a']
        b = quregs['b']
        less_than = quregs['less_than']
        greater_than = quregs['greater_than']

        def _decomposition() -> Iterator[cirq.Operation]:
            yield And(0, 1, uncompute=self.is_adjoint).on(*a, *b, *less_than)
            yield cirq.CNOT(*less_than, *greater_than)
            yield cirq.CNOT(*b, *greater_than)
            yield cirq.CNOT(*a, *b)
            yield cirq.CNOT(*a, *greater_than)
            yield cirq.X(*b)

        if self.is_adjoint:
            yield from reversed(tuple(_decomposition()))
        else:
            yield from _decomposition()

    def adjoint(self) -> 'SingleQubitCompare':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)

    def __pow__(self, power: int) -> Union['SingleQubitCompare', cirq.Gate]:
        if not isinstance(power, int):
            raise ValueError('SingleQubitCompare is only defined for integer powers.')
        if power % 2 == 0:
            return cirq.IdentityGate(4)
        if power == -1:
            return self.adjoint()
        return self

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {XGate(): 1, CNOT(): 4, And(uncompute=self.is_adjoint): 1}


@bloq_example
def _sq_cmp() -> SingleQubitCompare:
    sq_cmp = SingleQubitCompare()
    return sq_cmp


_SQ_CMP_DOC = BloqDocSpec(bloq_cls=SingleQubitCompare, examples=[_sq_cmp])


def _equality_with_zero(
    context: cirq.DecompositionContext, qubits: Sequence[cirq.Qid], z: cirq.Qid
) -> Iterator[cirq.OP_TREE]:
    """Helper decomposition used in `LessThanEqual`"""
    if len(qubits) == 1:
        (q,) = qubits
        yield cirq.X(q)
        yield cirq.CNOT(q, z)
        return
    if len(qubits) == 2:
        yield And(0, 0).on(*qubits, z)
    else:
        ancilla = context.qubit_manager.qalloc(len(qubits) - 2)
        yield MultiAnd(cvs=[0] * len(qubits)).on(*qubits, *ancilla, z)


@frozen
class LessThanEqual(GateWithRegisters, cirq.ArithmeticGate):  # type: ignore[misc]
    """Applies U|x>|y>|z> = |x>|y> |z ^ (x <= y)>

    Decomposes the gate in a T-complexity optimal way.

    The construction can be broken in 4 parts:
     1. In case of differing bitsizes then a multicontrol And Gate
        (Section III.A. of the first reference) is used to check whether
        the extra prefix is equal to zero and the result is stored in the `prefix_equality` qubit.
     2. The tree structure (Fig. 2) of the second reference.
        followed by a `SingleQubitCompare` to compute the result of comparison of
        the suffixes of equal length. The result is stored in `less_than` and `greater_than` and
        equality in `qubits[-2]`
     3. The results from the previous two steps are combined to update the target qubit.
     4. The adjoint of the previous operations is added to restore the input qubits
        to their original state and clean the ancilla qubits.

    When both registers are of the same size the T complexity is
    8n - 4 as in the second reference.

    When the registers differ in size and `n` is the size of the smaller one and
    `d` is the difference in size, the T complexity is the sum of the tree
    decomposition as before giving 8n + O(1); and the T complexity of an `And` gate
    over `d` registers giving 4d + O(1). This totals 8n + 4d + O(1).

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).

        Supplementary Materials: Improved Techniques for Preparing Eigenstates of Fermionic Hamiltonians.
        https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-018-0071-5/MediaObjects/41534_2018_71_MOESM1_ESM.pdf
    """

    x_bitsize: 'SymbolicInt'
    y_bitsize: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(
            x=QUInt(self.x_bitsize), y=QUInt(self.y_bitsize), target=QBit()
        )

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        if isinstance(self.x_bitsize, sympy.Expr):
            raise ValueError(f'Symbolic x bitsize {self.x_bitsize} not allowed')
        if isinstance(self.y_bitsize, sympy.Expr):
            raise ValueError(f'Symbolic y bitsize {self.y_bitsize} not allowed')
        return [2] * self.x_bitsize, [2] * self.y_bitsize, [2]

    def with_registers(self, *new_registers) -> "LessThanEqual":
        return LessThanEqual(len(new_registers[0]), len(new_registers[1]))

    def apply(self, *register_vals: int) -> Union[int, int, Iterable[int]]:
        x_val, y_val, target_val = register_vals
        return x_val, y_val, target_val ^ (x_val <= y_val)

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.name == "x":
            return TextBox('x')
        if reg.name == "y":
            return TextBox('y')
        if reg.name == "target":
            return TextBox('z∧(x<=y)')
        raise ValueError(f'Unknown register name {reg.name}')

    def on_classical_vals(self, *, x: int, y: int, target: int) -> Dict[str, 'ClassicalValT']:
        return {'x': x, 'y': y, 'target': target ^ (x <= y)}

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        if isinstance(self.x_bitsize, sympy.Expr):
            raise ValueError(f'Symbolic x bitsize {self.x_bitsize} not allowed')
        if isinstance(self.y_bitsize, sympy.Expr):
            raise ValueError(f'Symbolic y bitsize {self.y_bitsize} not allowed')
        wire_symbols = ["In(x)"] * self.x_bitsize
        wire_symbols += ["In(y)"] * self.y_bitsize
        wire_symbols += ['⨁(x <= y)']
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __pow__(self, power: int):
        if power in [1, -1]:
            return self
        return NotImplemented  # pragma: no cover

    def _decompose_via_tree(
        self, context: cirq.DecompositionContext, X: Sequence[cirq.Qid], Y: Sequence[cirq.Qid]
    ) -> Iterator[cirq.OP_TREE]:
        if len(X) == 1:
            return
        if len(X) == 2:
            yield BiQubitsMixer().on_registers(x=X, y=Y, ancilla=context.qubit_manager.qalloc(3))
            return

        m = len(X) // 2
        yield self._decompose_via_tree(context, X[:m], Y[:m])
        yield self._decompose_via_tree(context, X[m:], Y[m:])
        yield BiQubitsMixer().on_registers(
            x=(X[m - 1], X[-1]), y=(Y[m - 1], Y[-1]), ancilla=context.qubit_manager.qalloc(3)
        )

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> Iterator[cirq.OP_TREE]:
        lhs, rhs, (target,) = list(quregs['x']), list(quregs['y']), quregs['target']
        input_qubits = set(lhs + rhs + [target])

        n = min(len(lhs), len(rhs))

        prefix_equality = None
        adjoint: List[cirq.Operation] = []

        # if one of the registers is longer than the other store equality with |0--0>
        # into `prefix_equality` using d = |len(P) - len(Q)| And operations => 4d T.
        if len(lhs) != len(rhs):
            (prefix_equality,) = context.qubit_manager.qalloc(1)
            if len(lhs) > len(rhs):
                for op in cirq.flatten_to_ops(
                    _equality_with_zero(context, lhs[:-n], prefix_equality)
                ):
                    yield op
                    adjoint.append(cirq.inverse(op))
            else:
                for op in cirq.flatten_to_ops(
                    _equality_with_zero(context, rhs[:-n], prefix_equality)
                ):
                    yield op
                    adjoint.append(cirq.inverse(op))

                yield cirq.X(target), cirq.CNOT(prefix_equality, target)

        # compare the remaining suffix of P and Q
        lhs = lhs[-n:]
        rhs = rhs[-n:]
        for op in cirq.flatten_to_ops(self._decompose_via_tree(context, lhs, rhs)):
            yield op
            adjoint.append(cirq.inverse(op))

        less_than, greater_than = context.qubit_manager.qalloc(2)
        yield SingleQubitCompare().on_registers(
            a=lhs[-1], b=rhs[-1], less_than=less_than, greater_than=greater_than
        )
        adjoint.append(
            SingleQubitCompare()
            .adjoint()
            .on_registers(a=lhs[-1], b=rhs[-1], less_than=less_than, greater_than=greater_than)
        )

        if prefix_equality is None:
            yield cirq.X(target)
            yield cirq.CNOT(greater_than, target)
        else:
            (less_than_or_equal,) = context.qubit_manager.qalloc(1)
            yield And(1, 0).on(prefix_equality, greater_than, less_than_or_equal)
            adjoint.append(
                And(1, 0).adjoint().on(prefix_equality, greater_than, less_than_or_equal)
            )

            yield cirq.CNOT(less_than_or_equal, target)

        yield from reversed(adjoint)
        all_ancilla = set([q for op in adjoint for q in op.qubits if q not in input_qubits])
        context.qubit_manager.qfree(all_ancilla)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        if is_symbolic(self.x_bitsize, self.y_bitsize):
            return {
                BiQubitsMixer(): self.x_bitsize,
                BiQubitsMixer().adjoint(): self.x_bitsize,
                SingleQubitCompare(): 1,
                SingleQubitCompare().adjoint(): 1,
            }

        n = min(self.x_bitsize, self.y_bitsize)
        d = max(self.x_bitsize, self.y_bitsize) - n
        is_second_longer = self.y_bitsize > self.x_bitsize
        ret: Dict['Bloq', int] = defaultdict(lambda: 0)
        if d > 0:
            if d == 1:
                ret[CNOT()] += 2
                ret[XGate()] += 2
            elif d == 2:
                ret[And(0, 0)] += 1
                ret[And(0, 0).adjoint()] += 1
            else:
                ret[MultiAnd(cvs=[0] * d)] += 1
                ret[MultiAnd(cvs=[0] * d).adjoint()] += 1
            if is_second_longer:
                ret[CNOT()] += 1
                ret[XGate()] += 1
        ret[BiQubitsMixer()] += n - 1
        ret[BiQubitsMixer().adjoint()] += n - 1
        ret[SingleQubitCompare()] += 1
        ret[SingleQubitCompare().adjoint()] += 1
        if not d:
            ret[XGate()] += 1
            ret[CNOT()] += 1
        else:
            ret[And(1, 0)] += 1
            ret[And(1, 0).adjoint()] += 1
            ret[CNOT()] += 1

        return ret

    def _has_unitary_(self):
        return True

    def adjoint(self) -> 'Bloq':
        return self


@bloq_example
def _leq_symb() -> LessThanEqual:
    n1, n2 = sympy.symbols('n1 n2')
    leq_symb = LessThanEqual(x_bitsize=n1, y_bitsize=n2)
    return leq_symb


@bloq_example
def _leq() -> LessThanEqual:
    leq = LessThanEqual(x_bitsize=4, y_bitsize=8)
    return leq


_LEQ_DOC = BloqDocSpec(
    bloq_cls=LessThanEqual, examples=[_leq, _leq_symb]  # TODO: support symbolic call graph
)


@frozen
class GreaterThan(Bloq):
    r"""Compare two integers.

    Implements $U|a\rangle|b\rangle|0\rangle \rightarrow
    |a\rangle|b\rangle|a > b\rangle$ using $8n T$  gates.

    The bloq_counts and t_complexity are derived from equivalent qualtran gates
    assuming a clean decomposition which should yield identical costs.

    See: https://github.com/quantumlib/Qualtran/pull/381 and
    https://qualtran.readthedocs.io/en/latest/bloqs/comparison_gates.html

    Args:
        bitsize: Number of bits used to represent the two integers a and b.

    Registers:
        a: n-bit-sized input registers.
        b: n-bit-sized input registers.
        target: A single bit output register to store the result of A > B.
    """

    a_bitsize: 'SymbolicInt'
    b_bitsize: 'SymbolicInt'

    @property
    def signature(self):
        return Signature.build_from_dtypes(
            a=QUInt(self.a_bitsize), b=QUInt(self.b_bitsize), target=QBit()
        )

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> WireSymbol:
        if reg is None:
            return Text("a>b")
        if reg.name == 'a':
            return TextBox("In(a)")
        if reg.name == 'b':
            return TextBox("In(b)")
        elif reg.name == 'target':
            return TextBox("⨁(a > b)")
        raise ValueError(f'Unknown register name {reg.name}')

    def build_composite_bloq(
        self, bb: 'BloqBuilder', a: 'Soquet', b: 'Soquet', target: 'Soquet'
    ) -> Dict[str, 'SoquetT']:
        a, b, target = bb.add(
            LessThanEqual(self.a_bitsize, self.b_bitsize), x=a, y=b, target=target
        )
        target = bb.add(XGate(), q=target)
        return {'a': a, 'b': b, 'target': target}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {LessThanEqual(self.a_bitsize, self.b_bitsize): 1, XGate(): 1}


@bloq_example
def _greater_than() -> GreaterThan:
    greater_than = GreaterThan(a_bitsize=4, b_bitsize=4)
    return greater_than


_GREATER_THAN_DOC = BloqDocSpec(bloq_cls=GreaterThan, examples=[_greater_than])


@frozen
class LinearDepthGreaterThan(Bloq):
    r"""Compare two integers.

    Implements |a>|b>|t> => |a>|b>|t ⨁ (a > b)> using $4n$ T gates.

    This comparator relies on the fact that (b' + a)' = b - a. If a > b, then b - a < 0. We
    implement it by flipping all the bits in b, computing the first half of the addition circuit,
    copying out the carry, and uncomputing the addition circuit.

    Args:
        bitsize: Number of bits used to represent the two integers a and b.
        signed: A boolean condition which controls whether the a and b registers are represented
            in 2's Complement or Unsigned. This effects the decomposition of the comparison because
            it relies on the 1's complement trick described above which only works for signed
            values. If the input registers are unsigned we use 2 ancilla bits to represent the
            registers in 2's complement.

    Registers:
        a: n-bit-sized input registers.
        b: n-bit-sized input registers.
        target: A single bit output register to store the result of a > b.

    References:
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648).

        [Improved quantum circuits for elliptic curve discrete logarithms](https://arxiv.org/abs/2306.08585).
    """

    bitsize: 'SymbolicInt'
    signed: bool = False

    @property
    def signature(self):
        return Signature.build_from_dtypes(
            a=QUInt(self.bitsize), b=QUInt(self.bitsize), target=QBit()
        )

    def on_classical_vals(
        self, a: 'ClassicalValT', b: 'ClassicalValT', target: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        # target is a 1-bit register so we assert that it's classical value is binary.
        assert target == (target % 2)

        if a > b:
            target = (target + 1) % 2

        return {'a': a, 'b': b, 'target': target}

    def build_composite_bloq(
        self, bb: 'BloqBuilder', a: Soquet, b: Soquet, target: SoquetT
    ) -> Dict[str, 'SoquetT']:
        if isinstance(self.bitsize, sympy.Expr):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self}.")

        # Base Case: Comparing two qubits.
        # Signed doesn't matter because we can't represent signed integers with 1 qubit.
        if self.bitsize == 1:
            # We use a specially controlled Toffolli gate to implement GreaterThan.
            # If a is 1 and b is 0 then a > b and we can flip the target bit.
            ctrls = np.asarray([a, b])
            ctrls, target = bb.add(MultiControlX(cvs=(1, 0)), controls=ctrls, target=target)
            a, b = ctrls
            # Return the output registers.
            return {'a': a, 'b': b, 'target': target}

        # Allocate lists to store ancillas generated by the logical-and and control pairs input
        # into logical-ands.
        ancillas: List[SoquetT] = []
        and_ctrls = []

        # If the input registers are unsigned we need to append a sign bit to them in order to use
        # the 1's complement trick.
        if not self.signed:
            a_sign = bb.allocate(n=1)
            a_split = bb.split(a)
            a = bb.join(np.concatenate([[a_sign], a_split]), dtype=QUInt(self.bitsize + 1))

            b_sign = bb.allocate(n=1)
            b_split = bb.split(b)
            b = bb.join(np.concatenate([[b_sign], b_split]), dtype=QUInt(self.bitsize + 1))

        # Create variable true_bitsize to account for sign bit in bloq construction.
        true_bitsize = self.bitsize if self.signed else (self.bitsize + 1)

        # Flip all the bits in the b register.
        b_split = bb.split(b)

        for i in range(true_bitsize):
            b_split[i] = bb.add(XGate(), q=b_split[i])
        a_split = bb.split(a)

        # Iteratively implements the left adder circuit building block of the Gidney Adder. On
        # the first pair of qubits we only have to perform a logical-and operation. On all other
        # qubit pairs we perform two CNOTs, a logical-and, and a third CNOT operation.
        for i in range(true_bitsize - 1):
            if i > 0:
                carry_in = ancillas[i - 1]
                carry_in, b_split[-1 - i] = bb.add(CNOT(), ctrl=carry_in, target=b_split[-1 - i])
                carry_in, a_split[-1 - i] = bb.add(CNOT(), ctrl=carry_in, target=a_split[-1 - i])

            # Performs the logical-ands and stores all three bits' soquets in a list for later
            # uncomputing.
            and_ctrl = [b_split[-1 - i], a_split[-1 - i]]
            and_ctrl, ancilla = bb.add(And(), ctrl=and_ctrl)
            and_ctrls.append(and_ctrl)
            ancillas.append(ancilla)

            if i > 0:
                ancillas[i - 1], ancillas[i] = bb.add(CNOT(), ctrl=carry_in, target=ancillas[i])

        # Complete the addition in order to get the sign bit of (a' + b).
        ancillas[-1], a_split[0] = bb.add(CNOT(), ctrl=ancillas[-1], target=a_split[0])
        b_split[0], a_split[0] = bb.add(CNOT(), ctrl=b_split[0], target=a_split[0])

        # Use a 0-controlled NOT gate in order to flip the target bit if the sign bit of (b' + a)'
        # is 1. (b' + a)' = b - a therefore if a > b, then b - a < 0 and the sign bit of b - a will
        # be 1.
        a_split[0] = bb.add(XGate(), q=a_split[0])
        a_split[0], target = bb.add(CNOT(), ctrl=a_split[0], target=target)
        a_split[0] = bb.add(XGate(), q=a_split[0])

        # Uncompute the completion of addition on the last bit of a.
        b_split[0], a_split[0] = bb.add(CNOT(), ctrl=b_split[0], target=a_split[0])
        ancillas[-1], a_split[0] = bb.add(CNOT(), ctrl=ancillas[-1], target=a_split[0])

        # Iteratively uncomputes the left adder circuit building block by performing the operations
        # in reverse order. In a normal adder circuit we would use the right adder circuit building
        # block, but because we only need to compute the carry-out bit we uncompute the circuit to
        # restore a and b.
        for i in range(true_bitsize - 1):
            and_ctrl = and_ctrls.pop()
            ancilla = ancillas.pop()

            if i < true_bitsize - 2:
                carry_in = ancillas[-1]
                carry_in, ancilla = bb.add(CNOT(), ctrl=carry_in, target=ancilla)

            and_ctrl = bb.add(And(uncompute=True), ctrl=and_ctrl, target=ancilla)
            b_split[i + 1] = and_ctrl[0]
            a_split[i + 1] = and_ctrl[1]

            if i < true_bitsize - 2:
                carry_in, a_split[i + 1] = bb.add(CNOT(), ctrl=carry_in, target=a_split[i + 1])
                ancillas[-1], b_split[i + 1] = bb.add(CNOT(), ctrl=carry_in, target=b_split[i + 1])

        # Uncompute the bitflips done to represent x'.
        for i in range(true_bitsize):
            b_split[i] = bb.add(XGate(), q=b_split[i])

        a = bb.join(a_split, dtype=QUInt(true_bitsize))
        b = bb.join(b_split, dtype=QUInt(true_bitsize))

        # If the input registers were unsigned we free the ancilla sign bits.
        if not self.signed:
            a_split = bb.split(a)
            a_sign = a_split[0]
            a = bb.join(a_split[1:], dtype=QUInt(self.bitsize))
            bb.free(a_sign)

            b_split = bb.split(b)
            b_sign = b_split[0]
            b = bb.join(b_split[1:], dtype=QUInt(self.bitsize))
            bb.free(b_sign)

        # Return the output registers.
        return {'a': a, 'b': b, 'target': target}

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.name == "a":
            return TextBox('a')
        if reg.name == "b":
            return TextBox('b')
        if reg.name == "target":
            return TextBox('t⨁(a>b)')
        raise ValueError(f'Unknown register name {reg.name}')

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        if self.bitsize == 1:
            return {MultiControlX(cvs=(1, 0)): 1}

        if self.signed:
            return {
                CNOT(): 6 * self.bitsize - 7,
                XGate(): 2 * self.bitsize + 2,
                And(): self.bitsize - 1,
                And(uncompute=True): self.bitsize - 1,
            }

        return {
            CNOT(): 6 * self.bitsize - 1,
            XGate(): 2 * self.bitsize + 4,
            And(): self.bitsize,
            And(uncompute=True): self.bitsize,
        }


@frozen
class GreaterThanConstant(Bloq):
    r"""Implements $U_a|x\rangle = U_a|x\rangle|z\rangle = |x\rangle |z \land (x > a)\rangle$

    The bloq_counts and t_complexity are derived from equivalent qualtran gates
    assuming a clean decomposition which should yield identical costs.

    See: https://github.com/quantumlib/Qualtran/pull/381 and
    https://qualtran.readthedocs.io/en/latest/bloqs/comparison_gates.html


    Args:
        bitsize: bitsize of x register.
        val: integer to compare x against (a above.)

    Registers:
        x: Register to compare against val.
        target: Register to hold result of comparison.
    """

    bitsize: int
    val: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(x=QUInt(self.bitsize), target=QBit())

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> WireSymbol:
        if reg is None:
            return Text("")
        if reg.name == 'x':
            return TextBox("In(x)")
        elif reg.name == 'target':
            return TextBox(f"⨁(x > {self.val})")
        raise ValueError(f'Unknown register symbol {reg.name}')

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {LessThanConstant(self.bitsize, less_than_val=self.val): 1}


@bloq_example
def _gt_k() -> GreaterThanConstant:
    gt_k = GreaterThanConstant(bitsize=4, val=13)
    return gt_k


_GREATER_THAN_K_DOC = BloqDocSpec(bloq_cls=GreaterThanConstant, examples=[_gt_k])


@frozen
class EqualsAConstant(Bloq):
    r"""Implements $U_a|x\rangle|z\rangle = |x\rangle |z \oplus (x = a)\rangle$

    The bloq_counts and t_complexity are derived from:
    https://qualtran.readthedocs.io/en/latest/bloqs/comparison_gates.html#equality-as-a-special-case

    Args:
        bitsize: bitsize of x register.
        val: integer to compare x against (a above.)

    Registers:
        x: Register to compare against val.
        target: Register to hold result of comparison.
    """

    bitsize: SymbolicInt
    val: SymbolicInt

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(x=QUInt(self.bitsize), target=QBit())

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> WireSymbol:
        if reg is None:
            return Text("")
        if reg.name == 'x':
            return TextBox("In(x)")
        elif reg.name == 'target':
            return TextBox(f"⨁(x = {self.val})")
        raise ValueError(f'Unknown register symbol {reg.name}')

    def is_symbolic(self):
        return is_symbolic(self.bitsize, self.val)

    @property
    def bits_k(self) -> Union[tuple[int, ...], HasLength]:
        if is_symbolic(self.bitsize) or is_symbolic(self.val):
            return HasLength(self.bitsize)

        return tuple(QUInt(self.bitsize).to_bits(self.val))

    def build_composite_bloq(
        self, bb: 'BloqBuilder', x: 'Soquet', target: 'Soquet'
    ) -> Dict[str, 'SoquetT']:
        if is_symbolic(self.bitsize):
            raise DecomposeTypeError(f"Cannot decompose {self} with symbolic {self.bitsize=}")

        xs = bb.split(x)
        xs, target = bb.add(MultiControlX(self.bits_k), controls=xs, target=target)
        x = bb.join(xs)
        return {'x': x, 'target': target}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {MultiControlX(self.bits_k): 1}


def _make_equals_a_constant():
    from qualtran.bloqs.arithmetic import EqualsAConstant

    return EqualsAConstant(bitsize=4, val=13)


@bloq_example
def _eq_k() -> EqualsAConstant:
    eq_k = EqualsAConstant(bitsize=4, val=13)
    return eq_k


_EQUALS_K_DOC = BloqDocSpec(bloq_cls=EqualsAConstant, examples=[_eq_k])


@frozen
class CLinearDepthGreaterThan(Bloq):
    r"""Controlled greater than between two integers.

    Implements $\ket{c}\ket{a}\ket{b}\ket{t} \xrightarrow[]{} \ket{c}\ket{a}\ket{b}\ket{t ⨁ ((a > b)c)}>$
    using $n+2$ Toffoli gates.

    Note: the true cost is $n+1$ but an extra Toffoli comes from OutOfPlaceAdder which operates
    on $n+1$ qubits rather than $n$. Changing the definition of OutOfPlaceAdder will remove this
    extra Toffoli.

    This comparator relies on the fact that ~(~b + a) = b - a. If a > b, then b - a < 0. We
    implement it by flipping all the bits in b, computing the first half of the addition circuit,
    copying out the carry, and uncomputing the addition circuit.

    Args:
        dtype: type of the integer registers.
        cv: ctrl value at which the bloq is active.

    Registers:
        a: dtype input registers.
        b: dtype input registers.
        target: A single bit output register to store the result of a > b.

    References:
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648).

        [Improved quantum circuits for elliptic curve discrete logarithms](https://arxiv.org/abs/2306.08585)
            page 7.
    """

    dtype: Union[QInt, QUInt, QMontgomeryUInt]
    cv: int = 1

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(ctrl=QBit(), a=self.dtype, b=self.dtype, target=QBit())

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.name == 'ctrl':
            return Circle(filled=self.cv == 1)
        if reg.name == "a":
            return TextBox('a')
        if reg.name == "b":
            return TextBox('b')
        if reg.name == "target":
            return TextBox('t⨁((a>b)c)')
        raise ValueError(f'Unknown register name {reg.name}')

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrl: 'Soquet', a: 'Soquet', b: 'Soquet', target: 'Soquet'
    ) -> Dict[str, 'SoquetT']:

        if isinstance(self.dtype, QInt):
            a = bb.add(SignExtend(self.dtype, QInt(self.dtype.bitsize + 1)), x=a)
            b = bb.add(SignExtend(self.dtype, QInt(self.dtype.bitsize + 1)), x=b)
        else:
            a = bb.join(np.concatenate([[bb.allocate(1)], bb.split(a)]))
            b = bb.join(np.concatenate([[bb.allocate(1)], bb.split(b)]))

        dtype = attrs.evolve(self.dtype, bitsize=self.dtype.bitsize + 1)
        b = bb.add(BitwiseNot(dtype), x=b)  # b := -b-1
        a = bb.add(Cast(dtype, QUInt(dtype.bitsize)), reg=a)
        b = bb.add(Cast(dtype, QUInt(dtype.bitsize)), reg=b)
        a, b, c = bb.add(OutOfPlaceAdder(self.dtype.bitsize + 1), a=a, b=b)  # c := a - b - 1
        c = bb.add(BitwiseNot(QUInt(dtype.bitsize + 1)), x=c)  # c := b - a

        # Update `target`
        c_arr = bb.split(c)
        # The sign bit is usually the 0th bit however since we already appended an extra bit
        # to the input registers and OutOfPlaceAdder is unsigned and stores the result in
        # number bits + 1 (i.e. we are adding two extra bits), the sign bit becomes the 1st bit
        # with the 0th bit indicating whether an overflow happened or not.
        (ctrl, c_arr[1]), target = bb.add(
            MultiControlX((self.cv, 1)), controls=np.array([ctrl, c_arr[1]]), target=target
        )
        c = bb.join(c_arr)

        # Uncompute
        c = bb.add(BitwiseNot(QUInt(dtype.bitsize + 1)), x=c)
        a, b = bb.add(OutOfPlaceAdder(self.dtype.bitsize + 1).adjoint(), a=a, b=b, c=c)
        a = bb.add(Cast(dtype, QUInt(dtype.bitsize)).adjoint(), reg=a)
        b = bb.add(Cast(dtype, QUInt(dtype.bitsize)).adjoint(), reg=b)
        b = bb.add(BitwiseNot(dtype), x=b)

        if isinstance(self.dtype, QInt):
            a = bb.add(SignExtend(self.dtype, QInt(self.dtype.bitsize + 1)).adjoint(), x=a)
            b = bb.add(SignExtend(self.dtype, QInt(self.dtype.bitsize + 1)).adjoint(), x=b)
        else:
            a_arr = bb.split(a)
            a = bb.join(a_arr[1:])
            b_arr = bb.split(b)
            b = bb.join(b_arr[1:])
            bb.free(a_arr[0])
            bb.free(b_arr[0])
        return {'ctrl': ctrl, 'a': a, 'b': b, 'target': target}

    def on_classical_vals(
        self, ctrl: int, a: int, b: int, target: int
    ) -> Dict[str, 'ClassicalValT']:
        if ctrl == self.cv:
            return {'ctrl': ctrl, 'a': a, 'b': b, 'target': target ^ (a > b)}
        return {'ctrl': ctrl, 'a': a, 'b': b, 'target': target}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        signed_ops: 'MutableBloqCountDictT' = {}
        if isinstance(self.dtype, QInt):
            signed_ops = {
                SignExtend(self.dtype, QInt(self.dtype.bitsize + 1)): 2,
                SignExtend(self.dtype, QInt(self.dtype.bitsize + 1)).adjoint(): 2,
            }
        dtype = attrs.evolve(self.dtype, bitsize=self.dtype.bitsize + 1)
        return {
            BitwiseNot(dtype): 2,
            BitwiseNot(QUInt(dtype.bitsize + 1)): 2,
            OutOfPlaceAdder(self.dtype.bitsize + 1).adjoint(): 1,
            OutOfPlaceAdder(self.dtype.bitsize + 1): 1,
            MultiControlX((self.cv, 1)): 1,
            **signed_ops,
        }


@bloq_example(generalizer=ignore_split_join)
def _clineardepthgreaterthan_example() -> CLinearDepthGreaterThan:
    clineardepthgreaterthan_example = CLinearDepthGreaterThan(QInt(5))
    return clineardepthgreaterthan_example


_CLinearDepthGreaterThan_DOC = BloqDocSpec(
    bloq_cls=CLinearDepthGreaterThan, examples=[_clineardepthgreaterthan_example]
)
