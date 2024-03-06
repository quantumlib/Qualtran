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
from typing import Dict, Iterable, Iterator, List, Sequence, Set, TYPE_CHECKING, Union

import attrs
import cirq
import numpy as np
from attrs import frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    GateWithRegisters,
    QAny,
    QBit,
    QUInt,
    Register,
    Side,
    Signature,
    SoquetT,
)
from qualtran._infra.quantum_graph import Soquet
from qualtran.bloqs.basic_gates import CNOT, TGate, XGate
from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd
from qualtran.bloqs.mcmt.multi_control_multi_target_pauli import MultiControlX
from qualtran.cirq_interop.bit_tools import iter_bits
from qualtran.cirq_interop.t_complexity_protocol import t_complexity, TComplexity
from qualtran.drawing import WireSymbol
from qualtran.drawing.musical_score import TextBox

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class LessThanConstant(GateWithRegisters, cirq.ArithmeticGate):
    """Applies U_a|x>|z> = |x> |z ^ (x < a)>"""

    bitsize: int
    less_than_val: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(x=QUInt(self.bitsize), target=QBit())

    def short_name(self) -> str:
        return f'x<{self.less_than_val}'

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [2] * self.bitsize, self.less_than_val, [2]

    def with_registers(self, *new_registers) -> "LessThanConstant":
        return LessThanConstant(len(new_registers[0]), new_registers[1])

    def apply(self, *register_vals: int) -> Union[int, Iterable[int]]:
        input_val, less_than_val, target_register_val = register_vals
        return input_val, less_than_val, target_register_val ^ (input_val < less_than_val)

    def on_classical_vals(self, *, x: int, target: int) -> Dict[str, 'ClassicalValT']:
        return {'x': x, 'target': target ^ (x < self.less_than_val)}

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["In(x)"] * self.bitsize
        wire_symbols += [f'⨁(x < {self.less_than_val})']
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __pow__(self, power: int):
        if power in [1, -1]:
            return self
        return NotImplemented  # pragma: no cover

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        """Decomposes the gate into 4N And and And† operations for a T complexity of 4N.

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
        ancilla = context.qubit_manager.qalloc(self.bitsize)
        for b, q, a in zip(iter_bits(self.less_than_val, self.bitsize), qubits, ancilla):
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

    def _has_unitary_(self):
        return True

    def _t_complexity_(self) -> TComplexity:
        n = self.bitsize
        if self.less_than_val >= 2**n:
            return TComplexity(clifford=1)
        return TComplexity(t=4 * n, clifford=15 * n + 3 * bin(self.less_than_val).count("1") + 2)


@frozen
class BiQubitsMixer(GateWithRegisters):
    """Implements the COMPARE2 (Fig. 1) https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-018-0071-5/MediaObjects/41534_2018_71_MOESM1_ESM.pdf

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
    ) -> cirq.OP_TREE:
        x, y, ancilla = quregs['x'], quregs['y'], quregs['ancilla']
        x_msb, x_lsb = x
        y_msb, y_lsb = y

        def _cswap(control: cirq.Qid, a: cirq.Qid, b: cirq.Qid, aux: cirq.Qid) -> cirq.OP_TREE:
            """A CSWAP with 4T complexity and whose adjoint has 0T complexity.

                A controlled SWAP that swaps `a` and `b` based on `control`.
            It uses an extra qubit `aux` so that its adjoint would have
            a T complexity of zero.
            """
            yield cirq.CNOT(a, b)
            yield And(uncompute=self.is_adjoint).on(control, b, aux)
            yield cirq.CNOT(aux, a)
            yield cirq.CNOT(a, b)

        def _decomposition():
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

    def adjoint(self) -> 'Bloq':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)

    def __pow__(self, power: int) -> cirq.Gate:
        if power == 1:
            return self
        if power == -1:
            return self.adjoint()
        return NotImplemented  # pragma: no cover

    def _t_complexity_(self) -> TComplexity:
        if self.is_adjoint:
            return TComplexity(clifford=18)
        return TComplexity(t=8, clifford=28)

    def _has_unitary_(self):
        return not self.is_adjoint


@frozen
class SingleQubitCompare(GateWithRegisters):
    """Applies U|a>|b>|0>|0> = |a> |a=b> |(a<b)> |(a>b)>

    Source: (FIG. 3) in https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-018-0071-5/MediaObjects/41534_2018_71_MOESM1_ESM.pdf
    """  # pylint: disable=line-too-long

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
    ) -> cirq.OP_TREE:
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

    def adjoint(self) -> 'Bloq':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)

    def __pow__(self, power: int) -> cirq.Gate:
        if not isinstance(power, int):
            raise ValueError('SingleQubitCompare is only defined for integer powers.')
        if power % 2 == 0:
            return cirq.IdentityGate(4)
        if power == -1:
            return self.adjoint()
        return self

    def _t_complexity_(self) -> TComplexity:
        if self.is_adjoint:
            return TComplexity(clifford=11)
        return TComplexity(t=4, clifford=16)


def _equality_with_zero(
    context: cirq.DecompositionContext, qubits: Sequence[cirq.Qid], z: cirq.Qid
) -> cirq.OP_TREE:
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
class LessThanEqual(GateWithRegisters, cirq.ArithmeticGate):
    """Applies U|x>|y>|z> = |x>|y> |z ^ (x <= y)>"""

    x_bitsize: int
    y_bitsize: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(
            x=QUInt(self.x_bitsize), y=QUInt(self.y_bitsize), target=QBit()
        )

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [2] * self.x_bitsize, [2] * self.y_bitsize, [2]

    def with_registers(self, *new_registers) -> "LessThanEqual":
        return LessThanEqual(len(new_registers[0]), len(new_registers[1]))

    def apply(self, *register_vals: int) -> Union[int, int, Iterable[int]]:
        x_val, y_val, target_val = register_vals
        return x_val, y_val, target_val ^ (x_val <= y_val)

    def short_name(self) -> str:
        return 'x <= y'

    def on_classical_vals(self, *, x: int, y: int, target: int) -> Dict[str, 'ClassicalValT']:
        return {'x': x, 'y': y, 'target': target ^ (x <= y)}

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
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
    ) -> cirq.OP_TREE:
        """Returns comparison oracle from https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-018-0071-5/MediaObjects/41534_2018_71_MOESM1_ESM.pdf

        This decomposition follows the tree structure of (FIG. 2)
        """  # pylint: disable=line-too-long
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
    ) -> cirq.OP_TREE:
        """Decomposes the gate in a T-complexity optimal way.

        The construction can be broken in 4 parts:
            1. In case of differing bitsizes then a multicontrol And Gate
                - Section III.A. https://arxiv.org/abs/1805.03662) is used to check whether
                the extra prefix is equal to zero:
                    - result stored in: `prefix_equality` qubit.
            2. The tree structure (FIG. 2) https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-018-0071-5/MediaObjects/41534_2018_71_MOESM1_ESM.pdf
                followed by a SingleQubitCompare to compute the result of comparison of
                the suffixes of equal length:
                    - result stored in: `less_than` and `greater_than` with equality in qubits[-2]
            3. The results from the previous two steps are combined to update the target qubit.
            4. The adjoint of the previous operations is added to restore the input qubits
                to their original state and clean the ancilla qubits.
        """  # pylint: disable=line-too-long
        lhs, rhs, (target,) = quregs['x'], quregs['y'], quregs['target']

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

    def _t_complexity_(self) -> TComplexity:
        n = min(self.x_bitsize, self.y_bitsize)
        d = max(self.x_bitsize, self.y_bitsize) - n
        is_second_longer = self.y_bitsize > self.x_bitsize
        if d == 0:
            # When both registers are of the same size the T complexity is
            # 8n - 4 same as in https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-018-0071-5/MediaObjects/41534_2018_71_MOESM1_ESM.pdf.  pylint: disable=line-too-long
            return TComplexity(t=8 * n - 4, clifford=46 * n - 17)
        else:
            # When the registers differ in size and `n` is the size of the smaller one and
            # `d` is the difference in size. The T complexity is the sum of the tree
            # decomposition as before giving 8n + O(1) and the T complexity of an `And` gate
            # over `d` registers giving 4d + O(1) totaling 8n + 4d + O(1).
            # From the decomposition we get that the constant is -4 as well as the clifford counts.
            if d == 1:
                return TComplexity(t=8 * n, clifford=46 * n + 3 + 2 * is_second_longer)
            else:
                return TComplexity(
                    t=8 * n + 4 * d - 4, clifford=46 * n + 17 * d - 14 + 2 * is_second_longer
                )

    def _has_unitary_(self):
        return True


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
    a_bitsize: int
    b_bitsize: int

    @property
    def signature(self):
        return Signature.build_from_dtypes(
            a=QUInt(self.a_bitsize), b=QUInt(self.b_bitsize), target=QBit()
        )

    def short_name(self) -> str:
        return "a>b"

    def _t_complexity_(self) -> 'TComplexity':
        return t_complexity(LessThanEqual(self.a_bitsize, self.b_bitsize))

    def wire_symbol(self, soq: Soquet) -> WireSymbol:
        if soq.reg.name == 'a':
            return TextBox("In(a)")
        if soq.reg.name == 'b':
            return TextBox("In(b)")
        elif soq.reg.name == 'target':
            return TextBox("⨁(a > b)")

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # TODO Determine precise clifford count and/or ignore.
        # See: https://github.com/quantumlib/Qualtran/issues/219
        # See: https://github.com/quantumlib/Qualtran/issues/217
        t_complexity = self.t_complexity()
        return {(TGate(), t_complexity.t)}


@bloq_example
def _greater_than() -> GreaterThan:
    greater_than = GreaterThan(a_bitsize=4, b_bitsize=4)
    return greater_than


_GREATER_THAN_DOC = BloqDocSpec(
    bloq_cls=GreaterThan,
    import_line='from qualtran.bloqs.arithmetic.comparison import GreaterThan',
    examples=[_greater_than],
)


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
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648)
        [Improved quantum circuits for elliptic curve discrete logarithms](https://arxiv.org/abs/2306.08585)
    """
    bitsize: int
    signed: bool

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
        self, bb: 'BloqBuilder', a: SoquetT, b: SoquetT, target: SoquetT
    ) -> Dict[str, 'SoquetT']:

        # Base Case: Comparing two qubits.
        # Signed doesn't matter because we can't represent signed integers with 1 qubit.
        if self.bitsize == 1:
            # We use a specially controlled Toffolli gate to implement GreaterThan.
            # If a is 1 and b is 0 then a > b and we can flip the target bit.
            ctrls = [a, b]
            ctrls, target = bb.add(MultiControlX(cvs=(1, 0)), ctrls=ctrls, x=target)
            a, b = ctrls
            # Return the output registers.
            return {'a': a, 'b': b, 'target': target}

        # Allocate lists to store ancillas generated by the logical-and and control pairs input
        # into logical-ands.
        ancillas = []
        and_ctrls = []

        # If the input registers are unsigned we need to append a sign bit to them in order to use
        # the 1's complement trick.
        if not self.signed:
            a_sign = bb.allocate(n=1)
            a_split = bb.split(a)
            a = bb.join(np.concatenate([[a_sign], a_split]))

            b_sign = bb.allocate(n=1)
            b_split = bb.split(b)
            b = bb.join(np.concatenate([[b_sign], b_split]))

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

        a = bb.join(a_split)
        b = bb.join(b_split)

        # If the input registers were unsigned we free the ancilla sign bits.
        if not self.signed:
            a_split = bb.split(a)
            a_sign = a_split[0]
            a = bb.join(a_split[1:])
            bb.free(a_sign)

            b_split = bb.split(b)
            b_sign = b_split[0]
            b = bb.join(b_split[1:])
            bb.free(b_sign)

        # Return the output registers.
        return {'a': a, 'b': b, 'target': target}

    def short_name(self) -> str:
        return "a > b"


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

    def _t_complexity_(self) -> TComplexity:
        return t_complexity(LessThanConstant(self.bitsize, less_than_val=self.val))

    def short_name(self) -> str:
        return f"x > {self.val}"

    def wire_symbol(self, soq: Soquet) -> WireSymbol:
        if soq.reg.name == 'x':
            return TextBox("In(x)")
        elif soq.reg.name == 'target':
            return TextBox(f"⨁(x > {self.val})")

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # TODO Determine precise clifford count and/or ignore.
        # See: https://github.com/quantumlib/Qualtran/issues/219
        # See: https://github.com/quantumlib/Qualtran/issues/217
        t_complexity = self.t_complexity()
        return {(TGate(), t_complexity.t)}


@bloq_example
def _gt_k() -> GreaterThanConstant:
    gt_k = GreaterThanConstant(bitsize=4, val=13)
    return gt_k


_GREATER_THAN_K_DOC = BloqDocSpec(
    bloq_cls=GreaterThanConstant,
    import_line='from qualtran.bloqs.arithmetic.comparison import GreaterThanConstant',
    examples=[_gt_k],
)


@frozen
class EqualsAConstant(Bloq):
    r"""Implements $U_a|x\rangle = U_a|x\rangle|z\rangle = |x\rangle |z \land (x = a)\rangle$

    The bloq_counts and t_complexity are derived from:
    https://qualtran.readthedocs.io/en/latest/bloqs/comparison_gates.html#equality-as-a-special-case

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

    def _t_complexity_(self) -> 'TComplexity':
        return TComplexity(t=4 * (self.bitsize - 1))

    def short_name(self) -> str:
        return f"x == {self.val}"

    def wire_symbol(self, soq: Soquet) -> WireSymbol:
        if soq.reg.name == 'x':
            return TextBox("In(x)")
        elif soq.reg.name == 'target':
            return TextBox(f"⨁(x = {self.val})")

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # See: https://github.com/quantumlib/Qualtran/issues/219
        # See: https://github.com/quantumlib/Qualtran/issues/217
        return {(TGate(), 4 * (self.bitsize - 1))}


def _make_equals_a_constant():
    from qualtran.bloqs.arithmetic import EqualsAConstant

    return EqualsAConstant(bitsize=4, val=13)


@bloq_example
def _eq_k() -> EqualsAConstant:
    eq_k = EqualsAConstant(bitsize=4, val=13)
    return eq_k


_EQUALS_K_DOC = BloqDocSpec(
    bloq_cls=EqualsAConstant,
    import_line='from qualtran.bloqs.arithmetic.comparison import EqualsAConstant',
    examples=[_eq_k],
)
