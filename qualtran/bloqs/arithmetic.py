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
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple, TYPE_CHECKING, Union

import cirq
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import Bloq, GateWithRegisters, Register, Side, Signature
from qualtran.bloqs.and_bloq import And, MultiAnd
from qualtran.bloqs.basic_gates import TGate, Toffoli
from qualtran.bloqs.util_bloqs import ArbitraryClifford
from qualtran.cirq_interop.bit_tools import iter_bits
from qualtran.cirq_interop.t_complexity_protocol import t_complexity, TComplexity

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
        return Signature.build(x=self.bitsize, target=1)

    def short_name(self) -> str:
        return f'x<{self.less_than_val}'

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [2] * self.bitsize, self.less_than_val, [2]

    def with_registers(self, *new_registers) -> "LessThanConstant":
        return LessThanConstant(len(new_registers[0]), new_registers[1])

    def apply(self, *register_vals: int) -> Union[int, Iterable[int]]:
        input_val, less_than_val, target_register_val = register_vals
        return input_val, less_than_val, target_register_val ^ (input_val < less_than_val)

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["In(x)"] * self.bitsize
        wire_symbols += [f'+(x < {self.less_than_val})']
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
                adjoint.append(And(adjoint=True).on(q, are_equal, a))

                # target ^= is the current prefix of the qubit sequence < current prefix of `_val`
                yield cirq.CNOT(a, target)

                # If `a=1` (i.e. the current prefixes aren't equal) this means that
                # `are_equal` is currently = 1 and q[i] != _val[i] so we need to flip `are_equal`.
                yield cirq.CNOT(a, are_equal)
                adjoint.append(cirq.CNOT(a, are_equal))
            else:
                # ancilla[i] = are_equal so far and (q = 1).
                yield And().on(q, are_equal, a)
                adjoint.append(And(adjoint=True).on(q, are_equal, a))

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
    """  # pylint: disable=line-too-long

    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        one_side = Side.RIGHT if not self.adjoint else Side.LEFT
        return Signature(
            [Register('x', 2), Register('y', 2), Register('ancilla', 3, side=one_side)]
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
            yield And(adjoint=self.adjoint).on(control, b, aux)
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

        if self.adjoint:
            yield from reversed(tuple(cirq.flatten_to_ops(_decomposition())))
        else:
            yield from _decomposition()

    def __pow__(self, power: int) -> cirq.Gate:
        if power == 1:
            return self
        if power == -1:
            return BiQubitsMixer(adjoint=not self.adjoint)
        return NotImplemented  # pragma: no cover

    def _t_complexity_(self) -> TComplexity:
        if self.adjoint:
            return TComplexity(clifford=18)
        return TComplexity(t=8, clifford=28)

    def _has_unitary_(self):
        return not self.adjoint


@frozen
class SingleQubitCompare(GateWithRegisters):
    """Applies U|a>|b>|0>|0> = |a> |a=b> |(a<b)> |(a>b)>

    Source: (FIG. 3) in https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-018-0071-5/MediaObjects/41534_2018_71_MOESM1_ESM.pdf
    """  # pylint: disable=line-too-long

    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        one_side = Side.RIGHT if not self.adjoint else Side.LEFT
        return Signature(
            [
                Register('a', 1),
                Register('b', 1),
                Register('less_than', 1, side=one_side),
                Register('greater_than', 1, side=one_side),
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
            yield And(0, 1, adjoint=self.adjoint).on(*a, *b, *less_than)
            yield cirq.CNOT(*less_than, *greater_than)
            yield cirq.CNOT(*b, *greater_than)
            yield cirq.CNOT(*a, *b)
            yield cirq.CNOT(*a, *greater_than)
            yield cirq.X(*b)

        if self.adjoint:
            yield from reversed(tuple(_decomposition()))
        else:
            yield from _decomposition()

    def __pow__(self, power: int) -> cirq.Gate:
        if not isinstance(power, int):
            raise ValueError('SingleQubitCompare is only defined for integer powers.')
        if power % 2 == 0:
            return cirq.IdentityGate(4)
        if power < 0:
            return SingleQubitCompare(adjoint=not self.adjoint)
        return self

    def _t_complexity_(self) -> TComplexity:
        if self.adjoint:
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
        return Signature.build(x=self.x_bitsize, y=self.y_bitsize, target=1)

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [2] * self.x_bitsize, [2] * self.y_bitsize, [2]

    def with_registers(self, *new_registers) -> "LessThanEqual":
        return LessThanEqual(len(new_registers[0]), len(new_registers[1]))

    def apply(self, *register_vals: int) -> Union[int, int, Iterable[int]]:
        x_val, y_val, target_val = register_vals
        return x_val, y_val, target_val ^ (x_val <= y_val)

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["In(x)"] * self.x_bitsize
        wire_symbols += ["In(y)"] * self.y_bitsize
        wire_symbols += ['+(x <= y)']
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
            SingleQubitCompare(adjoint=True).on_registers(
                a=lhs[-1], b=rhs[-1], less_than=less_than, greater_than=greater_than
            )
        )

        if prefix_equality is None:
            yield cirq.X(target)
            yield cirq.CNOT(greater_than, target)
        else:
            (less_than_or_equal,) = context.qubit_manager.qalloc(1)
            yield And(1, 0).on(prefix_equality, greater_than, less_than_or_equal)
            adjoint.append(
                And(1, 0, adjoint=True).on(prefix_equality, greater_than, less_than_or_equal)
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
class Add(GateWithRegisters, cirq.ArithmeticGate):
    r"""An n-bit addition gate.

    Implements $U|a\rangle|b\rangle \rightarrow |a\rangle|a+b\rangle$ using $4n - 4 T$ gates.

    Args:
        bitsize: Number of bits used to represent each integer. Must be large
            enough to hold the result in the output register of a + b, or else it simply
            drops the most significant bits.

    Registers:
        a: A bitsize-sized input register (register a above).
        b: A bitsize-sized input/output register (register b above).

    References:
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648)
    """

    bitsize: int

    @property
    def signature(self):
        return Signature.build(a=self.bitsize, b=self.bitsize)

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [2] * self.bitsize, [2] * self.bitsize

    def with_registers(self, *new_registers) -> 'Add':
        return Add(len(new_registers[0]))

    def apply(self, *register_values: int) -> Union[int, Iterable[int]]:
        p, q = register_values
        return p, p + q

    def short_name(self) -> str:
        return "a+b"

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["In(x)"] * self.bitsize
        wire_symbols += ["In(y)/Out(x+y)"] * self.bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _has_unitary_(self):
        return True

    def _left_building_block(self, inp, out, anc, depth):
        if depth == self.bitsize - 1:
            return
        else:
            yield cirq.CX(anc[depth - 1], inp[depth])
            yield cirq.CX(anc[depth - 1], out[depth])
            yield And().on(inp[depth], out[depth], anc[depth])
            yield cirq.CX(anc[depth - 1], anc[depth])
            yield from self._left_building_block(inp, out, anc, depth + 1)

    def _right_building_block(self, inp, out, anc, depth):
        if depth == 0:
            return
        else:
            yield cirq.CX(anc[depth - 1], anc[depth])
            yield And(adjoint=True).on(inp[depth], out[depth], anc[depth])
            yield cirq.CX(anc[depth - 1], inp[depth])
            yield cirq.CX(inp[depth], out[depth])
            yield from self._right_building_block(inp, out, anc, depth - 1)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        input_bits = quregs['a']
        output_bits = quregs['b']
        ancillas = context.qubit_manager.qalloc(self.bitsize - 1)
        # Start off the addition by anding into the ancilla
        yield And().on(input_bits[0], output_bits[0], ancillas[0])
        # Left part of Fig.2
        yield from self._left_building_block(input_bits, output_bits, ancillas, 1)
        yield cirq.CX(ancillas[-1], output_bits[-1])
        yield cirq.CX(input_bits[-1], output_bits[-1])
        # right part of Fig.2
        yield from self._right_building_block(input_bits, output_bits, ancillas, self.bitsize - 2)
        yield And(adjoint=True).on(input_bits[0], output_bits[0], ancillas[0])
        yield cirq.CX(input_bits[0], output_bits[0])
        context.qubit_manager.qfree(ancillas)

    def _t_complexity_(self):
        num_clifford = (self.bitsize - 2) * 19 + 16
        num_t_gates = 4 * self.bitsize - 4
        return TComplexity(t=num_t_gates, clifford=num_clifford)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        num_clifford = (self.bitsize - 2) * 19 + 16
        num_toffoli = self.bitsize - 1
        return {(Toffoli(), num_toffoli), (ArbitraryClifford(n=1), num_clifford)}


@frozen
class OutOfPlaceAdder(GateWithRegisters, cirq.ArithmeticGate):
    r"""An n-bit addition gate.

    Implements $U|a\rangle|b\rangle 0\rangle \rightarrow |a\rangle|b\rangle|a+b\rangle$
    using $4n - 4 T$ gates. Uncomputation requires 0 T-gates.

    Args:
        bitsize: Number of bits used to represent each input integer. The allocated output register
            is of size `bitsize+1` so it has enough space to hold the sum of `a+b`.

    Registers:
     - a: A bitsize-sized input register (register a above).
     - b: A bitsize-sized input register (register b above).
     - c: A bitize+1-sized LEFT/RIGHT register depending on whether the gate adjoint or not.

    References:
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648)
    """

    bitsize: int
    adjoint: bool = False

    @property
    def signature(self):
        side = Side.LEFT if self.adjoint else Side.RIGHT
        return Signature(
            [
                Register('a', self.bitsize),
                Register('b', self.bitsize),
                Register('c', self.bitsize + 1, side=side),
            ]
        )

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [2] * self.bitsize, [2] * self.bitsize, [2] * (self.bitsize + 1)

    def apply(self, a: int, b: int, c: int) -> Tuple[int, int, int]:
        return a, b, c + a + b

    def on_classical_vals(
        self, a: 'ClassicalValT', b: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        return dict(zip('abc', (a, b, a + b)))

    def with_registers(self, *new_registers: Union[int, Sequence[int]]):
        raise NotImplementedError("no need to implement with_registers.")

    def short_name(self) -> str:
        return "c = a + b"

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        a, b, c = quregs['a'][::-1], quregs['b'][::-1], quregs['c'][::-1]
        optree = [
            [
                [cirq.CX(a[i], b[i]), cirq.CX(a[i], c[i])],
                And().on(b[i], c[i], c[i + 1]),
                [cirq.CX(a[i], b[i]), cirq.CX(a[i], c[i + 1]), cirq.CX(b[i], c[i])],
            ]
            for i in range(self.bitsize)
        ]
        return cirq.inverse(optree) if self.adjoint else optree

    def t_complexity(self) -> TComplexity:
        and_t = And(adjoint=self.adjoint).t_complexity()
        num_clifford = self.bitsize * (5 + and_t.clifford)
        num_t = self.bitsize * and_t.t
        return TComplexity(t=num_t, clifford=num_clifford)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {
            (And(adjoint=self.adjoint), self.bitsize),
            (ArbitraryClifford(n=2), 5 * self.bitsize),
        }

    def __pow__(self, power: int):
        if power == 1:
            return self
        if power == -1:
            return OutOfPlaceAdder(self.bitsize, adjoint=not self.adjoint)
        raise NotImplementedError("OutOfPlaceAdder.__pow__ defined only for +1/-1.")


@frozen
class HammingWeightCompute(GateWithRegisters):
    r"""A gate to compute the hamming weight of an n-bit register in a new log_{n} bit register.

    Implements $U|x\rangle |0\rangle \rightarrow |x\rangle|\text{hamming\_weight}(x)\rangle$
    using $\alpha$ Toffoli gates and $\alpha$ ancilla qubits, where
    $\alpha = n - \text{hamming\_weight}(n)$ for an n-bit input register.

    Args:
        bitsize: Number of bits in the input register. The allocated output register
            is of size $\log_2(\text{bitsize})$ so it has enough space to hold the hamming weight
            of x.

    Registers:
     - x: A $\text{bitsize}$-sized input register (register x above).
     - junk: A LEFT/RIGHT ancilla register, depending on whether gate is adjoint or not,
        of size $\text{bitsize} - \text{hamming\_weight(bitsize)}$.
     - out: A LEFT/RIGHT output register, depending on whether the gate is adjoint or not,
        of size $\log_2(\text{bitize})$.

    References:
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648), Page-4
    """

    bitsize: int
    adjoint: bool = False

    @cached_property
    def signature(self):
        side = Side.LEFT if self.adjoint else Side.RIGHT
        return Signature(
            [
                Register('x', self.bitsize),
                Register('junk', self.bitsize - self.bitsize.bit_count(), side=side),
                Register('out', self.bitsize.bit_length(), side=side),
            ]
        )

    def short_name(self) -> str:
        return "out = x.bit_count()"

    def _three_to_two_adder(self, a, b, c, out) -> cirq.OP_TREE:
        return [
            [cirq.CX(a, b), cirq.CX(a, c)],
            And().on(b, c, out),
            [cirq.CX(a, b), cirq.CX(a, out), cirq.CX(b, c)],
        ]

    def _decompose_using_three_to_two_adders(
        self, x: List[cirq.Qid], junk: List[cirq.Qid], out: List[cirq.Qid]
    ) -> cirq.OP_TREE:
        for out_idx in range(len(out)):
            y = []
            for in_idx in range(0, len(x) - 2, 2):
                a, b, c = x[in_idx], x[in_idx + 1], x[in_idx + 2]
                anc = junk.pop()
                y.append(anc)
                yield self._three_to_two_adder(a, b, c, anc)
            if len(x) % 2 == 1:
                yield cirq.CNOT(x[-1], out[out_idx])
            else:
                anc = junk.pop()
                yield self._three_to_two_adder(x[-2], x[-1], out[out_idx], anc)
                y.append(anc)
            x = [*y]

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        # Qubit order needs to be reversed because the registers store Big Endian representation
        # of integers.
        x: List[cirq.Qid] = [*quregs['x'][::-1]]
        junk: List[cirq.Qid] = [*quregs['junk'][::-1]]
        out: List[cirq.Qid] = [*quregs['out'][::-1]]
        optree = self._decompose_using_three_to_two_adders(x, junk, out)
        return cirq.inverse(optree) if self.adjoint else optree

    def t_complexity(self) -> TComplexity:
        and_t = And(adjoint=self.adjoint).t_complexity()
        junk_bitsize = self.bitsize - self.bitsize.bit_count()
        num_clifford = junk_bitsize * (5 + and_t.clifford) + self.bitsize.bit_count()
        num_t = junk_bitsize * and_t.t
        return TComplexity(t=num_t, clifford=num_clifford)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {
            (And(adjoint=self.adjoint), self.bitsize),
            (ArbitraryClifford(n=2), 5 * self.bitsize),
        }

    def __pow__(self, power: int):
        if power == 1:
            return self
        if power == -1:
            return HammingWeightCompute(self.bitsize, adjoint=not self.adjoint)
        raise NotImplementedError("HammingWeightCompute.__pow__ defined only for +1/-1.")


@frozen(auto_attribs=True)
class AddConstantMod(GateWithRegisters, cirq.ArithmeticGate):
    """Applies U_{M}_{add}|x> = |(x + add) % M> if x < M else |x>.

    Applies modular addition to input register `|x>` given parameters `mod` and `add_val` s.t.
        1) If integer `x` < `mod`: output is `|(x + add) % M>`
        2) If integer `x` >= `mod`: output is `|x>`.

    This condition is needed to ensure that the mapping of all input basis states (i.e. input
    states |0>, |1>, ..., |2 ** bitsize - 1) to corresponding output states is bijective and thus
    the gate is reversible.

    Also supports controlled version of the gate by specifying a per qubit control value as a tuple
    of integers passed as `cvs`.
    """

    bitsize: int
    mod: int = field()
    add_val: int = 1
    cvs: Tuple[int, ...] = field(
        converter=lambda v: (v,) if isinstance(v, int) else tuple(v), default=()
    )

    @mod.validator
    def _validate_mod(self, attribute, value):
        if not 1 <= value <= 2**self.bitsize:
            raise ValueError(f"mod: {value} must be between [1, {2 ** self.bitsize}].")

    @cached_property
    def signature(self) -> Signature:
        if self.cvs:
            return Signature.build(ctrl=len(self.cvs), x=self.bitsize)
        return Signature.build(x=self.bitsize)

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        add_reg = (2,) * self.bitsize
        control_reg = (2,) * len(self.cvs)
        return (control_reg, add_reg) if control_reg else (add_reg,)

    def with_registers(self, *new_registers: Union[int, Sequence[int]]) -> "AddMod":
        raise NotImplementedError()

    def apply(self, *args) -> Union[int, Iterable[int]]:
        target_val = args[-1]
        if target_val < self.mod:
            new_target_val = (target_val + self.add_val) % self.mod
        else:
            new_target_val = target_val
        if self.cvs and args[0] != int(''.join(str(x) for x in self.cvs), 2):
            new_target_val = target_val
        ret = (args[0], new_target_val) if self.cvs else (new_target_val,)
        return ret

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['@' if b else '@(0)' for b in self.cvs]
        wire_symbols += [f"Add_{self.add_val}_Mod_{self.mod}"] * self.bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __pow__(self, power: int) -> 'AddConstantMod':
        return AddConstantMod(self.bitsize, self.mod, add_val=self.add_val * power, cvs=self.cvs)

    def _t_complexity_(self) -> TComplexity:
        # Rough cost as given in https://arxiv.org/abs/1905.09749
        return 5 * Add(self.bitsize).t_complexity()


@frozen
class Square(Bloq):
    r"""Square an n-bit binary number.

    Implements $U|a\rangle|0\rangle \rightarrow |a\rangle|a^2\rangle$ using $n^2 - n$ Toffolis.

    Args:
        bitsize: Number of bits used to represent the integer to be squared. The
            result is stored in a register of size 2*bitsize.

    Registers:
        a: A bitsize-sized input register (register a above).
        result: A 2-bitsize-sized input/output register.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First
        Quantization](https://arxiv.org/abs/2105.12767). pg 76 for Toffoli complexity.
    """

    bitsize: int

    @property
    def signature(self):
        return Signature(
            [Register("a", self.bitsize), Register("result", 2 * self.bitsize, side=Side.RIGHT)]
        )

    def short_name(self) -> str:
        return "a^2"

    def t_complexity(self):
        # TODO Determine precise clifford count and/or ignore.
        # See: https://github.com/quantumlib/cirq-qubitization/issues/219
        # See: https://github.com/quantumlib/cirq-qubitization/issues/217
        num_toff = self.bitsize * (self.bitsize - 1)
        return TComplexity(t=4 * num_toff)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        num_toff = self.bitsize * (self.bitsize - 1)
        return {(Toffoli(), num_toff)}


@frozen
class SumOfSquares(Bloq):
    r"""Compute the sum of squares of k n-bit binary numbers.

    Implements $U|a\rangle|b\rangle\dots k\rangle|0\rangle \rightarrow
        |a\rangle|b\rangle\dots|k\rangle|a^2+b^2+\dots k^2\rangle$ using
        $4 k n^2 T$ gates.

    The number of bits required by the output register is 2*bitsize + ceil(log2(k)).

    Args:
        bitsize: Number of bits used to represent each of the k integers.
        k: The number of integers we want to square.

    Registers:
        input: k n-bit registers.
        result: 2 * bitsize + ceil(log2(k)) sized output register.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First
        Quantization](https://arxiv.org/abs/2105.12767) pg 80 give a Toffoli
        complexity for squaring.
    """

    bitsize: int
    k: int

    @property
    def signature(self):
        return Signature(
            [
                Register("input", bitsize=self.bitsize, shape=(self.k,)),
                Register(
                    "result", bitsize=2 * self.bitsize + (self.k - 1).bit_length(), side=Side.RIGHT
                ),
            ]
        )

    def short_name(self) -> str:
        return "SOS"

    def t_complexity(self):
        # TODO Determine precise clifford count and/or ignore.
        # See: https://github.com/quantumlib/cirq-qubitization/issues/219
        # See: https://github.com/quantumlib/cirq-qubitization/issues/217
        num_toff = self.k * self.bitsize**2 - self.bitsize
        if self.k % 3 == 0:
            num_toff -= 1
        return TComplexity(t=4 * num_toff)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        num_toff = self.k * self.bitsize**2 - self.bitsize
        if self.k % 3 == 0:
            num_toff -= 1
        return {(Toffoli(), num_toff)}


@frozen
class Product(Bloq):
    r"""Compute the product of an `n` and `m` bit binary number.

    Implements $U|a\rangle|b\rangle|0\rangle -\rightarrow
    |a\rangle|b\rangle|a\times b\rangle$ using $2nm-n$ Toffolis.

    Args:
        a_bitsize: Number of bits used to represent the first integer.
        b_bitsize: Number of bits used to represent the second integer.

    Registers:
        a: a_bitsize-sized input register.
        b: b_bitsize-sized input register.
        result: A 2*max(a_bitsize, b_bitsize) bit-sized output register to store the result a*b.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First
        Quantization](https://arxiv.org/abs/2105.12767) pg 81 gives a Toffoli
        complexity for multiplying two numbers.
    """

    a_bitsize: int
    b_bitsize: int

    @property
    def signature(self):
        return Signature(
            [
                Register("a", self.a_bitsize),
                Register("b", self.b_bitsize),
                Register("result", 2 * max(self.a_bitsize, self.b_bitsize), side=Side.RIGHT),
            ]
        )

    def short_name(self) -> str:
        return "a*b"

    def t_complexity(self):
        # TODO Determine precise clifford count and/or ignore.
        # See: https://github.com/quantumlib/cirq-qubitization/issues/219
        # See: https://github.com/quantumlib/cirq-qubitization/issues/217
        num_toff = 2 * self.a_bitsize * self.b_bitsize - max(self.a_bitsize, self.b_bitsize)
        return TComplexity(t=4 * num_toff)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        num_toff = 2 * self.a_bitsize * self.b_bitsize - max(self.a_bitsize, self.b_bitsize)
        return {(Toffoli(), num_toff)}


@frozen
class ScaleIntByReal(Bloq):
    r"""Scale an integer by fixed-point representation of a real number.

    i.e.

    $$
        |r\rangle|i\rangle|0\rangle \rightarrow |r\rangle|i\rangle|r \times i\rangle
    $$

    The real number is assumed to be in the range [0, 1).

    Args:
        r_bitsize: Number of bits used to represent the real number.
        i_bitsize: Number of bits used to represent the integer.

    Registers:
     - real_in: r_bitsize-sized input register.
     - int_in: i_bitsize-sized input register.
     - result: r_bitsize output register

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization]
        (https://arxiv.org/pdf/2007.07391.pdf) pg 70.
    """

    r_bitsize: int
    i_bitsize: int

    @property
    def signature(self):
        return Signature(
            [
                Register("real_in", self.r_bitsize),
                Register("int_in", self.i_bitsize),
                Register("result", self.r_bitsize, side=Side.RIGHT),
            ]
        )

    def short_name(self) -> str:
        return "r*i"

    def t_complexity(self):
        # Eq. D8, we are assuming dA and dB there are assumed as inputs and the
        # user has ensured these are large enough for their desired precision.
        num_toff = self.r_bitsize * (2 * self.i_bitsize - 1) - self.i_bitsize**2
        return TComplexity(t=4 * num_toff)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Eq. D8, we are assuming dA(r_bitsize) and dB(i_bitsize) are inputs and
        # the user has ensured these are large enough for their desired
        # precision.
        num_toff = self.r_bitsize * (2 * self.i_bitsize - 1) - self.i_bitsize**2
        return {(Toffoli(), num_toff)}


@frozen
class MultiplyTwoReals(Bloq):
    r"""Multiply two fixed-point representations of real numbers

    i.e.

    $$
        |a\rangle|b\rangle|0\rangle \rightarrow |a\rangle|b\rangle|a \times b\rangle
    $$

    The real numbers are assumed to be in the range [0, 1).

    Args:
        bitsize: Number of bits used to represent the real number.

    Registers:
     - a: bitsize-sized input register.
     - b: bitsize-sized input register.
     - result: bitsize output register

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization]
            (https://arxiv.org/pdf/2007.07391.pdf) pg 71.
    """

    bitsize: int

    @property
    def signature(self):
        return Signature(
            [
                Register("a", self.bitsize),
                Register("b", self.bitsize),
                Register("result", self.bitsize, side=Side.RIGHT),
            ]
        )

    def short_name(self) -> str:
        return "a*b"

    def t_complexity(self):
        # Eq. D13, there it is suggested keeping both registers the same size is optimal.
        num_toff = self.bitsize**2 - self.bitsize - 1
        return TComplexity(t=4 * num_toff)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Eq. D13, there it is suggested keeping both registers the same size is optimal.
        num_toff = self.bitsize**2 - self.bitsize - 1
        return {(Toffoli(), num_toff)}


@frozen
class SquareRealNumber(Bloq):
    r"""Square a fixed-point representation of a real number

    i.e.

    $$
        |a\rangle|0\rangle \rightarrow |a\rangle|a^2\rangle
    $$

    The real numbers are assumed to be in the range [0, 1).

    Args:
        bitsize: Number of bits used to represent the real number.

    Registers:
     - a: bitsize-sized input register.
     - b: bitsize-sized input register.
     - result: bitsize output register

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization
            ](https://arxiv.org/pdf/2007.07391.pdf) pg 74.
    """

    bitsize: int

    def __attrs_post_init__(self):
        if self.bitsize < 3:
            raise ValueError("bitsize must be at least 3 for SquareRealNumber bloq to make sense.")

    @property
    def signature(self):
        return Signature(
            [
                Register("a", self.bitsize),
                Register("b", self.bitsize),
                Register("result", self.bitsize, side=Side.RIGHT),
            ]
        )

    def short_name(self) -> str:
        return "a^2"

    def t_complexity(self):
        num_toff = self.bitsize**2 // 2 - 4
        return TComplexity(t=4 * num_toff)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Bottom of page 74
        num_toff = self.bitsize**2 // 2 - 4
        return {(Toffoli(), num_toff)}


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
        return Signature.build(a=self.a_bitsize, b=self.b_bitsize, target=1)

    def short_name(self) -> str:
        return "a>b"

    def t_complexity(self) -> 'TComplexity':
        return t_complexity(LessThanEqual(self.a_bitsize, self.b_bitsize))

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # TODO Determine precise clifford count and/or ignore.
        # See: https://github.com/quantumlib/cirq-qubitization/issues/219
        # See: https://github.com/quantumlib/cirq-qubitization/issues/217
        t_complexity = self.t_complexity()
        return {(TGate(), t_complexity.t)}


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
        return Signature.build(x=self.bitsize, target=1)

    def t_complexity(self) -> TComplexity:
        return t_complexity(LessThanConstant(self.bitsize, less_than_val=self.val))

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # TODO Determine precise clifford count and/or ignore.
        # See: https://github.com/quantumlib/cirq-qubitization/issues/219
        # See: https://github.com/quantumlib/cirq-qubitization/issues/217
        t_complexity = self.t_complexity()
        return {(TGate(), t_complexity.t)}


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
        return Signature.build(x=self.bitsize, target=1)

    def t_complexity(self) -> 'TComplexity':
        return TComplexity(t=4 * (self.bitsize - 1))

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # See: https://github.com/quantumlib/cirq-qubitization/issues/219
        # See: https://github.com/quantumlib/cirq-qubitization/issues/217
        return {(TGate(), 4 * (self.bitsize - 1))}


@frozen
class ToContiguousIndex(Bloq):
    r"""Build a contiguous register s from mu and nu.

    $$
        s = \nu (\nu + 1) / 2 + \mu
    $$

    Assuming nu is zero indexed (in contrast to the THC paper which assumes 1,
    hence the slightly different formula).

    Args:
        bitsize: number of bits for mu and nu registers.
        s_bitsize: Number of bits for contiguous register.

    Registers:
        mu: input register
        nu: input register
        s: output contiguous register

    References:
        (Even more efficient quantum computations of chemistry through
        tensor hypercontraction)[https://arxiv.org/pdf/2011.03494.pdf] Eq. 29.
    """

    bitsize: int
    s_bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("mu", bitsize=self.bitsize),
                Register("nu", bitsize=self.bitsize),
                Register("s", bitsize=self.s_bitsize),
            ]
        )

    def on_classical_vals(
        self, mu: 'ClassicalValT', nu: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        return {'mu': mu, 'nu': nu, 's': nu * (nu + 1) // 2 + mu}

    def t_complexity(self) -> 'TComplexity':
        num_toffoli = self.bitsize**2 + self.bitsize - 1
        return TComplexity(t=4 * num_toffoli)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        num_toffoli = self.bitsize**2 + self.bitsize - 1
        return {(Toffoli(), num_toffoli)}


@frozen
class SignedIntegerToTwosComplement(Bloq):
    """Convert a register storing the signed integer representation to two's complement inplace.

    Args:
        bitsize: size of the register.

    Regs:
        x: input signed integer register to convert to two-complement.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) page 24, 4th paragraph from the bottom.
    """

    bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(x=self.bitsize)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Take the sign qubit as a control and cnot the remaining qubits, then
        # add it to the remaining n-1 bits.
        return {(Toffoli(), (self.bitsize - 2))}
