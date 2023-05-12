from typing import Iterable, Sequence, Tuple, Union

import cirq

from cirq_qubitization import bit_tools, cirq_infra, t_complexity_protocol
from cirq_qubitization.cirq_algos.and_gate import And


class LessThanGate(cirq.ArithmeticGate):
    """Applies U_a|x>|z> = |x> |z ^ (x < a)>"""

    def __init__(self, input_register: Sequence[int], val: int) -> None:
        self._input_register = input_register
        self._val = val
        self._target_register = [2]

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return self._input_register, self._val, self._target_register

    def with_registers(self, *new_registers: Union[int, Sequence[int]]) -> "LessThanGate":
        return LessThanGate(new_registers[0], new_registers[1])

    def apply(self, input_val, max_val, target_register_val) -> Union[int, Iterable[int]]:
        return input_val, max_val, target_register_val ^ (input_val < max_val)

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["In(x)"] * len(self._input_register)
        wire_symbols += [f'+(x < {self._val})']
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __pow__(self, power: int):
        if power in [1, -1]:
            return self
        return NotImplemented

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        """Decomposes the gate into 4N And and And† operations for a T complexity of 4N.

        The decomposition proceeds from the most significant qubit -bit 0- to the least significant qubit
        while maintaining whether the qubit sequence is equal to the current prefix of the `_val` or not.

        The bare-bone logic is:
        1. if ith bit of `_val` is 1 then:
            - the qubit sequence is less than `_val` iff they are equal so far and the current qubit is 0.
        2. update are_equal: `are_equal := are_equal and (ith bit == ith qubit).`

        This logic is implemented using $n$ And & And† operations and n+1 clean ancillas where
            - one ancilla `are_equal` contains the equality informaiton
            - ancilla[i] contain whether the qubits[:i+1] != (i+1)th prefix of `_val`
        """

        qubits, target = qubits[:-1], qubits[-1]
        # Trivial case, self._val is larger than any value the registers could represent
        if self._val >= 2 ** len(self._input_register):
            yield cirq.X(target)
            return
        adjoint = []

        [are_equal] = cirq_infra.qalloc(1)

        # Initially our belief is that the numbers are equal.
        yield cirq.X(are_equal)
        adjoint.append(cirq.X(are_equal))

        # Scan from left to right.
        # `are_equal` contains whether the numbers are equal so far.
        ancilla = cirq_infra.qalloc(len(self._input_register))
        for b, q, a in zip(
            bit_tools.iter_bits(self._val, len(self._input_register)), qubits, ancilla
        ):
            if b:
                yield cirq.X(q)
                adjoint.append(cirq.X(q))

                # ancilla[i] = are_equal so far and (q_i != _val[i]).
                #               = equivalent to: Is the current prefix of the qubits < the prefix of `_val`?
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

                # if a is one then we need to flip `are_equal` since this means that
                # are_qual=1, b_i=0, q_i=1 => the current prefixes are not equal so we need to flip `are_equal`.
                yield cirq.CNOT(a, are_equal)
                adjoint.append(cirq.CNOT(a, are_equal))

        yield from reversed(adjoint)

    def _has_unitary_(self):
        return True

    def _t_complexity_(self) -> t_complexity_protocol.TComplexity:
        n = len(self._input_register)
        if self._val >= 2**n:
            return t_complexity_protocol.TComplexity(clifford=1)
        return t_complexity_protocol.TComplexity(
            t=4 * n, clifford=15 * n + 3 * self._val.bit_count() + 2
        )


def mix_double_qubit_registers(
    x: Tuple[cirq.Qid, cirq.Qid], y: Tuple[cirq.Qid, cirq.Qid]
) -> cirq.OP_TREE:
    """Implements the COMPARE2 circuit (Fig. 1) in https://www.nature.com/articles/s41534-018-0071-5#Sec8"""
    [ancilla] = cirq_infra.qalloc(1)
    x_1, x_0 = x
    y_1, y_0 = y

    def _cswap(c, a, b) -> cirq.CNOT:
        [q] = cirq_infra.qalloc(1)
        yield cirq.CNOT(a, b)
        yield And().on(c, b, q)
        yield cirq.CNOT(q, a)
        yield cirq.CNOT(a, b)

    yield cirq.X(ancilla)
    yield cirq.CNOT(y_1, x_1)
    yield cirq.CNOT(y_0, x_0)
    yield from _cswap(x_1, x_0, ancilla)
    yield from _cswap(x_1, y_1, y_0)
    yield cirq.CNOT(y_0, x_0)


def compare_qubits(
    x: cirq.Qid, y: cirq.Qid, less_than: cirq.Qid, greater_than: cirq.Qid
) -> cirq.OP_TREE:
    """Implements the comparison circuit (Fig. 3) in https://www.nature.com/articles/s41534-018-0071-5#Sec8

    Args:
        x: first qubit of the comparison and stays the same after circuit execution.
        y: second qubit of the comparison. The qubit will store equality value `x==y` after circuit execution.
        less_than: Assumed to be in zero state. Will store `x < y`.
        greater_than: Assumed to be in zero state. Will store `x > y`.
    """

    yield And([0, 1]).on(x, y, less_than)
    yield cirq.CNOT(less_than, greater_than)
    yield cirq.CNOT(y, greater_than)
    yield cirq.CNOT(x, y)
    yield cirq.CNOT(x, greater_than)
    yield cirq.X(y)


class LessThanEqualGate(cirq.ArithmeticGate):
    """Applies U|x>|y>|z> = |x>|y> |z ^ (x <= y)>"""

    def __init__(
        self, first_input_register: Sequence[int], second_input_register: Sequence[int]
    ) -> None:
        self._first_input_register = first_input_register  # |x>
        self._second_input_register = second_input_register  # |y>
        self._target_register = [2]  # |z>

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return (self._first_input_register, self._second_input_register, self._target_register)

    def with_registers(self, *new_registers: Union[int, Sequence[int]]) -> "LessThanEqualGate":
        return LessThanEqualGate(new_registers[0], new_registers[1])

    def apply(
        self, first_input_val, second_input_val, target_register_val
    ) -> Union[int, int, Iterable[int]]:
        return (
            first_input_val,
            second_input_val,
            target_register_val ^ (first_input_val <= second_input_val),
        )

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["In(x)"] * len(self._first_input_register)
        wire_symbols += ["In(y)"] * len(self._second_input_register)
        wire_symbols += ['+(x <= y)']
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __pow__(self, power: int):
        if power in [1, -1]:
            return self
        return NotImplemented

    def _has_unitary_(self):
        return True

    def _decompose_via_tree(
        self, X: Tuple[cirq.Qid], Y: Tuple[cirq.Qid]
    ) -> Tuple[Sequence[cirq.Operation], Tuple[cirq.Qid, cirq.Qid]]:
        assert len(X) == len(Y), f'{len(X)=} != {len(Y)=}'
        if len(X) == 1:
            return [], (X[0], Y[0])
        if len(X) == 2:
            return list(mix_double_qubit_registers(X, Y)), (X[1], Y[1])

        m = len(X) // 2
        op_left, ql = self._decompose_via_tree(X[:m], Y[:m])
        op_right, qr = self._decompose_via_tree(X[m:], Y[m:])
        return op_left + op_right + list(mix_double_qubit_registers(ql, qr)), (ql[1], qr[1])

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        P, Q, target = (
            qubits[: len(self._first_input_register)],
            qubits[len(self._first_input_register) : -1],
            qubits[-1],
        )

        n = min(len(P), len(Q))

        equal_so_far = None
        adjoint = []

        # if one of the registers is longer than the other compute store equality value
        # into `equal_so_far` using d = |len(P) - len(Q)| And operations => 4d T.
        if abs(len(P) - len(Q)) == 1:
            [equal_so_far] = cirq_infra.qalloc(1)
            yield cirq.X(equal_so_far)
            adjoint.append(cirq.X(equal_so_far))

            if len(P) > len(Q):
                yield cirq.CNOT(P[0], equal_so_far)
                adjoint.append(cirq.CNOT(P[0], equal_so_far))
            else:
                yield cirq.CNOT(Q[0], equal_so_far)
                adjoint.append(cirq.CNOT(Q[0], equal_so_far))

                yield cirq.CNOT(Q[0], target)
        elif len(P) > len(Q):
            [equal_so_far] = cirq_infra.qalloc(1)

            m = len(P) - n
            ancilla = cirq_infra.qalloc(m - 2)
            yield And(cv=[0] * m).on(*P[:m], *ancilla, equal_so_far)
            adjoint.append(And(cv=[0] * m, adjoint=True).on(*P[:m], *ancilla, equal_so_far))

        elif len(P) < len(Q):
            [equal_so_far] = cirq_infra.qalloc(1)

            m = len(Q) - n
            ancilla = cirq_infra.qalloc(m - 2)
            yield And(cv=[0] * m)(*Q[:m], *ancilla, equal_so_far)
            adjoint.append(And(cv=[0] * m, adjoint=True)(*Q[:m], *ancilla, equal_so_far))

            yield cirq.X(target), cirq.CNOT(equal_so_far, target)

        # compare the `n` suffix of P and Q
        P = P[-n:]
        Q = Q[-n:]
        decomposition, (x, y) = tuple(self._decompose_via_tree(P, Q))
        yield from decomposition
        adjoint.extend(op**-1 for op in decomposition)

        less_than, greater_than = cirq_infra.qalloc(2)
        decomposition = tuple(compare_qubits(x, y, less_than, greater_than))
        yield from decomposition
        adjoint.extend(op**-1 for op in decomposition)

        if equal_so_far is None:
            yield cirq.CNOT(greater_than, target)
            yield cirq.X(target)
        else:
            [ancilla] = cirq_infra.qalloc(1)
            yield And([1, 0]).on(equal_so_far, greater_than, ancilla)
            adjoint.append(And([1, 0], adjoint=True).on(equal_so_far, greater_than, ancilla))

            yield cirq.CNOT(ancilla, target)

        yield from reversed(adjoint)

    def _t_complexity_(self) -> 't_complexity_protocol.TComplexity':
        n = min(len(self._first_input_register), len(self._second_input_register))
        d = max(len(self._first_input_register), len(self._second_input_register)) - n
        is_second_longer = len(self._second_input_register) > len(self._first_input_register)
        if d == 0:
            return t_complexity_protocol.TComplexity(t=8 * n - 4, clifford=46 * n - 17)
        elif d == 1:
            return t_complexity_protocol.TComplexity(
                t=8 * n, clifford=46 * n + 3 + is_second_longer
            )
        else:
            return t_complexity_protocol.TComplexity(
                t=8 * n + 4 * d - 4, clifford=46 * n + 17 * d - 14 + 2 * is_second_longer
            )


class ContiguousRegisterGate(cirq.ArithmeticGate):
    """Applies U|p>|q>|0> -> |p>|q>|p * (p - 1) / 2 + q>

    This is useful in the case when $|p>$ and $|q>$ represent two selection registers such that
     $q < p$. For example, imagine a classical for-loop over two variables $p$ and $q$:

     >>> for p in range(N);
     >>>     for q in range(p):
     >>>         yield data[p][q] # Iterates over a total of (N * (N - 1)) / 2 elements.

     We can rewrite the above using a single for-loop that uses a "contiguous" variable `i` s.t.

     >>> for i in range((N * (N - 1)) / 2):
     >>>    p = np.floor((1 + np.sqrt(1 + 8 * i)) / 2)
     >>>    q = i - (p * (p - 1)) / 2
     >>>    yield data[p][q]

     Note that both the for-loops iterate over the same ranges and in the same order. The only
     difference is that the second loop is a "flattened" version of the first one.

     Such a flattening of selection registers is useful when we want to load multi dimensional
     data to a target register which is indexed on selection registers $p$ and $q$ such that
     $0<= q <= p < N$ and we want to use a `SelectSwapQROM` to laod this data; which gives a
     sqrt-speedup over a traditional QROM at the cost of using more memory and loading chunks
     of size `sqrt(N)` in a single iteration. See the reference for more details.

     References:
         [Even More Efficient Quantum Computations of Chemistry Through Tensor Hypercontraction]
         (https://arxiv.org/abs/2011.03494)
            Lee et. al. (2020). Appendix F, Page 67.
    """

    def __init__(self, selection_bitsize: int, target_bitsize: int):
        self._p_register = [2] * selection_bitsize
        self._q_register = [2] * selection_bitsize
        self._target_register = [2] * target_bitsize

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return (self._p_register, self._q_register, self._target_register)

    def with_registers(self, *new_registers: Union[int, Sequence[int]]) -> 'ContiguousRegisterGate':
        return ContiguousRegisterGate(len(new_registers[0]), len(new_registers[-1]))

    def apply(self, p: int, q: int, target: int) -> Union[int, Iterable[int]]:
        return p, q, target ^ ((p * (p - 1)) // 2 + q)

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["In(x)"] * len(self._p_register)
        wire_symbols += ["In(y)"] * len(self._q_register)
        wire_symbols += ['+[x(x-1)/2 + y]']
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _t_complexity_(self) -> 't_complexity_protocol.TComplexity':
        # See the linked reference for explanation of the Toffoli complexity.
        toffoli_complexity = t_complexity_protocol.t_complexity(cirq.CCNOT)
        n = len(self._p_register)
        return (n**2 + n - 1) * toffoli_complexity


class AdditionGate(cirq.ArithmeticGate):
    """Applies U|p>|q> -> |p>|p+q>.

    Args:
        bitsize: The number of bits used to represent each integer p and q.
            Note that this adder does not detect overflow if bitsize is not
            large enough to hold p + q.

    References:
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648)
    """

    def __init__(self, bitsize: int):
        self._input_register = [2] * bitsize
        self._output_register = [2] * bitsize
        self._nbits = bitsize

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return (self._input_register, self._output_register)

    def with_registers(self, *new_registers: Union[int, Sequence[int]]) -> 'AdditionGate':
        return AdditionGate(len(new_registers[0]))

    def apply(self, p: int, q: int) -> Union[int, Iterable[int]]:
        return p, p + q

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["In(x)"] * self._nbits
        wire_symbols += ["In(y)/Out(x+y)"] * self._nbits
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _has_unitary_(self):
        return True

    def _left_building_block(self, inp, out, anc, depth):
        if depth == self._nbits - 1:
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

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        input_bits = qubits[: self._nbits]
        output_bits = qubits[self._nbits :]
        ancillas = cirq_infra.qalloc(self._nbits - 1)
        # Start off the addition by anding into the ancilla
        yield And().on(input_bits[0], output_bits[0], ancillas[0])
        # Left part of Fig.2
        yield from self._left_building_block(input_bits, output_bits, ancillas, 1)
        yield cirq.CX(ancillas[-1], output_bits[-1])
        yield cirq.CX(input_bits[-1], output_bits[-1])
        # right part of Fig.2
        yield from self._right_building_block(input_bits, output_bits, ancillas, self._nbits - 2)
        yield And(adjoint=True).on(input_bits[0], output_bits[0], ancillas[0])
        yield cirq.CX(input_bits[0], output_bits[0])
        cirq_infra.qfree(ancillas)

    def _t_complexity_(self) -> 't_complexity_protocol.TComplexity':
        # There are N - 2 building blocks each with one And/And^dag contributing 13 cliffords and 6 CXs.
        # In addition there is one additional And/And^dag pair and 3 CXs.
        num_clifford = (self._nbits - 2) * 19 + 16
        return t_complexity_protocol.TComplexity(t=4 * self._nbits - 4, clifford=num_clifford)
