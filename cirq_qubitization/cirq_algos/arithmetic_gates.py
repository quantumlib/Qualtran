from typing import Iterable, Sequence, Union

import cirq

from cirq_qubitization import bit_tools, cirq_infra
from cirq_qubitization.cirq_algos.and_gate import And
from cirq_qubitization.t_complexity_protocol import TComplexity


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

    def __repr__(self) -> str:
        return f"cirq_qubitization.LessThanGate({self._input_register, self._val})"

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        """Decomposes the gate into 4N And and And† operations for a T complexity of 4N.

        The decomposition proceeds from the most significant qubit -bit 0- to the least significant qubit
        while maintaining whether the qubit sequence is equal to the current prefix of the `_val` or not.

        The bare-bone logic is:
        if ith bit of `_val` is 1 the
            the qubit sequence is less than `_val` iff they are equal so far and the current qubit is 0.
        are_equal &= ith bit == ith qubit.

        This logic is implemented using $n$ And & And† operations and n+1 ancillas where
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

    def _t_complexity_(self) -> TComplexity:
        n = len(self._input_register)
        if self._val >= 2**n:
            return TComplexity(clifford=1)
        return TComplexity(t=4 * n, clifford=15 * n + 3 * self._val.bit_count() + 2)


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

    def __repr__(self) -> str:
        return f"cirq_qubitization.LessThanEqualGate({self._first_input_register, self._second_input_register})"
