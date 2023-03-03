from typing import Iterable, Sequence, Union

import cirq

from cirq_qubitization import bit_tools
from cirq_qubitization.and_gate import And
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
        qubits, target = qubits[:-1], qubits[-1]
        # Trivial case, self._val is larger than any value the registers could represent
        if self._val >= 2 ** len(self._input_register):
            yield cirq.X(target)
            return
        adjoint = []

        # Initially our belief is that the numbers are equal.
        are_equal = cirq.NamedQubit('e')
        yield cirq.X(are_equal)
        adjoint.append(cirq.X(are_equal))

        # Scan from left to right.
        # `are_equal` contains whether the numbers are equal so far.
        ancilla = cirq.NamedQubit.range(len(self._input_register), prefix='c')
        for b, q, a in zip(
            bit_tools.iter_bits(self._val, len(self._input_register)), qubits, ancilla
        ):
            if b:
                yield cirq.X(q)
                adjoint.append(cirq.X(q))

                yield And().on(q, are_equal, a)
                adjoint.append(And(adjoint=True).on(q, are_equal, a))

                yield cirq.CNOT(a, target)

                yield cirq.CNOT(a, are_equal)
                adjoint.append(cirq.CNOT(a, are_equal))
            else:
                yield And().on(q, are_equal, a)
                adjoint.append(And(adjoint=True).on(q, are_equal, a))

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
