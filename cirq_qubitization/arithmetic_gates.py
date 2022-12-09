from typing import Union, Sequence, Iterable
import cirq

from cirq_qubitization import bit_tools
from cirq_qubitization.and_gate import And

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
        # trivial case, self._val is larger than any value the registers could represent
        if self._val >= 2**len(self._input_register):
            yield cirq.CNOT(target)
            return
        
        # scanning from left to right, initially our belief is that the numbers are equal.
        equal = cirq.NamedQubit('equal')
        yield cirq.X(equal)

        # uses n+1 ancilla => finishes with n+1  dirty ancilla
        # ancilla = cirq.NamedQubit.range(len(qubits), prefix='ancilla')
        # for a, b, q in zip(ancilla, bit_tools.iter_bits(self._val, len(self._input_register)), qubits):
        #     if b:
        #         yield cirq.X(q)
        #         yield cirq.CCNOT(q, equal, a)
        #         yield cirq.CNOT(a, target)
        #         yield cirq.CNOT(a, equal)
        #         yield cirq.X(q)
        #     else:
        #         yield cirq.CCNOT(q, equal, a)
        #         yield cirq.CNOT(a, equal)

        # uses 2 ancillas => finishes with 1 dirty
        # might have phase error
        ancilla = cirq.NamedQubit('ancilla')
        for b, q in zip(bit_tools.iter_bits(self._val, len(self._input_register)), qubits):
            if b:
                yield cirq.X(q)
                yield And().on(q, equal, ancilla)
                yield cirq.CNOT(ancilla, target)
                yield cirq.CNOT(ancilla, equal)
                yield And(adjoint=True).on(q, equal, ancilla)
                yield cirq.X(q)
            else:
                yield And().on(q, equal, ancilla)
                yield cirq.CNOT(ancilla, equal)
                yield And(adjoint=True).on(q, equal, ancilla)

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
