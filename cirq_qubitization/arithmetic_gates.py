from typing import Union, Sequence, Iterable
import cirq


class LessThanGate(cirq.ArithmeticGate):
    """Applies U_a|x>|z> = |x> |z ^ (x < a)>"""

    def __init__(self, input_register: Sequence[int], val: int) -> None:
        self._input_register = input_register
        self._val = val
        self._target_register = [2]

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return self._input_register, self._val, self._target_register

    def with_registers(
        self, *new_registers: Union[int, Sequence[int]]
    ) -> "LessThanGate":
        return LessThanGate(new_registers[0], new_registers[1])

    def apply(
        self, input_val, max_val, target_register_val
    ) -> Union[int, Iterable[int]]:
        return input_val, max_val, target_register_val ^ (input_val < max_val)

    def __repr__(self):
        return f"cirq_qubitization.LessThanGate({self._input_register, self._val})"
