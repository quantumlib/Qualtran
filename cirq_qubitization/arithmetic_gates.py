from re import I
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


class MultiInLessThanEqualGate(cirq.ArithmeticGate):
    """Applies U|x>|y>|z> = |x>|y> |z ^ (x <= y)>"""

    def __init__(self, first_input_register: Sequence[int], second_input_register: Sequence[int]) -> None:
        self._first_input_register = first_input_register  # |x>
        self._second_input_register = second_input_register # |y>
        self._target_register = [2]  # |z>

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return self._first_input_register, self._second_input_register, self._target_register

    def with_registers(
        self, *new_registers: Union[int, Sequence[int]]
    ) -> "MultiInLessThanEqualGate":
        return MultiInLessThanEqualGate(new_registers[0], new_registers[1])

    def apply(
        self, first_input_val, second_input_val, target_register_val
    ) -> Union[int, int, Iterable[int]]:
        return first_input_val, second_input_val, target_register_val ^ (first_input_val <= second_input_val)

    def __repr__(self):
        return f"cirq_qubitization.MultiInLessThanEqualGate({self._first_input_register, self._second_input_register})"


if __name__ == "__main__":
    import itertools
    circuit = cirq.Circuit(
        MultiInLessThanEqualGate([2, 2, 2], [2, 2, 2]).on(*cirq.LineQubit.range(7))
    )
    maps = {}
    for in1, in2 in itertools.product(range(2**3), repeat=2):
        for target_reg_val in range(2):
            target_bin = bin(target_reg_val)[2:] 
            in1_bin = bin(in1)[2:]
            in2_bin = bin(in2)[2:]
            out_bin = bin(target_reg_val ^ (in1 <= in2))[2:]
            true_out_int = target_reg_val ^ (in1 <= in2)
            input_int = int(in1_bin + in2_bin + target_bin, 2)
            output_int = int(in1_bin + in2_bin + out_bin, 2)
            print(in1, in2, target_reg_val, true_out_int, int(out_bin, 2))
            assert true_out_int == int(out_bin, 2)
            maps[input_int] = output_int
    cirq.testing.assert_equivalent_computational_basis_map(maps, circuit)
