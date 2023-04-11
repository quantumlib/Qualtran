from typing import Iterable, Sequence, Union

import cirq

from cirq_qubitization import t_complexity_protocol


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
