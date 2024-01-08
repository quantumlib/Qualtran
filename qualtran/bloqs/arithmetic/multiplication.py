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

from typing import Any, Dict, Iterable, Sequence, Set, TYPE_CHECKING, Union

import cirq
from attrs import frozen

from qualtran import Bloq, GateWithRegisters, Register, Side, Signature
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.cirq_interop.t_complexity_protocol import TComplexity

if TYPE_CHECKING:
    from qualtran import SoquetT
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class PlusEqualProduct(GateWithRegisters, cirq.ArithmeticGate):
    """Performs result += a * b"""

    a_bitsize: int
    b_bitsize: int
    result_bitsize: int
    adjoint: bool = False

    def short_name(self) -> str:
        return "result -= a*b" if self.adjoint else "result += a*b"

    @property
    def signature(self) -> 'Signature':
        return Signature.build(a=self.a_bitsize, b=self.b_bitsize, result=self.result_bitsize)

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [2] * self.a_bitsize, [2] * self.b_bitsize, [2] * self.result_bitsize

    def apply(self, a: int, b: int, result: int) -> Union[int, Iterable[int]]:
        return a, b, (result + a * b * ((-1) ** self.adjoint)) % (2**self.result_bitsize)

    def with_registers(self, *new_registers: Union[int, Sequence[int]]):
        raise NotImplementedError("Not needed.")

    def on_classical_vals(self, a: int, b: int, result: int) -> Dict[str, 'ClassicalValT']:
        result_out = (result + a * b * ((-1) ** self.adjoint)) % (2**self.result_bitsize)
        return {'a': a, 'b': b, 'result': result_out}

    def _t_complexity_(self) -> 'TComplexity':
        # TODO: The T-complexity here is approximate.
        return TComplexity(t=8 * max(self.a_bitsize, self.b_bitsize) ** 2)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['a'] * self.a_bitsize + ['b'] * self.b_bitsize
        wire_symbols += ['c-=a*b' if self.adjoint else 'c+=a*b'] * self.result_bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return PlusEqualProduct(
                self.a_bitsize, self.b_bitsize, self.result_bitsize, not self.adjoint
            )
        raise NotImplementedError("PlusEqualProduct.__pow__ defined only for powers +1/-1.")

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        from qualtran.cirq_interop._cirq_to_bloq import _add_my_tensors_from_gate

        _add_my_tensors_from_gate(
            self, self.signature, self.short_name(), tn, tag, incoming=incoming, outgoing=outgoing
        )


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
        # See: https://github.com/quantumlib/Qualtran/issues/219
        # See: https://github.com/quantumlib/Qualtran/issues/217
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
        # See: https://github.com/quantumlib/Qualtran/issues/219
        # See: https://github.com/quantumlib/Qualtran/issues/217
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
        # See: https://github.com/quantumlib/Qualtran/issues/219
        # See: https://github.com/quantumlib/Qualtran/issues/217
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
