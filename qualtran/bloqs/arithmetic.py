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

from typing import Optional, Set, Tuple, TYPE_CHECKING

from attrs import frozen
from cirq_ft import TComplexity

from qualtran import Bloq, Register, Side, Signature
from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.util_bloqs import ArbitraryClifford

if TYPE_CHECKING:
    from qualtran.resource_counting import SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class Add(Bloq):
    r"""An n-bit addition gate.

    Implements $U|a\rangle|b\rangle \rightarrow |a\rangle|a+b\rangle$ using $4n - 4 T$ gates.

    Args:
        bitsize: Number of bits used to represent each integer. Must be large
            enough to hold the result in the output register of a + b.

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

    def pretty_name(self) -> str:
        return "a + b"

    def t_complexity(self):
        num_clifford = (self.bitsize - 2) * 19 + 16
        num_t_gates = 4 * self.bitsize - 4
        return TComplexity(t=num_t_gates, clifford=num_clifford)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        num_clifford = (self.bitsize - 2) * 19 + 16
        num_t_gates = 4 * self.bitsize - 4
        return {(num_t_gates, TGate()), (num_clifford, ArbitraryClifford(n=1))}


@frozen
class OutOfPlaceAdder(Bloq):
    r"""An n-bit addition gate.

    Implements $U|a\rangle|b\rangle 0\rangle \rightarrow |a\rangle|b\rangle|a+b\rangle$
    using $4n - 4 T$ gates.

    Args:
        bitsize: Number of bits used to represent each integer. Must be large
            enough to hold the result in the output register of a + b.

    Registers:
     - a: A bitsize-sized input register (register a above).
     - b: A bitsize-sized input/output register (register b above).

    References:
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648)
    """

    bitsize: int

    @property
    def signature(self):
        return Signature.build(a=self.bitsize, b=self.bitsize, c=self.bitsize)

    def pretty_name(self) -> str:
        return "c = a + b"

    def t_complexity(self):
        # TODO: This is just copied from adder above, not sure if correct.
        # see https://github.com/quantumlib/Qualtran/issues/375
        num_clifford = (self.bitsize - 2) * 19 + 16
        num_t_gates = 4 * self.bitsize - 4
        return TComplexity(t=num_t_gates, clifford=num_clifford)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        num_clifford = (self.bitsize - 2) * 19 + 16
        num_t_gates = 4 * self.bitsize - 4
        return {(num_t_gates, TGate()), (num_clifford, ArbitraryClifford(n=1))}


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

    def pretty_name(self) -> str:
        return "a^2"

    def t_complexity(self):
        # TODO Determine precise clifford count and/or ignore.
        # See: https://github.com/quantumlib/cirq-qubitization/issues/219
        # See: https://github.com/quantumlib/cirq-qubitization/issues/217
        num_toff = self.bitsize * (self.bitsize - 1)
        return TComplexity(t=4 * num_toff)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        num_toff = self.bitsize * (self.bitsize - 1)
        return {(4 * num_toff, TGate())}


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

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        num_toff = self.k * self.bitsize**2 - self.bitsize
        if self.k % 3 == 0:
            num_toff -= 1
        return {(4 * num_toff, TGate())}


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

    def pretty_name(self) -> str:
        return "a*b"

    def t_complexity(self):
        # TODO Determine precise clifford count and/or ignore.
        # See: https://github.com/quantumlib/cirq-qubitization/issues/219
        # See: https://github.com/quantumlib/cirq-qubitization/issues/217
        num_toff = 2 * self.a_bitsize * self.b_bitsize - max(self.a_bitsize, self.b_bitsize)
        return TComplexity(t=4 * num_toff)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        num_toff = 2 * self.a_bitsize * self.b_bitsize - max(self.a_bitsize, self.b_bitsize)
        return {(4 * num_toff, TGate())}


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

    def pretty_name(self) -> str:
        return "r*i"

    def t_complexity(self):
        # Eq. D8, we are assuming dA and dB there are assumed as inputs and the
        # user has ensured these are large enough for their desired precision.
        num_toff = self.r_bitsize * (2 * self.i_bitsize - 1) - self.i_bitsize**2
        return TComplexity(t=4 * num_toff)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # Eq. D8, we are assuming dA(r_bitsize) and dB(i_bitsize) are inputs and
        # the user has ensured these are large enough for their desired
        # precision.
        num_toff = self.r_bitsize * (2 * self.i_bitsize - 1) - self.i_bitsize**2
        return {(4 * num_toff, TGate())}


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

    def pretty_name(self) -> str:
        return "a*b"

    def t_complexity(self):
        # Eq. D13, there it is suggested keeping both registers the same size is optimal.
        num_toff = self.bitsize**2 - self.bitsize - 1
        return TComplexity(t=4 * num_toff)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # Eq. D13, there it is suggested keeping both registers the same size is optimal.
        num_toff = self.bitsize**2 - self.bitsize - 1
        return {(4 * num_toff, TGate())}


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

    def pretty_name(self) -> str:
        return "a^2"

    def t_complexity(self):
        num_toff = self.bitsize**2 // 2 - 4
        return TComplexity(t=4 * num_toff)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        # Bottom of page 74
        num_toff = self.bitsize**2 // 2 - 4
        return {(4 * num_toff, TGate())}


@frozen
class GreaterThan(Bloq):
    r"""Compare two n-bit integers.

    Implements $U|a\rangle|b\rangle|0\rangle \rightarrow
    |a\rangle|b\rangle|a > b\rangle$ using $8n T$  gates.


    Args:
        bitsize: Number of bits used to represent the two integers a and b.

    Registers:
        a: n-bit-sized input registers.
        b: n-bit-sized input registers.
        result: A single bit output register to store the result of A > B.

    References:
        [Improved techniques for preparing eigenstates of fermionic
        Hamiltonians](https://www.nature.com/articles/s41534-018-0071-5#additional-information),
        Comparison Oracle from SI: Appendix 2B (pg 3)
    """
    bitsize: int

    @property
    def signature(self):
        return Signature.build(a=self.bitsize, b=self.bitsize, result=1)

    def pretty_name(self) -> str:
        return "a gt b"

    def t_complexity(self):
        # TODO Determine precise clifford count and/or ignore.
        # See: https://github.com/quantumlib/cirq-qubitization/issues/219
        # See: https://github.com/quantumlib/cirq-qubitization/issues/217
        return TComplexity(t=8 * self.bitsize)

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, Bloq]]:
        return {(8 * self.bitsize, TGate())}
