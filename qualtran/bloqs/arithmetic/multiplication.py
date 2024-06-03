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
import numpy as np
from attrs import evolve, frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    GateWithRegisters,
    QFxp,
    QUInt,
    Register,
    Side,
    Signature,
)
from qualtran.bloqs.basic_gates import TGate, Toffoli
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.symbolics import smax

if TYPE_CHECKING:
    import quimb.tensor as qtn

    from qualtran import SoquetT
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class PlusEqualProduct(GateWithRegisters, cirq.ArithmeticGate):  # type: ignore[misc]
    """Performs result += a * b"""

    a_bitsize: int
    b_bitsize: int
    result_bitsize: int
    is_adjoint: bool = False

    def pretty_name(self) -> str:
        return "result -= a*b" if self.is_adjoint else "result += a*b"

    @property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(
            a=QUInt(self.a_bitsize),
            b=QUInt(self.b_bitsize),
            result=QFxp(self.result_bitsize, self.result_bitsize),
        )

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [2] * self.a_bitsize, [2] * self.b_bitsize, [2] * self.result_bitsize

    def adjoint(self) -> 'PlusEqualProduct':
        return evolve(self, is_adjoint=not self.is_adjoint)

    def apply(self, a: int, b: int, result: int) -> Union[int, Iterable[int]]:
        return a, b, (result + a * b * ((-1) ** self.is_adjoint)) % (2**self.result_bitsize)

    def with_registers(self, *new_registers: Union[int, Sequence[int]]):
        raise NotImplementedError("Not needed.")

    def on_classical_vals(self, a: int, b: int, result: int) -> Dict[str, 'ClassicalValT']:
        result_out = (result + a * b * ((-1) ** self.is_adjoint)) % (2**self.result_bitsize)
        return {'a': a, 'b': b, 'result': result_out}

    def _t_complexity_(self) -> 'TComplexity':
        # TODO: The T-complexity here is approximate.
        return TComplexity(t=8 * max(self.a_bitsize, self.b_bitsize) ** 2)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['a'] * self.a_bitsize + ['b'] * self.b_bitsize
        wire_symbols += ['c-=a*b' if self.is_adjoint else 'c+=a*b'] * self.result_bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return PlusEqualProduct(
                self.a_bitsize, self.b_bitsize, self.result_bitsize, not self.is_adjoint
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
            self, self.signature, self.pretty_name(), tn, tag, incoming=incoming, outgoing=outgoing
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(TGate(), 8 * smax(self.a_bitsize, self.b_bitsize) ** 2)}


@bloq_example
def _plus_equal_product() -> PlusEqualProduct:
    a_bit, b_bit, res_bit = 2, 2, 4
    plus_equal_product = PlusEqualProduct(a_bit, b_bit, res_bit)
    return plus_equal_product


_PLUS_EQUALS_PRODUCT_DOC = BloqDocSpec(bloq_cls=PlusEqualProduct, examples=[_plus_equal_product])


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
    uncompute: bool = False

    @property
    def signature(self):
        side = Side.LEFT if self.uncompute else Side.RIGHT
        return Signature(
            [
                Register("a", QUInt(self.bitsize)),
                Register("result", QUInt(2 * self.bitsize), side=side),
            ]
        )

    def on_classical_vals(self, **vals: int) -> Dict[str, 'ClassicalValT']:
        if self.uncompute:
            a, result = vals["a"], vals["result"]
            assert result == a**2
            return {'a': a}
        a = vals["a"]
        return {'a': a, 'result': a**2}

    def pretty_name(self) -> str:
        return "a^2"

    def _t_complexity_(self):
        # TODO Determine precise clifford count and/or ignore.
        # See: https://github.com/quantumlib/Qualtran/issues/219
        # See: https://github.com/quantumlib/Qualtran/issues/217
        num_toff = self.bitsize * (self.bitsize - 1)
        return TComplexity(t=4 * num_toff)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        num_toff = self.bitsize * (self.bitsize - 1)
        return {(Toffoli(), num_toff)}

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        import quimb.tensor as qtn

        N = 2**self.bitsize
        data = np.zeros((N, N, N**2), dtype=np.complex128)
        for x in range(N):
            data[x, x, x**2] = 1

        trg = incoming['result'] if self.uncompute else outgoing['result']
        tn.add(
            qtn.Tensor(data=data, inds=(incoming['a'], outgoing['a'], trg), tags=['Square', tag])
        )

    def adjoint(self) -> 'Bloq':
        return Square(self.bitsize, not self.uncompute)


@bloq_example
def _square() -> Square:
    square = Square(bitsize=8)
    return square


_SQUARE_DOC = BloqDocSpec(bloq_cls=Square, examples=[_square])


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
        Quantization](https://arxiv.org/abs/2105.12767) pg 80 gives a Toffoli
        complexity for squaring.
    """

    bitsize: int
    k: int

    @property
    def signature(self):
        return Signature(
            [
                Register("input", QUInt(bitsize=self.bitsize), shape=(self.k,)),
                Register(
                    "result",
                    QUInt(bitsize=2 * self.bitsize + (self.k - 1).bit_length()),
                    side=Side.RIGHT,
                ),
            ]
        )

    def pretty_name(self) -> str:
        return "SOS"

    def _t_complexity_(self):
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


@bloq_example
def _sum_of_squares() -> SumOfSquares:
    sum_of_squares = SumOfSquares(bitsize=8, k=4)
    return sum_of_squares


_SUM_OF_SQUARES_DOC = BloqDocSpec(bloq_cls=SumOfSquares, examples=[_sum_of_squares])


@frozen
class Product(Bloq):
    r"""Compute the product of an `n` and `m` bit binary number.

    Implements $U|a\rangle|b\rangle|0\rangle \rightarrow
    |a\rangle|b\rangle|a\times b\rangle$ using $2nm-n$ Toffolis.

    Args:
        a_bitsize: Number of bits used to represent the first integer.
        b_bitsize: Number of bits used to represent the second integer.

    Registers:
        a: a_bitsize-sized input register.
        b: b_bitsize-sized input register.
        result: A 2*`max(a_bitsize, b_bitsize)` bit-sized output register to store the result a*b.

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
                Register("a", QUInt(self.a_bitsize)),
                Register("b", QUInt(self.b_bitsize)),
                Register("result", QUInt(2 * max(self.a_bitsize, self.b_bitsize)), side=Side.RIGHT),
            ]
        )

    def pretty_name(self) -> str:
        return "a*b"

    def _t_complexity_(self):
        # TODO Determine precise clifford count and/or ignore.
        # See: https://github.com/quantumlib/Qualtran/issues/219
        # See: https://github.com/quantumlib/Qualtran/issues/217
        num_toff = 2 * self.a_bitsize * self.b_bitsize - max(self.a_bitsize, self.b_bitsize)
        return TComplexity(t=4 * num_toff)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        num_toff = 2 * self.a_bitsize * self.b_bitsize - max(self.a_bitsize, self.b_bitsize)
        return {(Toffoli(), num_toff)}


@bloq_example
def _product() -> Product:
    product = Product(a_bitsize=4, b_bitsize=6)
    return product


_PRODUCT_DOC = BloqDocSpec(bloq_cls=Product, examples=[_product])


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
        real_in: r_bitsize-sized input fixed-point register.
        int_in: i_bitsize-sized input register.
        result: a r_bitsize sized output fixed-point register.

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization](
            https://arxiv.org/pdf/2007.07391.pdf) pg 70.
    """

    r_bitsize: int
    i_bitsize: int

    @property
    def signature(self):
        return Signature(
            [
                Register("real_in", QFxp(self.r_bitsize, self.r_bitsize)),
                Register("int_in", QUInt(self.i_bitsize)),
                Register(
                    "result", QFxp(self.r_bitsize, self.r_bitsize - self.i_bitsize), side=Side.RIGHT
                ),
            ]
        )

    def pretty_name(self) -> str:
        return "r*i"

    def _t_complexity_(self):
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


@bloq_example
def _scale_int_by_real() -> ScaleIntByReal:
    scale_int_by_real = ScaleIntByReal(r_bitsize=12, i_bitsize=4)
    return scale_int_by_real


_SCALE_INT_BY_REAL_DOC = BloqDocSpec(bloq_cls=ScaleIntByReal, examples=[_scale_int_by_real])


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
        a: bitsize-sized input register.
        b: bitsize-sized input register.
        result: bitsize output register

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization](https://arxiv.org/abs/2007.07391).
        Appendix D. Section 5. (p. 71).
    """

    bitsize: int

    @property
    def signature(self):
        return Signature(
            [
                Register("a", QFxp(self.bitsize, self.bitsize)),
                Register("b", QFxp(self.bitsize, self.bitsize)),
                Register("result", QFxp(self.bitsize, self.bitsize), side=Side.RIGHT),
            ]
        )

    def pretty_name(self) -> str:
        return "a*b"

    def _t_complexity_(self):
        # Eq. D13, there it is suggested keeping both registers the same size is optimal.
        num_toff = self.bitsize**2 - self.bitsize - 1
        return TComplexity(t=4 * num_toff)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Eq. D13, there it is suggested keeping both registers the same size is optimal.
        num_toff = self.bitsize**2 - self.bitsize - 1
        return {(Toffoli(), num_toff)}


@bloq_example
def _multiply_two_reals() -> MultiplyTwoReals:
    multiply_two_reals = MultiplyTwoReals(bitsize=10)
    return multiply_two_reals


_MULTIPLY_TWO_REALS_DOC = BloqDocSpec(bloq_cls=MultiplyTwoReals, examples=[_multiply_two_reals])


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
        a: bitsize-sized input register.
        b: bitsize-sized input register.
        result: bitsize output register

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization](https://arxiv.org/abs/2007.07391).
        Appendix D. Section 6. (p. 74).
    """

    bitsize: int

    def __attrs_post_init__(self):
        if self.bitsize < 3:
            raise ValueError("bitsize must be at least 3 for SquareRealNumber bloq to make sense.")

    @property
    def signature(self):
        return Signature(
            [
                Register("a", QFxp(self.bitsize, self.bitsize)),
                Register("b", QFxp(self.bitsize, self.bitsize)),
                Register("result", QFxp(self.bitsize, self.bitsize), side=Side.RIGHT),
            ]
        )

    def pretty_name(self) -> str:
        return "a^2"

    def _t_complexity_(self):
        num_toff = self.bitsize**2 // 2 - 4
        return TComplexity(t=4 * num_toff)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Bottom of page 74
        num_toff = self.bitsize**2 // 2 - 4
        return {(Toffoli(), num_toff)}


@bloq_example
def _square_real_number() -> SquareRealNumber:
    square_real_number = SquareRealNumber(bitsize=10)
    return square_real_number


_SQUARE_REAL_NUMBER_DOC = BloqDocSpec(bloq_cls=SquareRealNumber, examples=[_square_real_number])
