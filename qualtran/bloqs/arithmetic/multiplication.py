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

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import cirq
import numpy as np
from attrs import evolve, frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    ConnectionT,
    DecomposeTypeError,
    GateWithRegisters,
    QFxp,
    QUInt,
    Register,
    Side,
    Signature,
)
from qualtran.bloqs.arithmetic.subtraction import Subtract
from qualtran.bloqs.basic_gates import CNOT, TGate, Toffoli, XGate
from qualtran.bloqs.mcmt import MultiControlX
from qualtran.drawing import Text, WireSymbol
from qualtran.symbolics import ceil, HasLength, is_symbolic, log2, smax, SymbolicInt

if TYPE_CHECKING:
    import quimb.tensor as qtn

    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class PlusEqualProduct(GateWithRegisters, cirq.ArithmeticGate):  # type: ignore[misc]
    """Performs result += a * b.

    Args:
        a_bitsize: bitsize of input `a`.
        b_bitsize: bitsize of input `b`.
        result_bitsize: bitsize of the output register.
        is_adjoint: If true, performs `result -= a * b` instead. Defaults to False.

    Registers:
        a: QUInt of `a_bitsize` bits.
        b: QUInt of `b_bitsize` bits.
        result: QUInt of `result_bitsize` bits.
    """

    a_bitsize: SymbolicInt
    b_bitsize: SymbolicInt
    result_bitsize: SymbolicInt
    is_adjoint: bool = False

    def __attrs_post_init__(self):
        res_has_enough = self.a_bitsize + self.b_bitsize <= self.result_bitsize
        if not is_symbolic(res_has_enough) and not res_has_enough:
            raise ValueError(
                f"{self.result_bitsize=} must be at least the sum of input "
                f"bitsizes {self.a_bitsize} + {self.b_bitsize}"
            )

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text("result -= a*b") if self.is_adjoint else Text("result += a*b")
        return super().wire_symbol(reg, idx)

    @property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(a=self.a_dtype, b=self.b_dtype, result=self.result_dtype)

    @property
    def a_dtype(self):
        return QUInt(self.a_bitsize)

    @property
    def b_dtype(self):
        return QUInt(self.b_bitsize)

    @property
    def result_dtype(self):
        return QUInt(self.result_bitsize)

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        if is_symbolic(self.a_bitsize):
            raise ValueError(f'Symbolic bitsize {self.a_bitsize} not supported')
        if is_symbolic(self.b_bitsize):
            raise ValueError(f'Symbolic bitsize {self.b_bitsize} not supported')
        if is_symbolic(self.result_bitsize):
            raise ValueError(f'Symbolic bitsize {self.result_bitsize} not supported')
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

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        if is_symbolic(self.a_bitsize):
            raise ValueError(f'Symbolic bitsize {self.a_bitsize} not supported')
        if is_symbolic(self.b_bitsize):
            raise ValueError(f'Symbolic bitsize {self.b_bitsize} not supported')
        if is_symbolic(self.result_bitsize):
            raise ValueError(f'Symbolic bitsize {self.result_bitsize} not supported')
        wire_symbols = ['a'] * self.a_bitsize + ['b'] * self.b_bitsize
        wire_symbols += ['c-=a*b' if self.is_adjoint else 'c+=a*b'] * self.result_bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        from qualtran.cirq_interop._cirq_to_bloq import _my_tensors_from_gate

        return _my_tensors_from_gate(self, self.signature, incoming=incoming, outgoing=outgoing)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # TODO: The T-complexity here is approximate.
        return {TGate(): 8 * smax(self.a_bitsize, self.b_bitsize) ** 2}


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

    bitsize: SymbolicInt
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

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text("a^2")
        return super().wire_symbol(reg, idx)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # TODO Determine precise clifford count and/or ignore.
        # See: https://github.com/quantumlib/Qualtran/issues/219
        # See: https://github.com/quantumlib/Qualtran/issues/217
        num_toff = self.bitsize * (self.bitsize - 1)
        return {Toffoli(): num_toff}

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        if is_symbolic(self.bitsize):
            raise DecomposeTypeError(f"Cannot get tensors for symbolic {self=}")

        n = self.bitsize
        N = 2**self.bitsize
        data = np.zeros((N, N, N**2), dtype=np.complex128)
        for x in range(N):
            data[x, x, x**2] = 1

        trg = incoming['result'] if self.uncompute else outgoing['result']
        inds = (
            [(incoming['a'], j) for j in range(n)]
            + [(outgoing['a'], j) for j in range(n)]
            + [(trg, j) for j in range(2 * n)]
        )
        data = data.reshape((2,) * (4 * n))
        return [qtn.Tensor(data=data, inds=inds, tags=[str(self)])]

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

    bitsize: SymbolicInt
    k: SymbolicInt

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

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('SOS')
        return super().wire_symbol(reg, idx)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        num_toff = self.k * self.bitsize**2 - self.bitsize
        if self.k % 3 == 0:
            num_toff -= 1
        return {Toffoli(): num_toff}


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

    a_bitsize: SymbolicInt
    b_bitsize: SymbolicInt

    @property
    def signature(self):
        return Signature(
            [
                Register("a", QUInt(self.a_bitsize)),
                Register("b", QUInt(self.b_bitsize)),
                Register("result", QUInt(2 * max(self.a_bitsize, self.b_bitsize)), side=Side.RIGHT),
            ]
        )

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('a*b')
        return super().wire_symbol(reg, idx)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # TODO Determine precise clifford count and/or ignore.
        # See: https://github.com/quantumlib/Qualtran/issues/219
        # See: https://github.com/quantumlib/Qualtran/issues/217
        num_toff = 2 * self.a_bitsize * self.b_bitsize - max(self.a_bitsize, self.b_bitsize)
        return {Toffoli(): num_toff}


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

    r_bitsize: SymbolicInt
    i_bitsize: SymbolicInt

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

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('r*i')
        return super().wire_symbol(reg, idx)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # Eq. D8, we are assuming dA(r_bitsize) and dB(i_bitsize) are inputs and
        # the user has ensured these are large enough for their desired
        # precision.
        num_toff = self.r_bitsize * (2 * self.i_bitsize - 1) - self.i_bitsize**2
        return {Toffoli(): num_toff}


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

    bitsize: SymbolicInt

    @property
    def signature(self):
        return Signature(
            [
                Register("a", QFxp(self.bitsize, self.bitsize)),
                Register("b", QFxp(self.bitsize, self.bitsize)),
                Register("result", QFxp(self.bitsize, self.bitsize), side=Side.RIGHT),
            ]
        )

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('a*b')
        return super().wire_symbol(reg, idx)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # Eq. D13, there it is suggested keeping both registers the same size is optimal.
        num_toff = self.bitsize**2 - self.bitsize - 1
        return {Toffoli(): num_toff}


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

    bitsize: SymbolicInt

    def __attrs_post_init__(self):
        if not is_symbolic(self.bitsize) and self.bitsize < 3:
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

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('a^2')
        return super().wire_symbol(reg, idx)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # Bottom of page 74
        num_toff = self.bitsize**2 // 2 - 4
        return {Toffoli(): num_toff}


@bloq_example
def _square_real_number() -> SquareRealNumber:
    square_real_number = SquareRealNumber(bitsize=10)
    return square_real_number


_SQUARE_REAL_NUMBER_DOC = BloqDocSpec(bloq_cls=SquareRealNumber, examples=[_square_real_number])


@frozen
class InvertRealNumber(Bloq):
    r"""Invert a fixed-point representation of a real number.

    Implements the unitary:
    $$
        |a\rangle|0\rangle \rightarrow |a\rangle|1/a\rangle
    $$
    where $a \ge 1$.

    Args:
        bitsize: Number of bits used to represent the number.
        num_frac: Number of fraction bits in the number.

    Registers:
        a: `bitsize`-sized input register.
        result: `bitsize`-sized output register.

        References:
        [Quantum Algorithms and Circuits for Scientific Computing](https://arxiv.org/pdf/1511.08253). Section 2.1.
    """

    bitsize: SymbolicInt
    num_frac: SymbolicInt

    def __attrs_post_init__(self):
        if self.num_frac == self.bitsize:
            raise ValueError("num_frac must be < bitsize since a >= 1.")

    @property
    def signature(self):
        return Signature(
            [
                Register("a", QFxp(self.bitsize, self.num_frac)),
                Register("result", QFxp(self.bitsize, self.num_frac)),
            ]
        )

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('1/a')
        return super().wire_symbol(reg, idx)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # initial approximation: Figure 4
        num_int = self.bitsize - self.num_frac
        # Newton-Raphson: Eq. (1)
        # x' = -a * x^2 + 2 * x
        num_iters = ceil(log2(self.bitsize))
        # TODO: When decomposing we will potentially need to use larger registers.
        # Related issue: https://github.com/quantumlib/Qualtran/issues/655
        return {
            Toffoli(): num_int - 1,
            CNOT(): 2 + num_int - 1,
            MultiControlX(cvs=HasLength(num_int)): 1,
            XGate(): 1,
            SquareRealNumber(self.bitsize): num_iters,  # x^2
            MultiplyTwoReals(self.bitsize): num_iters,  # a * x^2
            ScaleIntByReal(self.bitsize, 2): num_iters,  # 2 * x
            Subtract(QUInt(self.bitsize)): num_iters,  # 2 * x - a * x^2
        }


@bloq_example
def _invert_real_number() -> InvertRealNumber:
    invert_real_number = InvertRealNumber(bitsize=10, num_frac=7)
    return invert_real_number


_INVERT_REAL_NUMBER_DOC = BloqDocSpec(bloq_cls=InvertRealNumber, examples=[_invert_real_number])
