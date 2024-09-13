#  Copyright 2024 Google LLC
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
from functools import cached_property
from typing import Dict, Optional, Sequence, TYPE_CHECKING

import numpy as np
import sympy
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    DecomposeTypeError,
    QAny,
    QDType,
    QMontgomeryUInt,
    QUInt,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.basic_gates import CNOT, OnEach, XGate
from qualtran.drawing import TextBox, WireSymbol
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.symbolics import is_symbolic, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class XorK(Bloq):
    r"""Maps |x> to |x \oplus k> for a constant k.

    Args:
        dtype: Data type of the input register `x`.
        k: The classical integer value to be XOR-ed to x.

    Registers:
        x: A quantum register of type `self.dtype` (see above).
    """

    dtype: QDType
    k: SymbolicInt

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(x=self.dtype)

    @cached_property
    def bitsize(self) -> SymbolicInt:
        return self.dtype.num_qubits

    def is_symbolic(self):
        return is_symbolic(self.k, self.dtype)

    def adjoint(self) -> 'XorK':
        return self

    @cached_property
    def _bits_k(self) -> Sequence[int]:
        return self.dtype.to_bits(self.k)

    def build_composite_bloq(self, bb: 'BloqBuilder', x: 'Soquet') -> dict[str, 'SoquetT']:
        if self.is_symbolic():
            raise DecomposeTypeError(f"cannot decompose symbolic {self}")

        xs = bb.split(x)

        for i, bit in enumerate(self._bits_k):
            if bit == 1:
                xs[i] = bb.add(XGate(), q=xs[i])

        x = bb.join(xs, dtype=self.dtype)

        return {'x': x}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        num_flips = self.bitsize if self.is_symbolic() else sum(self._bits_k)
        return {XGate(): num_flips}

    def on_classical_vals(self, x: 'ClassicalValT') -> dict[str, 'ClassicalValT']:
        if isinstance(self.k, sympy.Expr):
            raise ValueError(f"cannot classically simulate with symbolic {self.k=}")

        k: 'ClassicalValT' = self.k
        if isinstance(x, np.integer):
            k = np.array(k, dtype=x.dtype)[()]
        return {'x': x ^ k}

    def wire_symbol(
        self, reg: Optional['Register'], idx: tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return TextBox("")

        return TextBox(f"⊕{self.k}")


@bloq_example(generalizer=ignore_split_join)
def _xork() -> XorK:
    xork = XorK(QUInt(8), 0b01010111)
    return xork


@frozen
class Xor(Bloq):
    r"""Xor the value of one register into another via CNOTs.

    Maps basis states $|x, y\rangle$ to $|x, y \oplus x\rangle$.

    When both registers are in computational basis and the destination is 0,
    effectively copies the value of the source into the destination.

    Args:
        dtype: Data type of the input registers `x` and `y`.

    Registers:
        x: The source register.
        y: The target register.
    """

    dtype: QDType

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(x=self.dtype, y=self.dtype)

    def adjoint(self) -> 'Xor':
        return self

    def build_composite_bloq(self, bb: BloqBuilder, x: Soquet, y: Soquet) -> dict[str, SoquetT]:
        if not isinstance(self.dtype.num_qubits, int):
            raise DecomposeTypeError("`dtype.num_qubits` must be a concrete value.")

        xs = bb.split(x)
        ys = bb.split(y)

        for i in range(len(xs)):
            xs[i], ys[i] = bb.add_t(CNOT(), ctrl=xs[i], target=ys[i])

        return {'x': bb.join(xs, dtype=self.dtype), 'y': bb.join(ys, dtype=self.dtype)}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {CNOT(): self.dtype.num_qubits}

    def on_classical_vals(
        self, x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> dict[str, 'ClassicalValT']:
        return {'x': x, 'y': x ^ y}

    def wire_symbol(
        self, reg: Optional['Register'], idx: tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return TextBox('')
        elif reg.name == 'x':
            return TextBox('x')
        else:  # y
            return TextBox('x⊕y')


@bloq_example
def _xor() -> Xor:
    xor = Xor(QAny(4))
    return xor


@bloq_example
def _xor_symb() -> Xor:
    xor_symb = Xor(QAny(sympy.Symbol("n")))
    return xor_symb


_XOR_DOC = BloqDocSpec(bloq_cls=Xor, examples=(_xor, _xor_symb))


@frozen
class BitwiseNot(Bloq):
    r"""Flips every bit of the input register.

    Args:
        dtype: Data type of the input register `x`.

    Registers:
        x: A quantum register of type `self.dtype`.
    """

    dtype: QDType

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(x=self.dtype)

    def adjoint(self) -> 'BitwiseNot':
        return self

    def build_composite_bloq(self, bb: 'BloqBuilder', x: 'Soquet') -> dict[str, 'SoquetT']:
        x = bb.add(OnEach(self.dtype.num_qubits, XGate(), self.dtype), q=x)
        return {'x': x}

    def wire_symbol(
        self, reg: Optional['Register'], idx: tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return TextBox("")

        return TextBox("~x")

    def on_classical_vals(self, x: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        x = -x - 1
        if isinstance(self.dtype, (QUInt, QMontgomeryUInt)):
            x %= 2**self.dtype.bitsize
        return {'x': x}


@bloq_example
def _bitwise_not() -> BitwiseNot:
    bitwise_not = BitwiseNot(QUInt(4))
    return bitwise_not


@bloq_example
def _bitwise_not_symb() -> BitwiseNot:
    n = sympy.Symbol("n")
    bitwise_not_symb = BitwiseNot(QUInt(n))
    return bitwise_not_symb


_BITWISE_NOT_DOC = BloqDocSpec(bloq_cls=BitwiseNot, examples=(_bitwise_not, _bitwise_not_symb))
