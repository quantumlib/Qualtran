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
    QInt,
    QMontgomeryUInt,
    QUInt,
    QVar,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.basic_gates import CNOT, OnEach, XGate
from qualtran.drawing import Text, TextBox, WireSymbol
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.symbolics import is_symbolic, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT
    from qualtran.simulation.verification import ClassicalSimTestCase


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
        if self.is_symbolic():
            raise ValueError(f"cannot classically simulate with symbolic {self}")
        assert isinstance(self.k, int)
        k: 'ClassicalValT' = self.k
        if isinstance(x, np.integer):
            k = np.array(k, dtype=x.dtype)[()]
        return {'x': x ^ k}

    def wire_symbol(
        self, reg: Optional['Register'], idx: tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text("")

        return TextBox(f"⊕{self.k}")

    def __str__(self):
        return f"XorK({self.k})"


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

    @classmethod
    def qcall(cls, x: 'QVar', y: 'QVar'):
        xdtype = x.dtype
        ydtype = y.dtype
        if not xdtype == ydtype:
            raise ValueError(
                f"Cannot determine the dtype for Xor from soquets of type {xdtype} and {ydtype}"
            )
        assert isinstance(xdtype, QDType), xdtype
        return x.bb.add(cls(dtype=xdtype), x=x, y=y)

    def build_composite_bloq(self, bb: BloqBuilder, x: Soquet, y: Soquet) -> dict[str, SoquetT]:
        if not isinstance(self.dtype.num_qubits, int):
            raise DecomposeTypeError("`dtype.num_qubits` must be a concrete value.")

        xs = bb.split(x)
        ys = bb.split(y)

        for i in range(len(xs)):
            xs[i], ys[i] = bb.add_t(CNOT(), ctrl=xs[i], target=ys[i])  # type: ignore[assignment]

        return {'x': bb.join(xs, dtype=self.dtype), 'y': bb.join(ys, dtype=self.dtype)}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {CNOT(): self.dtype.num_qubits}

    def adjoint(self) -> 'Xor':
        return self

    def on_classical_vals(
        self, x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> dict[str, 'ClassicalValT']:
        if is_symbolic(self.dtype):
            raise ValueError(f"cannot classically simulate with symbolic {self.dtype=}")
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

    @classmethod
    def qcall(cls, x: 'QVar') -> 'QVar':
        return x.bb.add(cls(dtype=x.dtype), x=x)  # type: ignore[arg-type]

    def adjoint(self) -> 'BitwiseNot':
        return self

    def build_composite_bloq(self, bb: 'BloqBuilder', x: 'Soquet') -> dict[str, 'SoquetT']:
        if is_symbolic(self.dtype):
            raise DecomposeTypeError(f"cannot decompose symbolic {self}")
        x = bb.add(OnEach(self.dtype.num_qubits, XGate(), self.dtype), q=x)
        return {'x': x}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {XGate(): self.dtype.num_qubits}

    def wire_symbol(
        self, reg: Optional['Register'], idx: tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text("")

        return TextBox("~x")

    def on_classical_vals(self, x: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        if is_symbolic(self.dtype):
            raise ValueError(f"cannot classically simulate with symbolic {self.dtype=}")
        if isinstance(self.dtype, QInt):
            return {'x': ~x}
        return {'x': ~x % (2**self.dtype.num_qubits)}

    def __str__(self):
        return f'BitwiseNot({self.dtype.num_bits})'


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


def _get_xork_classical_sim_test_cases() -> list['ClassicalSimTestCase']:
    """Test cases for the `XorK` bloq."""
    import itertools

    from qualtran.simulation.verification import ClassicalSimTestCase

    cases: list[ClassicalSimTestCase] = []
    for bs, k in itertools.product([2, 3, 4], [0, 1, 3]):
        cases.append(
            ClassicalSimTestCase(bloq=XorK(QUInt(bs), k=k), name=f"XorK(QUInt({bs}), k={k})")
        )
    for bs, k in itertools.product([3, 4], [-2, 0, 1]):
        cases.append(
            ClassicalSimTestCase(bloq=XorK(QInt(bs), k=k), name=f"XorK(QInt({bs}), k={k})")
        )
    for bs, k in [(4, 5)]:
        cases.append(
            ClassicalSimTestCase(
                bloq=XorK(QMontgomeryUInt(bs), k=k), name=f"XorK(QMontgomeryUInt({bs}), k={k})"
            )
        )
    return cases


def _get_xor_classical_sim_test_cases() -> list['ClassicalSimTestCase']:
    """Test cases for the `Xor` bloq."""
    from qualtran.simulation.verification import ClassicalSimTestCase

    cases: list[ClassicalSimTestCase] = []
    for bs in [2, 3, 4]:
        cases.append(ClassicalSimTestCase(bloq=Xor(QUInt(bs)), name=f"Xor(QUInt({bs}))"))
        cases.append(ClassicalSimTestCase(bloq=Xor(QInt(bs)), name=f"Xor(QInt({bs}))"))
    cases.append(ClassicalSimTestCase(bloq=Xor(QMontgomeryUInt(4)), name="Xor(QMontgomeryUInt(4))"))
    return cases


def _get_bitwise_not_classical_sim_test_cases() -> list['ClassicalSimTestCase']:
    """Test cases for the `BitwiseNot` bloq."""
    from qualtran.simulation.verification import ClassicalSimTestCase

    cases: list[ClassicalSimTestCase] = []
    for bs in [1, 2, 3, 4]:
        cases.append(
            ClassicalSimTestCase(bloq=BitwiseNot(QUInt(bs)), name=f"BitwiseNot(QUInt({bs}))")
        )
    for bs in [2, 3, 4]:
        cases.append(
            ClassicalSimTestCase(bloq=BitwiseNot(QInt(bs)), name=f"BitwiseNot(QInt({bs}))")
        )
    cases.append(
        ClassicalSimTestCase(
            bloq=BitwiseNot(QMontgomeryUInt(4)), name="BitwiseNot(QMontgomeryUInt(4))"
        )
    )
    return cases
