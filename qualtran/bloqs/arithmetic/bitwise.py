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
from typing import TYPE_CHECKING

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
    QUInt,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.symbolics import is_symbolic, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


def _cvs_converter(vv):
    if isinstance(vv, (int, np.integer)):
        return (int(vv),)
    return tuple(int(v) for v in vv)


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

    def build_composite_bloq(self, bb: 'BloqBuilder', x: 'Soquet') -> dict[str, 'SoquetT']:
        if self.is_symbolic():
            raise DecomposeTypeError(f"cannot decompose symbolic {self}")

        xs = bb.split(x)

        for i, bit in enumerate(self.dtype.to_bits(self.k)):
            if bit == 1:
                xs[i] = bb.add(XGate(), q=xs[i])

        x = bb.join(xs, dtype=self.dtype)

        return {'x': x}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> set['BloqCountT']:
        num_flips = self.bitsize if self.is_symbolic() else sum(self.dtype.to_bits(self.k))
        return {(XGate(), num_flips)}


@bloq_example(generalizer=ignore_split_join)
def _xork() -> XorK:
    xork = XorK(QUInt(8), 0b01010111)
    return xork


@frozen
class Xor(Bloq):
    """Xor the value of one register into another via CNOTs.

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

    def build_composite_bloq(self, bb: BloqBuilder, x: Soquet, y: Soquet) -> dict[str, SoquetT]:
        if not isinstance(self.dtype.num_qubits, int):
            raise DecomposeTypeError("`dtype.num_qubits` must be a concrete value.")

        xs = bb.split(x)
        ys = bb.split(y)

        for i in range(len(xs)):
            xs[i], ys[i] = bb.add_t(CNOT(), ctrl=xs[i], target=ys[i])

        return {'x': bb.join(xs, dtype=self.dtype), 'y': bb.join(ys, dtype=self.dtype)}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> set['BloqCountT']:
        return {(CNOT(), self.dtype.num_qubits)}

    def on_classical_vals(
        self, x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> dict[str, 'ClassicalValT']:
        return {'x': x, 'y': x ^ y}


@bloq_example
def _xor() -> Xor:
    xor = Xor(QAny(4))
    return xor


@bloq_example
def _xor_symb() -> Xor:
    xor_symb = Xor(QAny(sympy.Symbol("n")))
    return xor_symb


_XOR_DOC = BloqDocSpec(
    bloq_cls=Xor,
    import_line='from qualtran.bloqs.arithmetic import Xor',
    examples=(_xor, _xor_symb),
)
