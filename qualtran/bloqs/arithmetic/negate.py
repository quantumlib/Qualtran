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

from attrs import frozen

from qualtran import Bloq, bloq_example, BloqDocSpec, QDType, QInt, Signature
from qualtran.bloqs.arithmetic import AddK
from qualtran.bloqs.arithmetic.bitwise import BitwiseNot

if TYPE_CHECKING:
    from qualtran import BloqBuilder, SoquetT


@frozen
class Negate(Bloq):
    """Compute the two's complement negation for a integer/fixed-point value.

    This bloq is equivalent to the "Unary minus" [1] C++ operator.
    - For a signed `x`, the output is `-x`.
    - For an unsigned `x`, the output is `2^n - x` (where `n` is the bitsize).

    This is computed by the bit-fiddling trick `-x = ~x + 1`, as follows:
    1. Flip all the bits (i.e. `x := ~x`)
    2. Add 1 to the value (interpreted as an unsigned integer), ignoring
       any overflow. (i.e. `x := x + 1`)

    For a controlled negate bloq: the second step uses a quantum-quantum adder by
    loading the constant (i.e. 1), therefore has an improved controlled version
    which only controls the constant load and not the adder circuit, hence halving
    the T-cost compared to a controlled adder.

    Args:
        dtype: The data type of the input value.

    Registers:
        x: Any unsigned value or signed value (in two's complement form).

    References:
        [Arithmetic Operators - cppreference](https://en.cppreference.com/w/cpp/language/operator_arithmetic)
        Operator "Unary Minus". Last accessed 17 July 2024.
    """

    dtype: QDType

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(x=self.dtype)

    def build_composite_bloq(self, bb: 'BloqBuilder', x: 'SoquetT') -> dict[str, 'SoquetT']:
        x = bb.add(BitwiseNot(self.dtype), x=x)  # ~x
        x = bb.add(AddK(self.dtype.num_qubits, k=1, signed=isinstance(self.dtype, QInt)), x=x)  # -x
        return {'x': x}


@bloq_example
def _negate() -> Negate:
    negate = Negate(QInt(8))
    return negate


@bloq_example
def _negate_symb() -> Negate:
    import sympy

    n = sympy.Symbol("n")
    negate_symb = Negate(QInt(n))
    return negate_symb


_NEGATE_DOC = BloqDocSpec(bloq_cls=Negate, examples=[_negate, _negate_symb])
