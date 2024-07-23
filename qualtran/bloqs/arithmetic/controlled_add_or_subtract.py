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
from typing import TYPE_CHECKING, Union

from attrs import field, frozen

from qualtran import Bloq, bloq_example, BloqDocSpec, QBit, QInt, QMontgomeryUInt, QUInt, Signature
from qualtran.bloqs.arithmetic.addition import Add
from qualtran.bloqs.arithmetic.bitwise import BitwiseNot
from qualtran.bloqs.basic_gates import XGate

if TYPE_CHECKING:
    from qualtran import BloqBuilder, Soquet, SoquetT


@frozen
class ControlledAddOrSubtract(Bloq):
    r"""Adds or subtracts in-place into the target, based on a control bit.

    Applies the transformation

    $$
        |1\rangle |a\rangle |b\rangle \mapsto |1\rangle |a\rangle |b + a\rangle \\
        |0\rangle |a\rangle |b\rangle \mapsto |0\rangle |a\rangle |b - a\rangle
    $$

    Given two numbers `a`, `b` and a control bit `ctrl`, this bloq computes:

    - the sum `b + a` when `ctrl=1`,
    - the difference `b - a` when `ctrl=0`,

    and stores the result in the second register (`b`).

    This uses an uncontrolled `Add` surrounded by controlled `BitwiseNot`s, and only
    the `Add` requires T gates, which has half the T-cost of a controlled `Add`.


    Args:
        a_dtype: dtype of the lhs `a`
        b_dtype: dtype of the rhs `b`. If it is not big enough to store the
                 result, the most significant bits are dropped on overflow.
        add_when_ctrl_is_on: If True (default), add when `ctrl=1` and subtract when
                             `ctrl=0`. If False, do the opposite: subtract when `ctrl=0`
                             and add when `ctrl=1`.

    Registers:
        ctrl: a single control bit
        a: an integer value.
        b: an integer value.

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization](https://arxiv.org/abs/2007.07391).
        Sanders et. al. Section II-A-1, Algorithm 1.
    """

    a_dtype: Union[QInt, QUInt, QMontgomeryUInt] = field()
    b_dtype: Union[QInt, QUInt, QMontgomeryUInt] = field()
    add_when_ctrl_is_on = True

    @b_dtype.default
    def b_dtype_default(self):
        return self.a_dtype

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(ctrl=QBit(), a=self.a_dtype, b=self.b_dtype)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrl: 'Soquet', a: 'Soquet', b: 'Soquet'
    ) -> dict[str, 'SoquetT']:
        if self.add_when_ctrl_is_on:
            # flip the control bit
            ctrl = bb.add(XGate(), q=ctrl)

        # subcircuit to add when ctrl=0 and subtract when ctrl=1.
        # (0, a, b) or (1, a, b)
        ctrl, b = bb.add(BitwiseNot(self.b_dtype).controlled(), ctrl=ctrl, x=b)
        # -> (0, a, b) or (1, a, -1 - b)
        a, b = bb.add(Add(self.a_dtype, self.b_dtype), a=a, b=b)
        # -> (0, a, b + a) or (1, a, -1 - b + a)
        ctrl, b = bb.add(BitwiseNot(self.b_dtype).controlled(), ctrl=ctrl, x=b)
        # -> (0, a, b + a) or (1, a, b - a)

        if self.add_when_ctrl_is_on:
            ctrl = bb.add(XGate(), q=ctrl)

        return {'ctrl': ctrl, 'a': a, 'b': b}


@bloq_example
def _ctrl_add_or_sub_unsigned() -> ControlledAddOrSubtract:
    ctrl_add_or_sub_unsigned = ControlledAddOrSubtract(QUInt(8), QUInt(8))
    return ctrl_add_or_sub_unsigned


@bloq_example
def _ctrl_add_or_sub_signed() -> ControlledAddOrSubtract:
    ctrl_add_or_sub_signed = ControlledAddOrSubtract(QInt(8), QInt(8))
    return ctrl_add_or_sub_signed


@bloq_example
def _ctrl_add_or_sub_signed_symb() -> ControlledAddOrSubtract:
    import sympy

    n = sympy.Symbol("n")
    ctrl_add_or_sub_signed_symb = ControlledAddOrSubtract(QInt(n), QInt(n))
    return ctrl_add_or_sub_signed_symb


_CONTROLLED_ADD_OR_SUBTRACT_DOC = BloqDocSpec(
    bloq_cls=ControlledAddOrSubtract,
    import_line='from qualtran.bloqs.arithmetic.controlled_add_or_subtract import ControlledAddOrSubtract',
    examples=[_ctrl_add_or_sub_signed_symb, _ctrl_add_or_sub_unsigned, _ctrl_add_or_sub_signed],
)
