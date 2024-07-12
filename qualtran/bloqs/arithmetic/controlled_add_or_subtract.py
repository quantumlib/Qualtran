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

from qualtran import Bloq, CtrlSpec, QBit, QInt, QMontgomeryUInt, QUInt, Signature
from qualtran.bloqs.arithmetic import Add, Negate

if TYPE_CHECKING:
    from qualtran import BloqBuilder, Soquet, SoquetT


@frozen
class ControlledAddOrSubtract(Bloq):
    """Transforms |1>|a>|b> to |1>|a>|a + b> and |0>|a>|b> to |0>|a>|a - b>

    Given two numbers `a`, `b` and a control bit `ctrl`, this bloq computes:
    - the sum `a + b` when `ctrl=1`,
    - the difference `a - b` when `ctrl=0`,
    and stores it in the second register (`b`).

    This uses a controlled `Negate` followed by an uncontrolled `Add`,
    which has half the T-cost of a controlled `Add`.

    Registers:
        ctrl: a single control bit
        a: an integer value.
        b: an integer value. If it is not big enough to store the result,
           the most significant bits are dropped.
    """

    a_dtype: Union[QInt, QUInt, QMontgomeryUInt] = field()
    b_dtype: Union[QInt, QUInt, QMontgomeryUInt] = field()

    @b_dtype.default
    def b_dtype_default(self):
        return self.a_dtype

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(ctrl=QBit(), a=self.a_dtype, b=self.b_dtype)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', ctrl: 'Soquet', a: 'Soquet', b: 'Soquet'
    ) -> dict[str, 'SoquetT']:
        ctrl, b = bb.add(Negate(self.b_dtype).controlled(CtrlSpec(cvs=0)), ctrl=ctrl, x=b)
        a, b = bb.add(Add(self.a_dtype, self.b_dtype), a=a, b=b)
        return {'ctrl': ctrl, 'a': a, 'b': b}
