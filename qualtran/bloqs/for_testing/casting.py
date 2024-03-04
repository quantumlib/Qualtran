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

from functools import cached_property
from typing import Dict

from attrs import frozen

from qualtran import Bloq, BloqBuilder, QFxp, QUInt, Register, Signature, Soquet
from qualtran.bloqs.arithmetic.addition import Add
from qualtran.bloqs.util_bloqs import Cast


@frozen
class TestCastToFrom(Bloq):
    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('a', QUInt(4)), Register('b', QFxp(4, 4))])

    def build_composite_bloq(
        self, bb: 'BloqBuilder', *, a: 'Soquet', b: 'Soquet'
    ) -> Dict[str, 'Soquet']:
        cast = Cast(b.reg.dtype, a.reg.dtype)
        b = bb.add(cast, reg=b)
        a, b = bb.add(Add(a.reg.dtype), a=a, b=b)
        b = bb.add(cast.adjoint(), reg=b)
        return {'a': a, 'b': b}
