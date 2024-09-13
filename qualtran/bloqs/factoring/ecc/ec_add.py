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

import sympy
from attrs import frozen

from qualtran import Bloq, bloq_example, BloqDocSpec, QUInt, Register, Signature
from qualtran.bloqs.arithmetic._shims import MultiCToffoli
from qualtran.bloqs.mod_arithmetic import CModAdd, CModNeg, CModSub, ModAdd, ModNeg, ModSub
from qualtran.bloqs.mod_arithmetic._shims import ModDbl, ModInv, ModMul
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@frozen
class ECAdd(Bloq):
    r"""Add two elliptic curve points.

    This takes elliptic curve points given by (a, b) and (x, y)
    and outputs the sum (x_r, y_r) in the second pair of registers.

    Args:
        n: The bitsize of the two registers storing the elliptic curve point
        mod: The modulus of the field in which we do the addition.

    Registers:
        a: The x component of the first input elliptic curve point of bitsize `n`.
        b: The y component of the first input elliptic curve point of bitsize `n`.
        x: The x component of the second input elliptic curve point of bitsize `n`, which
           will contain the x component of the resultant curve point.
        y: The y component of the second input elliptic curve point of bitsize `n`, which
           will contain the y component of the resultant curve point.
        lam: The precomputed lambda slope used in the addition operation.

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585).
        Litinski. 2023. Fig 5.
    """

    n: int
    mod: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('a', QUInt(self.n)),
                Register('b', QUInt(self.n)),
                Register('x', QUInt(self.n)),
                Register('y', QUInt(self.n)),
                Register('lam', QUInt(self.n)),
            ]
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # litinksi
        return {
            MultiCToffoli(n=self.n): 18,
            ModAdd(bitsize=self.n, mod=self.mod): 3,
            CModAdd(QUInt(self.n), mod=self.mod): 2,
            ModSub(QUInt(self.n), mod=self.mod): 2,
            CModSub(QUInt(self.n), mod=self.mod): 4,
            ModNeg(QUInt(self.n), mod=self.mod): 2,
            CModNeg(QUInt(self.n), mod=self.mod): 1,
            ModDbl(QUInt(self.n), mod=self.mod): 2,
            ModMul(n=self.n, mod=self.mod): 10,
            ModInv(n=self.n, mod=self.mod): 4,
        }


@bloq_example
def _ec_add() -> ECAdd:
    n, p = sympy.symbols('n p')
    ec_add = ECAdd(n, mod=p)
    return ec_add


_EC_ADD_DOC = BloqDocSpec(bloq_cls=ECAdd, examples=[_ec_add])
