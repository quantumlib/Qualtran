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

import numpy as np
from attrs import frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    BloqBuilder,
    BQUInt,
    QAny,
    QBit,
    QFxp,
    QUInt,
    Register,
    Signature,
    Soquet,
    SoquetT,
)

from .atom import TestAtom, TestTwoBitOp


@frozen
class TestMultiRegister(Bloq):
    """A bloq with multiple, interesting registers.

    Registers:
        xx: A one-bit register that gets an operation applied to it.
        yy: A matrix of 2-bit registers each of which gets a two bit operation applied to it.
        zz: A three-bit register that is split and re-joined.
    """

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [Register('xx', QBit()), Register('yy', QAny(2), shape=(2, 2)), Register('zz', QAny(3))]
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', xx: 'SoquetT', yy: NDArray['Soquet'], zz: Soquet  # type: ignore[type-var]
    ) -> Dict[str, 'SoquetT']:
        xx = bb.add(TestAtom(), q=xx)
        for i in range(2):
            for j in range(2):
                a, b = bb.split(yy[i, j])  # type: ignore[index]
                a, b = bb.add(TestTwoBitOp(), ctrl=a, target=b)
                yy[i, j] = bb.join(np.array([a, b]))
        a, b, c = bb.split(zz)
        zz = bb.join(np.array([a, b, c]))
        return {'xx': xx, 'yy': yy, 'zz': zz}

    def pretty_name(self) -> str:
        return 'xyz'


@frozen
class TestBoundedQUInt(Bloq):
    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('xx', BQUInt(4, 3)), Register('yy', QFxp(8, 6, True))])


@frozen
class TestQFxp(Bloq):
    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('xx', QFxp(8, 6, True)), Register('yy', QFxp(8, 8))])


@frozen
class TestMultiTypedRegister(Bloq):
    """A bloq with multiple, interesting registers.

    Registers:
        xx: A one-bit register that gets an operation applied to it.
        yy: A matrix of 2-bit registers each of which gets a two bit operation applied to it.
        zz: A three-bit register that is split and re-joined.
    """

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('a', BQUInt(4, 3)),
                Register('b', QFxp(8, 6, True)),
                Register('c', QFxp(8, 8)),
                Register('d', QUInt(8)),
            ]
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', a: 'SoquetT', b: 'SoquetT', c: 'SoquetT', d: 'Soquet'
    ) -> Dict[str, 'Soquet']:
        a, b = bb.add(TestBoundedQUInt(), xx=a, yy=d)
        b, c = bb.add(TestQFxp(), xx=b, yy=c)
        return {'a': a, 'b': b, 'c': c, 'd': d}

    def pretty_name(self) -> str:
        return 'abcd[T]'
