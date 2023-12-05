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

from qualtran import Bloq, Register, Signature

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
        return Signature([Register('xx', 1), Register('yy', 2, shape=(2, 2)), Register('zz', 3)])

    def build_composite_bloq(
        self, bb: 'BloqBuilder', xx: 'SoquetT', yy: 'SoquetT', zz: 'SoquetT'
    ) -> Dict[str, 'Soquet']:
        xx = bb.add(TestAtom(), q=xx)
        for i in range(2):
            for j in range(2):
                a, b = bb.split(yy[i, j])
                a, b = bb.add(TestTwoBitOp(), ctrl=a, target=b)
                yy[i, j] = bb.join(np.array([a, b]))
        a, b, c = bb.split(zz)
        zz = bb.join(np.array([a, b, c]))
        return {'xx': xx, 'yy': yy, 'zz': zz}

    def short_name(self) -> str:
        return 'xyz'
