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
import itertools
import math
from typing import Any, Dict

import attrs
import cirq
from numpy._typing import NDArray

from qualtran import GateWithRegisters, QAny, QUInt, Signature
from qualtran.bloqs.basic_gates import TwoBitCSwap


@attrs.frozen
class OneHotEncoding(GateWithRegisters):
    """
    One-hot encode a binary integer into a target register.

    Registers:
        a: an unsigned integer
        b: the target to one-hot encode.

    References:
        [Windowed quantum arithmetic](https://arxiv.org/pdf/1905.07682.pdf)
        Figure 4
    """

    binary_bitsize: int

    @property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(
            a=QUInt(self.binary_bitsize), b=QAny(2**self.binary_bitsize)
        )

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        a = quregs['a'][::-1]
        b = quregs['b']

        yield cirq.X(b[0])
        for i in range(len(a)):
            for j in range(2**i):
                yield TwoBitCSwap().on_registers(ctrl=a[i], x=b[j], y=b[2**i + j])
