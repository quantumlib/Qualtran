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
from typing import Any, Dict, List

import attrs
import cirq
import quimb.tensor as qtn
from numpy._typing import NDArray

from qualtran import GateWithRegisters, QAny, QUInt, Register, Side, Signature, SoquetT
from qualtran.bloqs.basic_gates import TwoBitCSwap
from qualtran.cirq_interop._cirq_to_bloq import _add_my_tensors_from_gate
from qualtran.simulation.classical_sim import ClassicalValT


@attrs.frozen
class OneHotEncoding(GateWithRegisters):
    """One-hot encode a binary integer into a target register.

    Args:
        binary_bitsize: The number of bits in the binary integer register. There will be 2^binary_bitsize bits in the one-hot-encoded register.

    Registers:
        a: an unsigned integer
        b: the target to one-hot encode.

    References:
        [Windowed quantum arithmetic](https://arxiv.org/abs/1905.07682)
        Figure 4
    """

    binary_bitsize: int

    @property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('a', QUInt(self.binary_bitsize), side=Side.THRU),
                Register('b', QAny(2**self.binary_bitsize), side=Side.THRU),
            ]
        )

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        _add_my_tensors_from_gate(
            self, self.signature, self.short_name(), tn, tag, incoming=incoming, outgoing=outgoing
        )

    def on_classical_vals(
        self, a: 'ClassicalValT', b: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        return {'a': a, 'b': int(2**a)}

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]  # type: ignore[type-var]
    ) -> cirq.OP_TREE:
        a = quregs['a'][::-1]
        b = quregs['b']

        op_tree: List[cirq.Operation] = []
        op_tree.append(cirq.X(b[0]))
        for i in range(len(a)):
            for j in range(2**i):
                op_tree.append(TwoBitCSwap().on_registers(ctrl=a[i], x=b[j], y=b[2**i + j]))
        return op_tree

    def short_name(self) -> str:
        return "one-hot-enc"
