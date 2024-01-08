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

import attrs
import cirq
from numpy.typing import NDArray

from qualtran import GateWithRegisters, Signature
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientUnitary


@attrs.frozen
class QFTTextBook(GateWithRegisters):
    r"""Textbook version of QFT using $\frac{N (N + 1)}{2}$ controlled rotations.

    Args:
        bitsize: Size of the input register to apply QFT on.
        with_reverse: Whether or not to include the swaps at the end
            of the circuit decomposition that reverse the order of the
            qubits. If True, the swaps are inserted. Defaults to True.
            These are technically necessary in order to perform the
            correct effect, but can almost always be optimized away by just
            performing later operations on different qubits.
    """

    bitsize: int
    with_reverse: bool = True

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(q=self.bitsize)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, q: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        yield cirq.H(q[0])
        for i in range(1, len(q)):
            yield PhaseGradientUnitary(i, exponent=0.5, controlled=True).on_registers(
                ctrl=q[i], phase_grad=q[:i][::-1]
            )
            yield cirq.H(q[i])
        if self.with_reverse:
            for i in range(self.bitsize // 2):
                yield cirq.SWAP(q[i], q[-i - 1])
