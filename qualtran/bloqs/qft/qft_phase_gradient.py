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
from qualtran.bloqs.arithmetic.multiplication import PlusEqualProduct


@attrs.frozen
class QFTPhaseGradient(GateWithRegisters):
    r"""QFT implemented using phase gradient trick. Uses O(n**2) T-gates for an n-bit register.

    Given an n-bit phase gradient state $|\phi\rangle$ prepared as

    $$
        |\phi\rangle = \frac{1}{\sqrt{2^{n}}} \sum_{k=0}^{2^{n} - 1} \omega_{n}^{-k} |k\rangle
    $$

    Phase gradient rotations can be synthesized via additions into the phase gradient register.
    This leads to significant reductions in T/Toffoli complexity and requires 0 arbitrary
    rotations (given a one-time cost to prepare the gradient register). See the linked reference
    for more details.

    Args:
        bitsize: Size of input register to apply QFT on.
        with_reverse: Whether or not to include the swaps at the end
            of the circuit decomposition that reverse the order of the
            qubits. If True, the swaps are inserted. Defaults to True.
            These are technically necessary in order to perform the
            correct effect, but can almost always be optimized away by just
            performing later operations on different qubits.

    References:
        [Turning Gradients into Additions into QFTs](https://algassert.com/post/1620)
    """

    bitsize: int
    with_reverse: bool = True

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(q=self.bitsize, phase_grad=self.bitsize)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        if self.bitsize == 1:
            yield cirq.H(*quregs['q'])
            return
        q, phase_grad = quregs['q'], quregs['phase_grad']
        a, b = q[: self.bitsize // 2], q[self.bitsize // 2 :]
        yield QFTPhaseGradient(len(a), False).on_registers(q=a, phase_grad=phase_grad[: len(a)])
        yield PlusEqualProduct(len(a), len(b), len(phase_grad)).on_registers(
            a=a[::-1], b=b, result=phase_grad
        )
        yield QFTPhaseGradient(len(b), False).on_registers(q=b, phase_grad=phase_grad[: len(b)])
        if self.with_reverse:
            for i in range(self.bitsize // 2):
                yield cirq.SWAP(q[i], q[-i - 1])
