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
from typing import Iterator

import attrs
import cirq
from numpy.typing import NDArray
from sympy.functions.elementary.exponential import log

from qualtran import bloq_example, BloqDocSpec, GateWithRegisters, QFxp, QUInt, Signature
from qualtran.bloqs.arithmetic.multiplication import PlusEqualProduct
from qualtran.bloqs.basic_gates import Hadamard
from qualtran.bloqs.basic_gates.swap import Swap
from qualtran.resource_counting import BloqCountDictT, MutableBloqCountDictT, SympySymbolAllocator
from qualtran.symbolics.types import is_symbolic


@attrs.frozen
class QFTPhaseGradient(GateWithRegisters):
    r"""QFT implemented using coherent addition into a phase gradient register

     A variant of the Quantum Fourier Transform (QFT) that utilizes an additional register provided
     in a phase gradient state to switch controlled rotations to coherent additions. Given an
     ancilla register prepared in the state
    $$
        \frac{1}{\sqrt{2^{n}}} \sum_{k=0}^{2^{n} - 1} \omega_{n}^{-k} |k\rangle,
    $$
    then coherent addition from the system into the ancilla applies the same phase that the
    controlled rotation in textbook QFT does. This reduces the number of T-gates to $O(n^2)$
    and requires no additional arbitrary rotations beyond the one time ancilla preparation cost.

    The size of the ancilla register is important, if the ancilla has less
    qubits than the system register then the accuracy of the QFT applied will be
    reduced. This implementation assumes an ancilla with `bitsize` qubits. See `ApproximateQFT` for
    an implementation or the linked reference for details.

    Args:
        bitsize: Size of input register to apply QFT on.
        with_reverse: Whether or not to include the swaps at the end
            of the circuit decomposition that reverse the order of the
            qubits. If True, the swaps are inserted. Defaults to True.
            These are technically necessary in order to perform the
            correct effect, but can almost always be optimized away by just
            performing later operations on different qubits.

    Registers:
        q: The register to perform the QFT on.
        phase_grad: An ancilla register assumed to be prepared in a phase gradient state. See
            `qualtran/bloqs/rotations/phase_gradient` for more information on how to prepare these
            states.

    Costs:
        Qubits: Requires $2n$ qubits, $n$ for the register the QFT is performed on and $n$ for the
            phase gradient ancilla. No additional qubits are allocated.
        T gates: $O(n^2)$, based on the approximation for `PlusEqualsProduct`

    References:
        [Turning Gradients into Additions into QFTs](https://algassert.com/post/1620)
    """

    bitsize: int
    with_reverse: bool = True

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(
            q=QUInt(self.bitsize), phase_grad=QFxp(self.bitsize, self.bitsize)
        )

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]  # type: ignore[type-var]
    ) -> Iterator[cirq.OP_TREE]:
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

    def build_call_graph(self, ssa: SympySymbolAllocator) -> 'BloqCountDictT':
        if is_symbolic(self.bitsize):
            # TODO: The T-gate cost here is an upper bound constructed off of the recurrence
            # relation for the QFT as used in decompose_from_registers above.
            return {
                PlusEqualProduct(
                    self.bitsize // 2, self.bitsize - (self.bitsize // 2), self.bitsize
                ): log(self.bitsize, 2)
            }

        if self.bitsize == 1:
            return {Hadamard(): 1}
        ret: 'MutableBloqCountDictT' = {
            QFTPhaseGradient(self.bitsize // 2): 1,
            QFTPhaseGradient(self.bitsize - (self.bitsize // 2)): 1,
            PlusEqualProduct(
                self.bitsize // 2, self.bitsize - (self.bitsize // 2), self.bitsize
            ): 1,
        }
        if self.with_reverse:
            ret[Swap(1)] = self.bitsize // 2
        return ret


@bloq_example
def _qft_phase_gradient_small() -> QFTPhaseGradient:
    qft_phase_gradient_small = QFTPhaseGradient(3)
    return qft_phase_gradient_small


_QFT_PHASE_GRADIENT_DOC = BloqDocSpec(
    bloq_cls=QFTPhaseGradient, examples=(_qft_phase_gradient_small,)
)
