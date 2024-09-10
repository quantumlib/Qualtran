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
from collections import defaultdict
from functools import cached_property
from typing import Iterator, TYPE_CHECKING

import attrs
import cirq
import numpy as np
import sympy
from attr import field
from numpy.typing import NDArray

from qualtran import bloq_example, BloqDocSpec, GateWithRegisters, QFxp, QUInt, Signature
from qualtran.bloqs.basic_gates import Hadamard, TwoBitSwap
from qualtran.bloqs.rotations import AddIntoPhaseGrad
from qualtran.symbolics import ceil, is_symbolic, log2, SymbolicFloat, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import (
        BloqCountDictT,
        MutableBloqCountDictT,
        SympySymbolAllocator,
    )


@attrs.frozen
class ApproximateQFT(GateWithRegisters):
    r"""An approximate QFT in which phase shifts smaller than a certain threshold are deleted.

    Given a b-bit phase gradient state $|\phi\rangle$ prepared as

    $$
        |\phi\rangle = \frac{1}{\sqrt{2^{b}}} \sum_{k=0}^{2^{b} - 1} \omega_{b}^{-k} |k\rangle
    $$

    Phase gradient rotations can be synthesized via additions into the phase gradient register.
    This leads to significant reductions in T/Toffoli complexity and requires 0 arbitrary
    rotations (given a one-time cost to prepare the gradient register). See the linked reference
    for more details.

    The QFT uses exponentially small z-power gates. In practice, it is often sufficient to perform
    an approximate qft, where z-power gates smaller than a certain threshold are dropped. When using
    the "add into phase-gradient trick", this amounts to doing smaller additions with a smaller
    phase gradient register.

    Args:
        bitsize: Size of input register to apply QFT on.
        phase_bitsize: The size of the phase gradient register. Defaults to being math.ceil(math.log2(bitsize)).
        with_reverse: Whether or not to include the swaps at the end
            of the circuit decomposition that reverse the order of the
            qubits. If True, the swaps are inserted. Defaults to True.
            These are technically necessary in order to perform the
            correct effect, but can almost always be optimized away by just
            performing later operations on different qubits.

    References:
        [Turning Gradients into Additions into QFTs](https://algassert.com/post/1620).
        Gidney, C. 2016.

        [Approximation Errors](https://arxiv.org/abs/quant-ph/0008056).
        Panike, N. 2000.
    """

    bitsize: SymbolicInt
    phase_bitsize: SymbolicInt = field()
    with_reverse: bool = True

    @phase_bitsize.default
    def ceiling_of_log_bitsize(self):
        return ceil(log2(self.bitsize))

    @classmethod
    def from_epsilon(cls, n: SymbolicInt, eps: SymbolicFloat) -> 'ApproximateQFT':
        """Builds an ApproximateQFT instance using total tolerable error `eps`.

        From a given error threshold, epsilon, calculates what size the
        phase register should be and returns an ApproximateQFT instance.

        Args:
            n: The size of the input register.
            eps: The error threshold.

        Returns:
            An ApproximateQFT instance.
        """

        # solving for k in quant-ph/0008056 (12)
        # This equation is transcendental because it involves both
        # algebraic and exponential terms in k. Therefore, I upper-bounded
        # the error by replacing (n - k) with n.
        num = sympy.pi * sympy.sqrt(2) if is_symbolic(n, eps) else np.pi * np.sqrt(2)
        phase_bitsize = ceil(log2((n * num) / eps))
        return cls(n, phase_bitsize)

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(
            q=QUInt(self.bitsize), phase_grad=QFxp(self.phase_bitsize, self.phase_bitsize)
        )

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]  # type: ignore[type-var]
    ) -> Iterator[cirq.OP_TREE]:
        if self.bitsize == 1:
            yield cirq.H(*quregs['q'])
            return
        q, phase_grad = quregs['q'], quregs['phase_grad']
        for i in range(len(q)):
            if i == 0:
                yield cirq.H(q[i])
                continue
            addition_bitsize = min(i, len(phase_grad) - 1)
            addition_start_index = i - addition_bitsize
            a, b = q[addition_start_index:i], phase_grad[: addition_bitsize + 1]

            yield AddIntoPhaseGrad(
                addition_bitsize, addition_bitsize + 1, right_shift=1, controlled_by=1
            ).on_registers(ctrl=q[i], x=a[::-1], phase_grad=b)
            yield cirq.H(q[i])

        if self.with_reverse:
            for i in range(self.bitsize // 2):
                yield cirq.SWAP(q[i], q[-i - 1])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        phase_dict: 'MutableBloqCountDictT' = defaultdict(int)
        if is_symbolic(self.bitsize, self.phase_bitsize):
            phase_dict[
                AddIntoPhaseGrad(
                    self.phase_bitsize, self.phase_bitsize, right_shift=1, controlled_by=1
                )
            ] = self.bitsize
        else:
            for i in range(1, int(self.bitsize)):
                b = min(i, self.phase_bitsize - 1)
                phase_dict[AddIntoPhaseGrad(b, b + 1, right_shift=1, controlled_by=1)] += 1
        phase_dict[Hadamard()] = self.bitsize
        if self.with_reverse:
            phase_dict[TwoBitSwap()] = self.bitsize // 2
        return phase_dict


@bloq_example
def _approximate_qft_small() -> ApproximateQFT:
    approximate_qft_small = ApproximateQFT(6, 5)
    return approximate_qft_small


@bloq_example
def _approximate_qft_from_epsilon() -> ApproximateQFT:
    epsilon = 1e-5
    approximate_qft_from_epsilon = ApproximateQFT.from_epsilon(50, epsilon)
    return approximate_qft_from_epsilon


_CC_AQFT_DOC = BloqDocSpec(
    bloq_cls=ApproximateQFT, examples=(_approximate_qft_small, _approximate_qft_from_epsilon)
)
