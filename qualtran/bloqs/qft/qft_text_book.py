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
import sympy
from numpy.typing import NDArray

from qualtran import bloq_example, BloqDocSpec, GateWithRegisters, QUInt, Signature
from qualtran.bloqs.basic_gates.hadamard import Hadamard
from qualtran.bloqs.basic_gates.swap import TwoBitSwap
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientUnitary
from qualtran.resource_counting import BloqCountDictT, MutableBloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import SymbolicInt
from qualtran.symbolics.types import is_symbolic


@attrs.frozen
class QFTTextBook(GateWithRegisters):
    r"""Standard Quantum Fourier Transform from Nielsen and Chuang

    Performs the QFT on a register of `bitsize` qubits utilizing
    $n$ Hadamards and $n * (n - 1) / 2$ controlled Z
    rotations, along with a reversal of qubit ordering specified via
    `with_reverse` which defaults to `True`. `bitsize` can be provided numerically or symbolically.
    More specific QFT implementations can be found:
    - `ApproximateQFT` A less accurate QFT which ignores small phase rotations.
    - `QFTPhaseGradient` requires an additional input phase gradient register
    to be provided but utilizes controlled addition instead of rotations, which leads to reduced
    T-gate complexity.
    - `TwoBitFFFT` if you need to implement a two-qubit fermionic Fourier transform.

    Registers:
        q: A `QUInt` of `bitsize` qubits on which the QFT is performed.

    References:
        [Quantum Computation and Quantum Information: 10th Anniversary Edition,
            Nielsen & Chuang](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview)
            Chapter 5.1
    Args:
        bitsize: Size of the input register to apply QFT on.
        with_reverse: Whether or not to include the swaps at the end
            of the circuit decomposition that reverse the order of the
            qubits. If True, the swaps are inserted. Defaults to True.
            These are technically necessary in order to perform the
            correct effect, but can almost always be optimized away by just
            performing later operations on different qubits.

    Costs:
        Qubits: $n$ qubits, no additional ancilla required.
        Gates: $n * (n - 1) / 2$ controlled-rotation gates and $n$ Hadamard gates.
    """

    bitsize: SymbolicInt
    with_reverse: bool = True

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(q=QUInt(self.bitsize))

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, q: NDArray[cirq.Qid]  # type: ignore[type-var]
    ) -> Iterator[cirq.OP_TREE]:
        yield cirq.H(q[0])
        for i in range(1, len(q)):
            yield PhaseGradientUnitary(i, exponent=0.5, is_controlled=True).on_registers(
                ctrl=q[i], phase_grad=q[:i][::-1]
            )
            yield cirq.H(q[i])
        if self.with_reverse:
            for i in range(self.bitsize // 2):
                yield cirq.SWAP(q[i], q[-i - 1])

    def build_call_graph(self, ssa: SympySymbolAllocator) -> 'BloqCountDictT':
        ret: 'MutableBloqCountDictT' = {Hadamard(): self.bitsize}
        if is_symbolic(self.bitsize):
            ret[PhaseGradientUnitary(self.bitsize - 1, exponent=0.5, is_controlled=True)] = (
                self.bitsize // 2
            )
        else:
            for i in range(1, self.bitsize):
                ret[PhaseGradientUnitary(i, exponent=0.5, is_controlled=True)] = 1
        if self.with_reverse:
            ret[TwoBitSwap()] = self.bitsize // 2
        return ret


@bloq_example
def _qft_text_book() -> QFTTextBook:
    qft_text_book = QFTTextBook(3)
    return qft_text_book


@bloq_example
def _symbolic_qft() -> QFTTextBook:
    n = sympy.symbols('n')
    symbolic_qft = QFTTextBook(bitsize=n)
    return symbolic_qft


_QFT_TEXT_BOOK_DOC = BloqDocSpec(bloq_cls=QFTTextBook, examples=(_qft_text_book, _symbolic_qft))
