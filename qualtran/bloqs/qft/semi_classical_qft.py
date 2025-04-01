#  Copyright 2025 Google LLC
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
from typing import TYPE_CHECKING

import attrs

from qualtran import Bloq, bloq_example, BloqDocSpec, QUInt, Register, Side, Signature
from qualtran.bloqs.basic_gates import Hadamard, Rz
from qualtran.bloqs.basic_gates._shims import Measure
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator

if TYPE_CHECKING:
    from qualtran.symbolics import SymbolicInt


@attrs.frozen
class SemiClassicalQFT(Bloq):
    r"""Represents QFT followed by measurement.

    When QFT is followed by measurement, we can replace all two qubit gates with classically
    controlled Rz rotations (at most $n-1$ Rz rotations). The two structures (QFT + Measurement)
    and SemiClassicaQFT behave statistically the same.


    Registers:
        q: A `QUInt` of `bitsize` qubits on which the QFT is performed and then measured.

    Args:
        bitsize: Size of the input register to apply QFT on.
        adjoint: Whether to apply QFT or QFT†.

    References:
        [Semiclassical Fourier Transform for Quantum Computation, Griffiths & Niu](https://arxiv.org/abs/quant-ph/9511007)
        
        [Implementation of the Semiclassical Quantum Fourier Transform in a Scalable System](https://www.science.org/doi/10.1126/science.1110335)

        [stackexchange answer, Gidney](https://quantumcomputing.stackexchange.com/a/23712)
    """

    bitsize: 'SymbolicInt'
    adjoint: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('q', QUInt(self.bitsize), side=Side.LEFT)])

    def build_call_graph(self, ssa: SympySymbolAllocator) -> 'BloqCountDictT':
        t = ssa.new_symbol('t')
        return {Hadamard(): self.bitsize, Measure(): self.bitsize, Rz(t): self.bitsize - 1}


@bloq_example
def _semi_classical_qft() -> SemiClassicalQFT:
    semi_classical_qft = SemiClassicalQFT(3)
    return semi_classical_qft


_SEMI_CLASSICAL_QFT_DOC = BloqDocSpec(bloq_cls=SemiClassicalQFT, examples=(_semi_classical_qft,))
