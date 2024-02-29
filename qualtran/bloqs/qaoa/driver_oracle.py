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
"""Oracles for implementing the mixing  Sherrington-Kirkpatrick (SK) model"""
from typing import Set

import attrs
import cirq
from numpy.typing import NDArray

from qualtran import GateWithRegisters, QFxp, Signature
from qualtran.bloqs.basic_gates import Rx
from qualtran.resource_counting.symbolic_counting_utils import SymbolicFloat, SymbolicInt


@attrs.frozen
class DriverOracle(GateWithRegisters):
    """Implements the problem-independent driver unitary $U_{B}(β)=\prod_j \exp(-i β X_j)$"""

    bitsize: SymbolicInt
    beta: SymbolicFloat
    eps: SymbolicFloat

    @property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(x=QFxp(self.bitsize, self.bitsize))

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, x: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        yield cirq.Moment(Rx(angle=2 * self.beta, eps=self.eps / self.bitsize).on_each(*x))

    def __str__(self):
        return f'DriverOracle[{self.beta}]'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(Rx(angle=2 * self.beta, eps=self.eps / self.bitsize), self.bitsize)}
