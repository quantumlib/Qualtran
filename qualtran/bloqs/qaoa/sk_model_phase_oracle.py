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
"""Phase Oracles for Sherrington-Kirkpatrick (SK) model

The Sherrington-Kirkpatrick (SK) model describes a classical spin system with all-to-all
couplings between the $n$ spins. The classical cost function $C$ for $n$-bit SK model
is defined as:

$$
C(z) = \sum_{j < k}^{n} w_{j, k} z_{j} z_{k} \; \text{where}\;  w_{j, k} \in \{0, 1\}
$$
"""
import abc
from typing import Dict, Set, Tuple

import attrs
import cirq
import numpy as np
from numpy.typing import NDArray
from sympy.functions.combinatorial.factorials import binomial

from qualtran import Bloq, GateWithRegisters, QFxp, Register, Signature
from qualtran.bloqs.basic_gates.rotation import ZZPowGate
from qualtran.bloqs.qaoa.sk_model_cost_oracle import SKModelCostEval
from qualtran.bloqs.rotations.phasing_via_cost_function import PhasingViaCostFunction
from qualtran.bloqs.rotations.quantum_variable_rotation import (
    QvrInterface,
    QvrPhaseGradient,
    QvrZPow,
)
from qualtran.resource_counting.symbolic_counting_utils import (
    ceil,
    log2,
    SymbolicFloat,
    SymbolicInt,
)


@attrs.frozen
class SKPhaseOracleNaiveRZZ(GateWithRegisters):
    """Implements the problem-dependent unitary $U_{C}(γ)=\exp(-i γ C)$

    For the SK-model, the phase oracle $U_{C}(γ)$ can be expressed as
    $$
       U_{C}(γ) =\prod_{j<k}\exp(-i γ w_{j,k}Z_jZ_k)$
    $$
    """

    bitsize: SymbolicInt
    gamma: SymbolicFloat
    weights: Tuple[int, ...]
    eps: SymbolicFloat

    @property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(x=QFxp(self.bitsize, self.bitsize))

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, x: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        w_idx = 0
        n = self.bitsize
        eps_per_rotation = 2 * self.eps / (n * (n - 1))
        for i in range(n):
            for j in range(i + 1, n):
                # Equivalent to an Rzz rotation by angle w[w_idx] * gamma.
                yield ZZPowGate(
                    exponent=2 * self.w[w_idx] * self.gamma / np.pi,
                    global_shift=-0.5,
                    eps=eps_per_rotation,
                ).on(x[i], x[j])
                w_idx = w_idx + 1

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        n = self.bitsize
        num_rotations = binomial(n, 2)
        rzz = ZZPowGate(
            exponent=2 * self.gamma / np.pi, global_shift=-0.5, eps=self.eps / num_rotations
        )
        return {(rzz, num_rotations)}

    def __str__(self):
        return f'{self.__class__.__name__}[{self.gamma}, {self.eps}]'


@attrs.frozen
class SKPhaseViaCostBase(Bloq, metaclass=abc.ABCMeta):
    """Implements the problem-dependent unitary $U_{C}(γ)=\exp(-i γ C)$ using `PhasingViaCostFunction`"""

    bitsize: SymbolicInt
    gamma: SymbolicFloat
    weights: Tuple[int, ...]
    eps: SymbolicFloat

    @property
    def signature(self) -> Signature:
        return self.phase_via_cost_function_oracle.signature

    @property
    def cost_reg(self) -> Register:
        return Register(
            'cost_reg',
            QFxp(2 * ceil(log2(self.bitsize)), 2 * ceil(log2(self.bitsize)), signed=False),
        )

    @property
    def cost_eval_oracle(self) -> Bloq:
        return SKModelCostEval(self.bitsize, self.weights)

    @property
    @abc.abstractmethod
    def qvr_oracle(self) -> QvrInterface:
        ...

    @property
    def phase_via_cost_function_oracle(self) -> PhasingViaCostFunction:
        return PhasingViaCostFunction(self.cost_eval_oracle, self.qvr_oracle)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        return bb.add_d(self.phase_via_cost_function_oracle, **soqs)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(self.phase_via_cost_function_oracle, 1)}

    def __str__(self):
        return f'{self.__class__.__name__}[{self.gamma}, {self.eps}]'


@attrs.frozen
class SKPhaseViaCostQvrZPow(SKPhaseViaCostBase):
    """Implements the problem-dependent unitary $U_{C}(γ)=\exp(-i γ C) using `QvrZPow`$"""

    @property
    def qvr_oracle(self) -> QvrZPow:
        return QvrZPow(self.cost_reg, self.gamma, self.eps)


@attrs.frozen
class SKPhaseViaCostQvrPhaseGrad(SKPhaseViaCostBase):
    """Implements the problem-dependent unitary $U_{C}(γ)=\exp(-i γ C)$ using `QvrPhaseGradient`"""

    @property
    def qvr_oracle(self) -> QvrPhaseGradient:
        return QvrPhaseGradient(self.cost_reg, self.gamma, self.eps)
