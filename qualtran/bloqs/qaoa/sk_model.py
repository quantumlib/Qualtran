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
"""Oracles for Sherrington-Kirkpatrick (SK) model

The Sherrington-Kirkpatrick (SK) model describes a classical spin system with all-to-all
couplings between the $n$ spins. The classical cost function $C$ for $n$-bit SK model
is defined as:

$$
C(z) = \sum_{j < k}^{n} w_{j, k} z_{j} z_{k} \; \text{where}\;  w_{j, k} \in \{0, 1\}
$$
"""
from functools import cached_property
from typing import Tuple, Sequence

import attrs
import cirq
from numpy.typing import NDArray

from qualtran import GateWithRegisters, Signature, QFxp
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientUnitary

from qualtran import GateWithRegisters, Signature, Register, Side, SoquetT, BloqBuilder
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState, AddScaledValIntoPhaseReg
from qualtran.bloqs.basic_gates import TGate, Hadamard, Rx
from qualtran.bloqs.on_each import OnEach
from typing import Dict, Optional, Set


def _cost(n: int, x: int, w: Sequence[int]) -> int:
    ret = 0
    w_idx = 0
    for j in range(n):
        for k in range(j + 1, n):
            ret += (((x >> j) ^ (x >> k)) & 1) * w[w_idx]
            w_idx += 1
    return w_idx


@attrs.frozen
class SKModelCostEval(GateWithRegisters):
    """Implements the cost function evaluation oracle $O^{\text{direct}}$ for SK-Model.

    Evaluates the cost function $C(z)$ for SK-Model and stores the output in a new clean register.
    The oracle is defined as

    $$
        O^{\text{direct}}|z\rangle |0\rangle^{\otimes b_{\text{dir}}}  = |z\rangle |c(z)\rangle
    $$

    Here $b_{\text{dir}}$ is the bitsize of the newly allocated ancilla register and in the case of
    SK model, it is equal to $2\log{N}$.

    Args:
        n: Size of the spin system described by the SK model.
        w: A tuple of size $n \choose 2$ describing coefficients ($\in \{0, 1\}$) for the
            cost function $C(z)$.
        uncompute: Whether to uncompute the hamming weight instead.

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization]
        (https://arxiv.org/abs/2007.07391)
        Appendix A: Addition for controlled rotations
    """

    n: int
    w: Tuple[int, ...]
    uncompute: bool = False

    @property
    def signature(self) -> Signature:
        side = Side.LEFT if self.uncompute else Side.RIGHT
        return Signature(
            [
                Register('x', self.n),
                Register(
                    'cost_reg',
                    QFxp(2 * self.n.bit_length(), 2 * self.n.bit_length(), signed=False),
                    side=side,
                ),
            ]
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set[Tuple[int, 'Bloq']]:
        num_t = 0 if self.uncompute else self.n**2
        return {(num_t, TGate())}

    def on_classical_vals(self, **vals: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        x = vals['x']
        if self.uncompute:
            _ = vals.pop('cost_reg')
            return vals
        return {'x': x, 'cost_reg': _cost(self.n, x, self.w)}
