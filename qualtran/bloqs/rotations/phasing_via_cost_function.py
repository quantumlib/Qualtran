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
from typing import Dict, TYPE_CHECKING

import attrs

from qualtran import Bloq, bloq_example, BloqBuilder, BloqDocSpec, Side, Signature
from qualtran.bloqs.rotations.quantum_variable_rotation import QvrInterface

if TYPE_CHECKING:
    from qualtran import SoquetT


@attrs.frozen
class PhasingViaCostFunction(Bloq):
    r"""Phases every basis state $|x\rangle$ by an amount proportional to a cost function $f(x)$

    This Bloq implements a unitary $U_f(\gamma)$ which phases each computational state on which
    the wave-function has support, by an amount proportional to a function of that computational
    basis state. The general unitary can be defined as
    $$
        U_f(\gamma) = \sum_{x=0}^{N-1} e^{i 2 \pi \gamma f(x)} |x\rangle \langle x|
    $$

    The strategy to implement $U_f(\gamma)$ is to use two oracles $O_\text{direct}$
    and $O_\text{phase}$ s.t.
    $$
    U_f(\gamma) = O_\text{direct}^\dagger(\mathbb{I}\otimes O_\text{phase})O_\text{direct}
    $$

    $O^\text{direct}$ evaluates a $b_\text{direct}$-bit approximation of the cost function $f(x)$
    and stores it in a new output cost register. Note that the cost register can represent
    arbitrary fixed point values and be of type `QFxp(b_direct, n_frac, signed)`.
    $$
    O^\text{direct}|x\rangle|0\rangle^{\otimes b_\text{direct}}_\text{cost}=|x\rangle|f(x)\rangle
    $$

    $O^\text{phase}$ acts on the cost register computed by $O^\text{direct}$ and phases the
    state $|f(x)\rangle$ by $e^{i 2\pi \gamma f(x)}$
    $$
    O^\text{phase}(\gamma)=\sum_{k=0}^{2^{b_\text{direct}}-1}e^{i 2\pi\gamma k}|k\rangle\langle k|
    $$


    Different strategies for implementing the two oracles would give different costs tradeoffs.
    See `QvrZPow` and `QvrPhaseGradient` for two different implementations of
    phase oracles described in the reference.

    Args:
        cost_eval_oracle: Cost function evaluation oracle. Must compute the cost in a
            newly allocated RIGHT register.
        phase_oracle: Oracle to phase the cost register. Must consume the cost register
            allocated by `cost_eval_oracle` as a THRU input.

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization](https://arxiv.org/abs/2007.07391).
        Appendix C: Oracles for phasing by cost function
    """
    cost_eval_oracle: Bloq
    phase_oracle: QvrInterface

    def __attrs_post_init__(self):
        for cost_reg in self.phase_oracle.cost_registers:
            cost_eval_right_reg = self.cost_eval_oracle.signature.get_right(cost_reg.name)
            assert cost_reg.dtype.num_qubits == cost_eval_right_reg.dtype.num_qubits
            assert cost_reg.shape == cost_eval_right_reg.shape
            assert cost_reg.side == Side.THRU
            assert cost_eval_right_reg.side == Side.RIGHT

    @cached_property
    def signature(self) -> 'Signature':
        registers = [*self.cost_eval_oracle.signature.lefts(), *self.phase_oracle.extra_registers]
        return Signature(registers)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        def _extract_soqs(bloq: Bloq) -> Dict[str, 'SoquetT']:
            return {reg.name: soqs.pop(reg.name) for reg in bloq.signature.lefts()}

        soqs |= bb.add_d(self.cost_eval_oracle, **_extract_soqs(self.cost_eval_oracle))
        soqs |= bb.add_d(self.phase_oracle, **_extract_soqs(self.phase_oracle))
        cost_eval_adjoint = self.cost_eval_oracle.adjoint()
        soqs |= bb.add_d(cost_eval_adjoint, **_extract_soqs(cost_eval_adjoint))
        return soqs


@bloq_example
def _square_via_zpow_phasing() -> PhasingViaCostFunction:
    from qualtran import QFxp, Register
    from qualtran.bloqs.arithmetic import Square
    from qualtran.bloqs.rotations.quantum_variable_rotation import QvrZPow

    n, gamma, eps = 5, 0.1234, 1e-8
    cost_reg = Register('result', QFxp(2 * n, 2 * n, signed=False))
    cost_eval_oracle = Square(n)
    phase_oracle = QvrZPow(cost_reg, gamma, eps)
    square_via_zpow_phasing = PhasingViaCostFunction(cost_eval_oracle, phase_oracle)
    return square_via_zpow_phasing


@bloq_example
def _square_via_phase_gradient() -> PhasingViaCostFunction:
    from qualtran import QFxp, Register
    from qualtran.bloqs.arithmetic import Square
    from qualtran.bloqs.rotations.quantum_variable_rotation import QvrPhaseGradient

    n, gamma, eps = 5, 0.1234, 1e-8
    cost_reg = Register('result', QFxp(2 * n, 2 * n, signed=False))
    cost_eval_oracle = Square(n)
    phase_oracle = QvrPhaseGradient(cost_reg, gamma, eps)
    square_via_phase_gradient = PhasingViaCostFunction(cost_eval_oracle, phase_oracle)
    return square_via_phase_gradient


_PHASING_VIA_COST_FUNCTION = BloqDocSpec(
    bloq_cls=PhasingViaCostFunction, examples=(_square_via_phase_gradient, _square_via_zpow_phasing)
)
