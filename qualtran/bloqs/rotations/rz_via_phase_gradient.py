#  Copyright 2024 Google LLC
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
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    QBit,
    QDType,
    QUInt,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic.controlled_add_or_subtract import ControlledAddOrSubtract
from qualtran.bloqs.bookkeeping import Cast
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@frozen
class RzViaPhaseGradient(Bloq):
    r"""Apply a controlled-Rz using a phase gradient state.

    Implements the following unitary action:

    $$
        |\psi\rangle \otimes |x\rangle \mapsto \text{Rz}(4 \pi x) |\psi\rangle \otimes |x\rangle
    $$

    for every state $|\psi\rangle$ and every $x$, or equivalently

    $$
        |b\rangle|x\rangle \mapsto |b\rangle e^{- (-1)^b i x/2} |x\rangle
    $$

    for every $b \in \{0, 1\}$ and every $x$.

    To apply an $\text{Rz}(\theta) = e^{-i Z \theta/2}$, the angle register $x$ should store $\theta/(4\pi)$.

    Args:
        angle_dtype: Data type for the `angle_data` register.
        phasegrad_dtype: Data type for the phase gradient register.

    Registers:
        q: The qubit to apply Rz on.
        angle: The rotation angle in radians.
        phase_grad: The phase gradient register of sufficient width.

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization](https://arxiv.org/abs/2007.07391).
        Section II-C: Oracles for phasing by cost function.
        Appendix A: Addition for controlled rotations.
    """

    angle_dtype: QDType
    phasegrad_dtype: QDType

    @property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            q=QBit(), angle=self.angle_dtype, phase_grad=self.phasegrad_dtype
        )

    @property
    def _angle_int_dtype(self) -> QUInt:
        return QUInt(self.angle_dtype.num_qubits)

    @property
    def _phasegrad_int_dtype(self) -> QUInt:
        return QUInt(self.phasegrad_dtype.num_qubits)

    def build_composite_bloq(
        self, bb: BloqBuilder, q: Soquet, angle: Soquet, phase_grad: Soquet
    ) -> dict[str, SoquetT]:
        angle = bb.add(Cast(self.angle_dtype, self._angle_int_dtype), reg=angle)
        phase_grad = bb.add(Cast(self.phasegrad_dtype, self._phasegrad_int_dtype), reg=phase_grad)

        q, angle, phase_grad = bb.add(
            ControlledAddOrSubtract(self._angle_int_dtype, self._phasegrad_int_dtype),
            ctrl=q,
            a=angle,
            b=phase_grad,
        )

        angle = bb.add(Cast(self._angle_int_dtype, self.angle_dtype), reg=angle)
        phase_grad = bb.add(Cast(self._phasegrad_int_dtype, self.phasegrad_dtype), reg=phase_grad)

        return {'q': q, 'angle': angle, 'phase_grad': phase_grad}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {ControlledAddOrSubtract(self._angle_int_dtype, self._phasegrad_int_dtype): 1}


@bloq_example
def _rz_via_phase_gradient() -> RzViaPhaseGradient:
    from qualtran import QFxp

    rz_via_phase_gradient = RzViaPhaseGradient(angle_dtype=QFxp(4, 4), phasegrad_dtype=QFxp(4, 4))
    return rz_via_phase_gradient
