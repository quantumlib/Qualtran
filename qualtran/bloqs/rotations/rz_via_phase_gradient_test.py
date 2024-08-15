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
import numpy as np
import pytest
import sympy
from attrs import frozen

from qualtran import Bloq, BloqBuilder, QBit, QFxp, QUInt, Signature, Soquet, SoquetT
from qualtran.bloqs.basic_gates import IntState, Rz, TGate
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from qualtran.bloqs.rotations.rz_via_phase_gradient import (
    _rz_via_phase_gradient,
    RzViaPhaseGradient,
)
from qualtran.resource_counting import BloqCount, get_cost_value


def test_examples(bloq_autotester):
    bloq_autotester(_rz_via_phase_gradient)


def test_costs():
    n = sympy.Symbol("n")
    dtype = QUInt(n)
    bloq = RzViaPhaseGradient(angle_dtype=dtype, phasegrad_dtype=dtype)
    # TODO need to improve this to `4 * n - 8` (i.e. Toffoli cost of `n - 2`)
    assert get_cost_value(bloq, BloqCount.for_gateset('t')) == {TGate(): 4 * n - 4}


@frozen
class TestRz(Bloq):
    angle: float
    phasegrad_bitsize: int = 4

    @property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(q=QBit())

    @property
    def dtype(self) -> QFxp:
        return QFxp(self.phasegrad_bitsize, self.phasegrad_bitsize)

    @property
    def load_angle_bloq(self) -> IntState:
        return IntState(
            self.dtype.to_fixed_width_int(self.angle / (4 * np.pi), require_exact=False),
            bitsize=self.phasegrad_bitsize,
        )

    @property
    def load_phasegrad_bloq(self) -> PhaseGradientState:
        return PhaseGradientState(self.phasegrad_bitsize)

    def build_composite_bloq(self, bb: 'BloqBuilder', q: Soquet) -> dict[str, 'SoquetT']:
        angle = bb.add(self.load_angle_bloq)
        phase_grad = bb.add(self.load_phasegrad_bloq)
        q, angle, phase_grad = bb.add(
            RzViaPhaseGradient(angle_dtype=self.dtype, phasegrad_dtype=self.dtype),
            q=q,
            angle=angle,
            phase_grad=phase_grad,
        )
        bb.add(self.load_angle_bloq.adjoint(), val=angle)
        bb.add(self.load_phasegrad_bloq.adjoint(), phase_grad=phase_grad)
        return {'q': q}


@pytest.mark.parametrize("bitsize", [3, 4])
def test_tensor(bitsize: int):
    for coeff in [1, 2]:
        theta = 4 * np.pi * coeff / 2**bitsize

        actual = TestRz(angle=theta, phasegrad_bitsize=bitsize).tensor_contract()
        expected = Rz(theta).tensor_contract()

        np.testing.assert_allclose(actual, expected, atol=1 / 2**bitsize)
