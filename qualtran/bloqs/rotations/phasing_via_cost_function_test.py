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
from typing import Dict

import attrs
import cirq
import numpy as np
import pytest

from qualtran import Bloq, BloqBuilder, GateWithRegisters, Register, Signature, SoquetT
from qualtran._infra.data_types import QFxp
from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute
from qualtran.bloqs.arithmetic.multiplication import Square
from qualtran.bloqs.basic_gates import Hadamard
from qualtran.bloqs.on_each import OnEach
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from qualtran.bloqs.rotations.phasing_via_cost_function import (
    PhaseOraclePhaseGradient,
    PhaseOracleZPow,
    PhasingViaCostFunction,
)
from qualtran.cirq_interop.testing import GateHelper
from qualtran.testing import assert_valid_bloq_decomposition


@attrs.frozen
class TestHammingWeightPhasing(GateWithRegisters):
    bitsize: int
    exponent: float
    eps: float
    use_phase_gradient: bool = False

    @property
    def signature(self) -> 'Signature':
        return Signature.build(x=self.bitsize)

    @property
    def cost_reg(self) -> Register:
        return Register('out', QFxp(self.bitsize.bit_length(), 0, signed=False))

    @property
    def phase_oracle(self) -> Bloq:
        if self.use_phase_gradient:
            return PhaseOraclePhaseGradient(self.cost_reg, self.exponent / 2, self.eps)
        else:
            return PhaseOracleZPow(self.cost_reg, self.exponent / 2, self.eps)

    @property
    def cost_eval_oracle(self) -> Bloq:
        return HammingWeightCompute(self.bitsize)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        if self.use_phase_gradient:
            soqs['phase_grad'] = bb.add(PhaseGradientState(self.phase_oracle.b_grad))
        soqs = bb.add_d(PhasingViaCostFunction(self.cost_eval_oracle, self.phase_oracle), **soqs)
        if self.use_phase_gradient:
            bb.add(
                PhaseGradientState(self.phase_oracle.b_grad, adjoint=True),
                phase_grad=soqs.pop('phase_grad'),
            )
        return soqs


@pytest.mark.parametrize('n', [2, 3])
@pytest.mark.parametrize('exponent, eps', [(1 / 10, 1e-4), (1.20345, 1e-4), (-4.1934341, 1e-4)])
@pytest.mark.parametrize('use_phase_gradient', [True, False])
def test_hamming_weight_phasing_using_phase_via_cost_function(
    n: int, exponent: float, eps: float, use_phase_gradient: bool
):
    gate = TestHammingWeightPhasing(n, exponent, eps, use_phase_gradient)
    assert_valid_bloq_decomposition(gate)

    gh = GateHelper(gate)
    sim = cirq.Simulator(dtype=np.complex128)
    initial_state = cirq.testing.random_superposition(dim=2**n, random_state=12345)
    gamma = exponent / 2
    phases = np.array([np.exp(1j * 2 * np.pi * gamma * x.bit_count()) for x in range(2**n)])
    expected_final_state = np.multiply(initial_state, phases)

    state_prep = cirq.Circuit(cirq.StatePreparationChannel(initial_state).on(*gh.quregs['x']))
    hw_phasing = cirq.Circuit(state_prep, gh.operation)
    hw_final_state = sim.simulate(hw_phasing).final_state_vector
    np.testing.assert_allclose(expected_final_state, hw_final_state, atol=eps)


@attrs.frozen
class TestSquarePhasing(GateWithRegisters):
    bitsize: int
    gamma: float
    eps: float
    use_phase_gradient: bool = False

    @property
    def signature(self) -> 'Signature':
        return Signature.build(a=self.bitsize)

    @property
    def cost_reg(self) -> Register:
        return Register('result', QFxp(2 * self.bitsize, 0, signed=False))

    @property
    def phase_oracle(self) -> Bloq:
        if self.use_phase_gradient:
            return PhaseOraclePhaseGradient(self.cost_reg, self.gamma, self.eps)
        else:
            return PhaseOracleZPow(self.cost_reg, self.gamma, self.eps)

    @property
    def cost_eval_oracle(self) -> Bloq:
        return Square(self.bitsize)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        if self.use_phase_gradient:
            soqs['phase_grad'] = bb.add(PhaseGradientState(self.phase_oracle.b_grad))
        soqs = bb.add_d(PhasingViaCostFunction(self.cost_eval_oracle, self.phase_oracle), **soqs)
        if self.use_phase_gradient:
            bb.add(
                PhaseGradientState(self.phase_oracle.b_grad, adjoint=True),
                phase_grad=soqs.pop('phase_grad'),
            )
        return soqs


@pytest.mark.parametrize('n', [2])
@pytest.mark.parametrize('gamma, eps', [(0.1, 1e-2), (1.20345, 1e-2), (-4.1934341, 1e-2)])
@pytest.mark.parametrize('use_phase_gradient', [True, False])
def test_square_phasing_via_phase_gradient(
    n: int, gamma: float, eps: float, use_phase_gradient: bool
):
    initial_state = np.array([1 / np.sqrt(2**n)] * 2**n)
    phases = np.array([np.exp(1j * 2 * np.pi * gamma * x**2) for x in range(2**n)])
    expected_final_state = np.multiply(initial_state, phases)

    test_bloq = TestSquarePhasing(n, gamma, eps, use_phase_gradient)
    bb = BloqBuilder()
    a = bb.allocate(n)
    a = bb.add(OnEach(n, Hadamard()), q=a)
    a = bb.add(test_bloq, a=a)
    cbloq = bb.finalize(a=a)
    hw_final_state = cbloq.tensor_contract()
    np.testing.assert_allclose(expected_final_state, hw_final_state, atol=eps)
