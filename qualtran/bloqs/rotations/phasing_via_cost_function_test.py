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

from qualtran import Bloq, BloqBuilder, BloqError, GateWithRegisters, Register, Signature, SoquetT
from qualtran._infra.data_types import QFxp
from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute
from qualtran.bloqs.arithmetic.multiplication import Square
from qualtran.bloqs.basic_gates import Hadamard
from qualtran.bloqs.basic_gates.on_each import OnEach
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from qualtran.bloqs.rotations.phasing_via_cost_function import (
    _square_via_zpow_phasing,
    PhasingViaCostFunction,
)
from qualtran.bloqs.rotations.quantum_variable_rotation import (
    QvrInterface,
    QvrPhaseGradient,
    QvrZPow,
)
from qualtran.cirq_interop.testing import GateHelper
from qualtran.testing import assert_valid_bloq_decomposition


def test_square_via_zpow_phasing(bloq_autotester):
    bloq_autotester(_square_via_zpow_phasing)


@attrs.frozen
class TestHammingWeightPhasing(GateWithRegisters):
    bitsize: int
    normalize: bool
    exponent: float
    eps: float
    use_phase_gradient: bool = False

    @property
    def signature(self) -> 'Signature':
        return Signature.build(x=self.bitsize)

    @property
    def cost_reg(self) -> Register:
        return Register(
            'out',
            QFxp(
                self.bitsize.bit_length(),
                self.bitsize.bit_length() * int(self.normalize),
                signed=False,
            ),
        )

    @property
    def phase_gradient_oracle(self) -> QvrPhaseGradient:
        return QvrPhaseGradient(self.cost_reg, self.exponent / 2, self.eps)

    @property
    def phase_oracle(self) -> QvrInterface:
        if self.use_phase_gradient:
            return self.phase_gradient_oracle
        else:
            return QvrZPow(self.cost_reg, self.exponent / 2, self.eps)

    @property
    def cost_eval_oracle(self) -> Bloq:
        return HammingWeightCompute(self.bitsize)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        if self.use_phase_gradient:
            soqs['phase_grad'] = bb.add(PhaseGradientState(int(self.phase_gradient_oracle.b_grad)))
        soqs = bb.add_d(PhasingViaCostFunction(self.cost_eval_oracle, self.phase_oracle), **soqs)
        if self.use_phase_gradient:
            bb.add(
                PhaseGradientState(int(self.phase_gradient_oracle.b_grad)).adjoint(),
                phase_grad=soqs.pop('phase_grad'),
            )
        return soqs


@pytest.mark.slow
@pytest.mark.parametrize('normalize_cost_function', [True, False])
@pytest.mark.parametrize('use_phase_gradient', [True, False])
@pytest.mark.parametrize('exponent, eps', [(1 / 10, 5e-4), (1.20345, 5e-4), (-1.1934341, 5e-4)])
@pytest.mark.parametrize('n', [2, 3])
def test_hamming_weight_phasing_using_phase_via_cost_function(
    n: int, exponent: float, eps: float, use_phase_gradient: bool, normalize_cost_function: bool
):
    cost_reg_size = 2 ** n.bit_length()
    normalization_factor = 1 if normalize_cost_function else cost_reg_size
    sim = cirq.Simulator(dtype=np.complex128)
    initial_state = cirq.testing.random_superposition(dim=2**n, random_state=12345)
    gamma = exponent * normalization_factor / 2
    phases = np.array(
        [np.exp(1j * 2 * np.pi * gamma * x.bit_count() / cost_reg_size) for x in range(2**n)]
    )
    expected_final_state = np.multiply(initial_state, phases)
    gate_one = TestHammingWeightPhasing(
        n, True, exponent * normalization_factor, eps, use_phase_gradient
    )
    gate_two = TestHammingWeightPhasing(
        n, False, exponent * normalization_factor / cost_reg_size, eps, use_phase_gradient
    )
    for gate in [gate_one, gate_two]:
        assert_valid_bloq_decomposition(gate)
        gh = GateHelper(gate)
        state_prep = cirq.Circuit(cirq.StatePreparationChannel(initial_state).on(*gh.quregs['x']))
        hw_phasing = cirq.Circuit(state_prep, gh.operation)
        hw_final_state = sim.simulate(hw_phasing).final_state_vector
        np.testing.assert_allclose(expected_final_state, hw_final_state, atol=eps)


@attrs.frozen
class TestSquarePhasing(GateWithRegisters):
    bitsize: int
    normalize: bool
    gamma: float
    eps: float
    use_phase_gradient: bool = False

    @property
    def signature(self) -> 'Signature':
        return Signature.build(a=self.bitsize)

    @property
    def cost_reg(self) -> Register:
        return Register(
            'result', QFxp(2 * self.bitsize, 2 * self.bitsize * int(self.normalize), signed=False)
        )

    @property
    def phase_gradient_oracle(self) -> QvrPhaseGradient:
        return QvrPhaseGradient(self.cost_reg, self.gamma, self.eps)

    @property
    def phase_oracle(self) -> QvrInterface:
        if self.use_phase_gradient:
            return self.phase_gradient_oracle
        else:
            return QvrZPow(self.cost_reg, self.gamma, self.eps)

    @property
    def cost_eval_oracle(self) -> Bloq:
        return Square(self.bitsize)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        if self.use_phase_gradient:
            soqs['phase_grad'] = bb.add(PhaseGradientState(int(self.phase_gradient_oracle.b_grad)))
        soqs = bb.add_d(PhasingViaCostFunction(self.cost_eval_oracle, self.phase_oracle), **soqs)
        if self.use_phase_gradient:
            bb.add(
                PhaseGradientState(int(self.phase_gradient_oracle.b_grad)).adjoint(),
                phase_grad=soqs.pop('phase_grad'),
            )
        return soqs


@pytest.mark.slow
@pytest.mark.parametrize('normalize_cost_function', [True, False])
@pytest.mark.parametrize('use_phase_gradient', [True, False])
@pytest.mark.parametrize('gamma, eps', [(0.1, 5e-2), (1.20345, 5e-2), (-1.1934341, 5e-2)])
@pytest.mark.parametrize('n', [2])
def test_square_phasing_via_phase_gradient(
    n: int, gamma: float, eps: float, use_phase_gradient: bool, normalize_cost_function: bool
):
    initial_state = np.array([1 / np.sqrt(2**n)] * 2**n)
    normalization_factor = 1 if normalize_cost_function else 4**n
    phases = np.array(
        [
            np.exp(1j * 2 * np.pi * gamma * x**2 * normalization_factor / 4**n)
            for x in range(2**n)
        ]
    )
    expected_final_state = np.multiply(initial_state, phases)
    test_bloq_one = TestSquarePhasing(
        n, True, gamma * normalization_factor, eps, use_phase_gradient
    )
    test_bloq_two = TestSquarePhasing(
        n, False, gamma * normalization_factor / (4**n), eps, use_phase_gradient
    )
    for test_bloq in [test_bloq_one, test_bloq_two]:
        bb = BloqBuilder()
        a = bb.allocate(n)
        a = bb.add(OnEach(n, Hadamard()), q=a)
        a = bb.add(test_bloq, a=a)
        cbloq = bb.finalize(a=a)
        try:
            flat_cbloq = cbloq.flatten()
        except BloqError:
            pytest.xfail("https://github.com/quantumlib/Qualtran/issues/1069")
        hw_final_state = flat_cbloq.tensor_contract()
        np.testing.assert_allclose(expected_final_state, hw_final_state, atol=eps)
