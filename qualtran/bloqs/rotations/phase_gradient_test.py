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
import cirq
import numpy as np
import pytest

from qualtran import BloqBuilder
from qualtran.bloqs.rotations.phase_gradient import (
    AddIntoPhaseGrad,
    AddScaledValIntoPhaseReg,
    PhaseGradientState,
    PhaseGradientUnitary,
)
from qualtran.cirq_interop.bit_tools import float_as_fixed_width_int
from qualtran.testing import assert_valid_bloq_decomposition


@pytest.mark.parametrize('n', [6, 7, 8])
def test_phase_gradient_state(n: int):
    gate = PhaseGradientState(n)
    assert_valid_bloq_decomposition(gate)
    assert_valid_bloq_decomposition(gate**-1)

    q = cirq.LineQubit.range(n)
    state_prep_cirq_circuit = cirq.Circuit(
        cirq.H.on_each(*q), cirq.PhaseGradientGate(num_qubits=n, exponent=-1).on(*q)
    )
    assert np.allclose(cirq.unitary(gate), cirq.unitary(state_prep_cirq_circuit))
    assert np.allclose(
        cirq.unitary(gate**-1), cirq.unitary(cirq.inverse(state_prep_cirq_circuit))
    )
    assert gate.t_complexity().rotations == n - 2
    assert gate.t_complexity().clifford == n + 2


@pytest.mark.parametrize('n', [6, 7, 8])
@pytest.mark.parametrize('t', [+0.124, -0.124, -1, +1])
def test_phase_gradient_state_tensor_contract(n: int, t: float):
    omega = np.exp(np.pi * 2 * t * 1j / (2**n))
    state_coefs = 1 / np.sqrt(2**n) * np.array([omega**k for k in range(2**n)])
    bloq = PhaseGradientState(n, t)
    np.testing.assert_allclose(state_coefs, bloq.tensor_contract())

    bb = BloqBuilder()
    phase_reg = bb.add(bloq)
    bb.add(PhaseGradientState(n, t, adjoint=True), phase_grad=phase_reg)
    circuit = bb.finalize()
    assert np.isclose(circuit.tensor_contract(), 1)


@pytest.mark.parametrize('n', [6, 7, 8])
@pytest.mark.parametrize('exponent', [-0.5, 1, 1 / 10])
@pytest.mark.parametrize('controlled', [True, False])
def test_phase_gradient_gate(n: int, exponent, controlled):
    bloq = PhaseGradientUnitary(n, exponent, controlled)
    assert_valid_bloq_decomposition(bloq)
    assert_valid_bloq_decomposition(bloq**-1)
    cirq_gate = cirq.PhaseGradientGate(num_qubits=n, exponent=exponent)
    if controlled:
        cirq_gate = cirq_gate.controlled()
    assert np.allclose(cirq.unitary(bloq), cirq.unitary(cirq_gate))


def test_add_into_phase_grad():
    x_bit, phase_bit = 4, 7
    bloq = AddIntoPhaseGrad(x_bit, phase_bit)
    basis_map = {}
    for x in range(2**x_bit):
        for phase_grad in range(2**phase_bit):
            phase_grad_out = (phase_grad + x) % 2**phase_bit
            # Test Bloq style classical simulation.
            assert bloq.call_classically(x=x, phase_grad=phase_grad) == (x, phase_grad_out)
            # Prepare basis states mapping for cirq-style simulation.
            input_state = int(f'{x:0{x_bit}b}' + f'{phase_grad:0{phase_bit}b}', 2)
            output_state = int(f'{x:0{x_bit}b}' + f'{phase_grad_out:0{phase_bit}b}', 2)
            basis_map[input_state] = output_state
    # Test cirq style simulation.
    num_bits = x_bit + phase_bit
    assert len(basis_map) == len(set(basis_map.values()))
    circuit = cirq.Circuit(bloq.on(*cirq.LineQubit.range(num_bits)))
    cirq.testing.assert_equivalent_computational_basis_map(basis_map, circuit)


def test_add_scaled_val_into_phase_reg():
    x_bit, phase_bit, gamma, gamma_bit = 4, 7, 0.123, 6
    bloq = AddScaledValIntoPhaseReg(x_bit, phase_bit, gamma, gamma_bit)
    gamma_fixed_width_float = float_as_fixed_width_int(gamma, gamma_bit + 1)[1] / (2**gamma_bit)
    gamma_int = float_as_fixed_width_int(gamma_fixed_width_float, phase_bit + 1)[1]
    basis_map = {}
    for x in range(2**x_bit):
        for phase_grad in range(2**phase_bit):
            phase_grad_out = (phase_grad + x * gamma_int) % 2**phase_bit
            # Test Bloq style classical simulation.
            assert bloq.call_classically(x=x, phase_grad=phase_grad) == (x, phase_grad_out)
            # Prepare basis states mapping for cirq-style simulation.
            input_state = int(f'{x:0{x_bit}b}' + f'{phase_grad:0{phase_bit}b}', 2)
            output_state = int(f'{x:0{x_bit}b}' + f'{phase_grad_out:0{phase_bit}b}', 2)
            basis_map[input_state] = output_state
    # Test cirq style simulation.
    num_bits = x_bit + phase_bit
    assert len(basis_map) == len(set(basis_map.values()))
    circuit = cirq.Circuit(bloq.on(*cirq.LineQubit.range(num_bits)))
    cirq.testing.assert_equivalent_computational_basis_map(basis_map, circuit)
