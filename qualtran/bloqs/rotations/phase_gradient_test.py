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
from typing import Optional

import cirq
import numpy as np
import pytest

from qualtran import BloqBuilder, QFxp
from qualtran.bloqs.rotations.phase_gradient import (
    _phase_gradient_unitary_symbolic,
    AddIntoPhaseGrad,
    AddScaledValIntoPhaseReg,
    PhaseGradientState,
    PhaseGradientUnitary,
)
from qualtran.cirq_interop.testing import GateHelper
from qualtran.testing import assert_valid_bloq_decomposition


@pytest.mark.parametrize('n', [6, 7, 8])
def test_phase_gradient_state(n: int):
    gate = PhaseGradientState(n, eps=0)
    assert_valid_bloq_decomposition(gate)
    assert_valid_bloq_decomposition(gate.adjoint())

    q = cirq.LineQubit.range(n)
    state_prep_cirq_circuit = cirq.Circuit(
        cirq.H.on_each(*q), cirq.PhaseGradientGate(num_qubits=n, exponent=-1).on(*q)
    )
    np.testing.assert_allclose(cirq.unitary(gate), cirq.unitary(state_prep_cirq_circuit))
    np.testing.assert_allclose(
        cirq.unitary(gate.adjoint()), cirq.unitary(cirq.inverse(state_prep_cirq_circuit))
    )
    assert gate.t_complexity().t == 1  # one of the rotations is a T gate
    assert gate.t_complexity().rotations == n - 3
    assert gate.t_complexity().clifford == n + 2  # two of the rotations are clifford


@pytest.mark.parametrize('n', [6, 7, 8])
@pytest.mark.parametrize('t', [+0.124, -0.124, -1, +1])
def test_phase_gradient_state_tensor_contract(n: int, t: float):
    omega = np.exp(np.pi * 2 * t * 1j / (2**n))
    state_coefs = 1 / np.sqrt(2**n) * np.array([omega**k for k in range(2**n)])
    bloq = PhaseGradientState(n, t)
    np.testing.assert_allclose(state_coefs, bloq.tensor_contract())

    bb = BloqBuilder()
    phase_reg = bb.add(bloq)
    bb.add(PhaseGradientState(n, t).adjoint(), phase_grad=phase_reg)
    circuit = bb.finalize()
    assert np.isclose(circuit.tensor_contract(), 1)


@pytest.mark.parametrize('n', [6, 7, 8])
@pytest.mark.parametrize('exponent', [-0.5, 1, 1 / 10])
@pytest.mark.parametrize('controlled', [True, False])
def test_phase_gradient_gate(n: int, exponent, controlled):
    eps = 1e-4
    bloq = PhaseGradientUnitary(n, exponent, controlled, eps=eps)
    assert_valid_bloq_decomposition(bloq)
    assert_valid_bloq_decomposition(bloq**-1)
    cirq_gate: cirq.Gate = cirq.PhaseGradientGate(num_qubits=n, exponent=exponent)
    if controlled:
        cirq_gate = cirq_gate.controlled()
    assert np.allclose(cirq.unitary(bloq), cirq.unitary(cirq_gate), atol=eps)


@pytest.mark.parametrize("controlled_by", [None, 0, 1])
def test_add_into_phase_grad_classical_sim(controlled_by: Optional[int]):
    from qualtran.bloqs.rotations.phase_gradient import _fxp

    x_bit, phase_bit = 4, 7
    bloq = AddIntoPhaseGrad(x_bit, phase_bit, controlled_by=controlled_by)

    for x in range(2**x_bit):
        for phase_grad in range(2**phase_bit):
            phase_fxp = _fxp(phase_grad / 2**phase_bit, phase_bit)
            x_fxp = _fxp(x / 2**x_bit, x_bit).like(phase_fxp)
            phase_grad_out = int((phase_fxp + x_fxp).astype(float) * 2**phase_bit)

            if controlled_by is None:
                assert bloq.call_classically(x=x, phase_grad=phase_grad) == (x, phase_grad_out)
            else:
                for control in [0, 1]:
                    phase_grad_out_ctrld = (
                        phase_grad_out if controlled_by == control else phase_grad
                    )
                    assert bloq.call_classically(ctrl=control, x=x, phase_grad=phase_grad) == (
                        control,
                        x,
                        phase_grad_out_ctrld,
                    )


@pytest.mark.parametrize("controlled_by", [None, 0, 1])
def test_add_into_phase_grad_unitary(controlled_by: Optional[int]):
    from qualtran.bloqs.rotations.phase_gradient import _fxp

    x_bit, phase_bit = 4, 7
    bloq = AddIntoPhaseGrad(x_bit, phase_bit, controlled_by=controlled_by)

    # compute expected unitary manually
    num_bits = x_bit + phase_bit + (1 if controlled_by is not None else 0)
    expected_unitary = np.zeros((2**num_bits, 2**num_bits))

    for x in range(2**x_bit):
        for phase_grad in range(2**phase_bit):
            phase_fxp = _fxp(phase_grad / 2**phase_bit, phase_bit)
            x_fxp = _fxp(x / 2**x_bit, x_bit).like(phase_fxp)
            phase_grad_out = int((phase_fxp + x_fxp).astype(float) * 2**phase_bit)

            if controlled_by is None:
                input_state = int(f'{x:0{x_bit}b}' + f'{phase_grad:0{phase_bit}b}', 2)
                output_state = int(f'{x:0{x_bit}b}' + f'{phase_grad_out:0{phase_bit}b}', 2)
                expected_unitary[output_state, input_state] = 1
            else:
                for control in [0, 1]:
                    phase_grad_out_ctrld = (
                        phase_grad_out if controlled_by == control else phase_grad
                    )
                    input_state = int(
                        f'{control}' + f'{x:0{x_bit}b}' + f'{phase_grad:0{phase_bit}b}', 2
                    )
                    output_state = int(
                        f'{control}' + f'{x:0{x_bit}b}' + f'{phase_grad_out_ctrld:0{phase_bit}b}', 2
                    )
                    expected_unitary[output_state, input_state] = 1

    # verify bloq unitary
    circuit = cirq.Circuit(bloq.on(*cirq.LineQubit.range(num_bits)))
    np.testing.assert_allclose(circuit.unitary(), expected_unitary, atol=1e-8)


def test_add_into_phase_grad_t_complexity():
    x_bit, phase_bit = 4, 7
    bloq = AddIntoPhaseGrad(x_bit, phase_bit)
    ((toffoli, n),) = bloq.bloq_counts().items()
    assert bloq.t_complexity() == n * toffoli.t_complexity()


_ADD_SCALED_VAL_INTO_PHASE_REG_EXAMPLES: list[AddScaledValIntoPhaseReg] = [
    AddScaledValIntoPhaseReg.from_bitsize(4, 7, 0.123, 6),
    AddScaledValIntoPhaseReg.from_bitsize(2, 8, 1.3868682, 8),
    AddScaledValIntoPhaseReg.from_bitsize(4, 9, -9.0949456, 5),
    AddScaledValIntoPhaseReg.from_bitsize(6, 4, 2.5, 2),
    AddScaledValIntoPhaseReg(QFxp(4, 0, signed=False), 4, 1.3868682, QFxp(8, 7, signed=False)),
]


@pytest.mark.parametrize(
    'bloq',
    [
        pytest.param(bloq, marks=pytest.mark.slow if bloq.signature.n_qubits() > 10 else ())
        for bloq in _ADD_SCALED_VAL_INTO_PHASE_REG_EXAMPLES
    ],
)
def test_add_scaled_val_into_phase_reg_classical_sim(bloq: AddScaledValIntoPhaseReg):
    cbloq = bloq.decompose_bloq()
    for x in range(2**bloq.x_dtype.bitsize):
        for phase_grad in range(2**bloq.phase_bitsize):
            d = {'x': x, 'phase_grad': phase_grad}
            c1 = bloq.on_classical_vals(**d)
            c2 = cbloq.on_classical_vals(**d)
            assert c1 == c2, f'{d=}, {c1=}, {c2=}'


@pytest.mark.parametrize(
    'bloq',
    [
        pytest.param(bloq, marks=pytest.mark.slow if bloq.signature.n_qubits() > 12 else ())
        for bloq in _ADD_SCALED_VAL_INTO_PHASE_REG_EXAMPLES
    ],
)
def test_add_scaled_val_into_phase_reg_unitary(bloq: AddScaledValIntoPhaseReg):
    bloq_unitary = cirq.unitary(bloq)
    op = GateHelper(bloq).operation
    circuit = cirq.Circuit(cirq.I.on_each(*op.qubits), cirq.decompose_once(op))
    decomposed_unitary = circuit.unitary(qubit_order=op.qubits)
    np.testing.assert_allclose(bloq_unitary, decomposed_unitary)


@pytest.mark.parametrize('bloq', _ADD_SCALED_VAL_INTO_PHASE_REG_EXAMPLES)
def test_add_scaled_val_into_phase_reg_t_complexity(bloq):
    ((add_into_phase, n),) = bloq.bloq_counts().items()
    assert bloq.t_complexity() == n * add_into_phase.t_complexity()


def test_phase_gradient_unitary_symbolic_auto(bloq_autotester):
    bloq_autotester(_phase_gradient_unitary_symbolic)
