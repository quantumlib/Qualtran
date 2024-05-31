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
import pytest

from qualtran.bloqs import basic_gates, mcmt, rotations
from qualtran.surface_code.algorithm_summary import AlgorithmSummary
from qualtran.surface_code.magic_count import MagicCount


def test_mul():
    assert AlgorithmSummary(t_gates=9) == 3 * AlgorithmSummary(t_gates=3)

    with pytest.raises(TypeError):
        _ = complex(1, 0) * AlgorithmSummary(rotation_gates=1)  # type: ignore[operator]


def test_addition():
    with pytest.raises(TypeError):
        _ = AlgorithmSummary() + 5  # type: ignore[operator]

    a = AlgorithmSummary(
        algorithm_qubits=7,
        measurements=8,
        t_gates=8,
        toffoli_gates=9,
        rotation_gates=8,
        rotation_circuit_depth=3,
    )
    b = AlgorithmSummary(
        algorithm_qubits=4,
        measurements=1,
        t_gates=1,
        toffoli_gates=4,
        rotation_gates=2,
        rotation_circuit_depth=1,
    )
    assert a + b == AlgorithmSummary(
        algorithm_qubits=11,
        measurements=9,
        t_gates=9,
        toffoli_gates=13,
        rotation_gates=10,
        rotation_circuit_depth=4,
    )


def test_subtraction():
    with pytest.raises(TypeError):
        _ = AlgorithmSummary() - 5  # type: ignore[operator]

    a = AlgorithmSummary(
        algorithm_qubits=7,
        measurements=8,
        t_gates=8,
        toffoli_gates=9,
        rotation_gates=8,
        rotation_circuit_depth=3,
    )
    b = AlgorithmSummary(
        algorithm_qubits=4,
        measurements=1,
        t_gates=1,
        toffoli_gates=4,
        rotation_gates=2,
        rotation_circuit_depth=1,
    )
    assert a - b == AlgorithmSummary(
        algorithm_qubits=3,
        measurements=7,
        t_gates=7,
        toffoli_gates=5,
        rotation_gates=6,
        rotation_circuit_depth=2,
    )

    assert AlgorithmSummary(t_gates=1, toffoli_gates=4).to_magic_count() == MagicCount(
        n_ccz=4, n_t=1
    )

    with pytest.raises(ValueError):
        _ = AlgorithmSummary(rotation_gates=1).to_magic_count()


@pytest.mark.parametrize(
    ['bloq', 'summary'],
    [
        [basic_gates.TGate(is_adjoint=False), AlgorithmSummary(algorithm_qubits=1, t_gates=1)],
        [basic_gates.Toffoli(), AlgorithmSummary(algorithm_qubits=3, toffoli_gates=1)],
        [basic_gates.TwoBitCSwap(), AlgorithmSummary(algorithm_qubits=3, toffoli_gates=1)],
        [mcmt.And(), AlgorithmSummary(algorithm_qubits=3, toffoli_gates=1)],
        [
            basic_gates.ZPowGate(exponent=0.1, global_shift=0.0, eps=1e-11),
            AlgorithmSummary(algorithm_qubits=1, rotation_gates=1),
        ],
        [
            rotations.phase_gradient.PhaseGradientUnitary(
                bitsize=10, exponent=1, is_controlled=False, eps=1e-10
            ),
            AlgorithmSummary(algorithm_qubits=10, rotation_gates=10, rotation_circuit_depth=1),
        ],
        [
            mcmt.MultiControlPauli(cvs=(1, 1, 1), target_gate=cirq.X),
            AlgorithmSummary(
                algorithm_qubits=6, toffoli_gates=2, rotation_circuit_depth=2, measurements=2
            ),
        ],
    ],
)
def test_summary_from_bloq(bloq, summary):
    assert AlgorithmSummary.from_bloq(bloq) == summary
