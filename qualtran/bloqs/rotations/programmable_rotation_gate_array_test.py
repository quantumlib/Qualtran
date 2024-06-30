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
from typing import Iterator, Tuple

import cirq
import numpy as np
import pytest
from numpy.typing import NDArray

from qualtran import QUInt, Register, Signature
from qualtran._infra.gate_with_registers import merge_qubits
from qualtran.bloqs.rotations.programmable_rotation_gate_array import (
    ProgrammableRotationGateArray,
    ProgrammableRotationGateArrayBase,
)
from qualtran.cirq_interop.testing import GateHelper
from qualtran.testing import assert_valid_bloq_decomposition


class CustomProgrammableRotationGateArray(ProgrammableRotationGateArrayBase):
    def interleaved_unitary(
        self, index: int, **qubit_regs: NDArray[cirq.Qid]  # type:ignore[type-var]
    ) -> cirq.Operation:
        two_qubit_ops_factory = [
            cirq.X(*qubit_regs['unrelated_target']).controlled_by(*qubit_regs['rotations_target']),
            cirq.Z(*qubit_regs['unrelated_target']).controlled_by(*qubit_regs['rotations_target']),
        ]
        return two_qubit_ops_factory[index % 2]

    @cached_property
    def interleaved_unitary_target(self) -> Tuple[Register, ...]:
        return tuple(Signature.build(unrelated_target=1))


def construct_custom_prga(*args, **kwargs) -> ProgrammableRotationGateArrayBase:
    return CustomProgrammableRotationGateArray(*args, **kwargs)


def construct_prga_with_phase(*args, **kwargs) -> ProgrammableRotationGateArrayBase:
    return ProgrammableRotationGateArray(
        *args, **kwargs, interleaved_unitaries=[cirq.Z] * (len(args) - 1)
    )


def construct_prga_with_identity(*args, **kwargs) -> ProgrammableRotationGateArrayBase:
    return ProgrammableRotationGateArray(*args, **kwargs)


@pytest.mark.slow
@pytest.mark.parametrize(
    "angles", [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[3, 4, 5], [10, 11, 12]]]
)
@pytest.mark.parametrize("kappa", [*range(1, 12)])
@pytest.mark.parametrize(
    "constructor", [construct_custom_prga, construct_prga_with_phase, construct_prga_with_identity]
)
def test_programmable_rotation_gate_array(angles, kappa, constructor):
    rotation_gate = cirq.X
    programmable_rotation_gate = constructor(*angles, kappa=kappa, rotation_gate=rotation_gate)

    assert_valid_bloq_decomposition(programmable_rotation_gate)

    greedy_mm = cirq.GreedyQubitManager(prefix="_a")
    g = GateHelper(programmable_rotation_gate, context=cirq.DecompositionContext(greedy_mm))
    decomposed_circuit = cirq.Circuit(cirq.I.on_each(*g.all_qubits)) + g.decomposed_circuit
    # Get interleaved unitaries.
    interleaved_unitaries = [
        programmable_rotation_gate.interleaved_unitary(i, **g.quregs)
        for i in range(len(angles) - 1)
    ]
    # Get qubits on which rotations + unitaries act.
    rotations_and_unitary_registers = Signature(
        [
            *programmable_rotation_gate.rotations_target,
            *programmable_rotation_gate.interleaved_unitary_target,
        ]
    )
    rotations_and_unitary_qubits = merge_qubits(rotations_and_unitary_registers, **g.quregs)

    # Build circuit.
    simulator = cirq.Simulator(dtype=np.complex128)

    def rotation_ops(theta: int) -> Iterator[cirq.OP_TREE]:
        # OP-TREE to apply rotation, by integer-approximated angle `theta`, on the target register.
        for i, b in enumerate(bin(theta)[2:][::-1]):
            if b == '1':
                yield cirq.pow(rotation_gate.on(*g.quregs['rotations_target']), (1 / 2 ** (1 + i)))

    for selection_integer in range(len(angles[0])):
        # Set bits in initial_state s.t. selection register stores `selection_integer`.
        qubit_vals = {x: 0 for x in g.all_qubits}
        qubit_vals.update(
            zip(
                g.quregs['selection'],
                QUInt(g.r.get_left('selection').total_bits()).to_bits(selection_integer),
            )
        )
        initial_state = [qubit_vals[x] for x in g.all_qubits]
        # Actual circuit simulation.
        result = simulator.simulate(
            decomposed_circuit, initial_state=initial_state, qubit_order=g.all_qubits
        )
        ru_state_vector = cirq.sub_state_vector(
            result.final_state_vector,
            keep_indices=[g.all_qubits.index(q) for q in rotations_and_unitary_qubits],
        )
        # Expected circuit simulation by applying rotations directly.
        expected_circuit = cirq.Circuit(
            [
                [rotation_ops(angles[i][selection_integer]), u]
                for i, u in enumerate(interleaved_unitaries)
            ],
            rotation_ops(angles[-1][selection_integer]),
        )
        expected_ru_state_vector = simulator.simulate(
            expected_circuit, qubit_order=rotations_and_unitary_qubits
        ).final_state_vector
        # Assert that actual and expected match.
        cirq.testing.assert_allclose_up_to_global_phase(
            ru_state_vector, expected_ru_state_vector, atol=1e-8
        )
        # Assert that all other qubits are returned to their original state.
        ancilla_indices = [
            g.all_qubits.index(q) for q in g.all_qubits if q not in rotations_and_unitary_qubits
        ]
        ancilla_state_vector = cirq.sub_state_vector(
            result.final_state_vector, keep_indices=ancilla_indices
        )
        expected_ancilla_state_vector = cirq.quantum_state(
            [initial_state[x] for x in ancilla_indices],
            qid_shape=(2,) * len(ancilla_indices),
            dtype=np.complex128,
        ).state_vector()
        assert expected_ancilla_state_vector is not None
        cirq.testing.assert_allclose_up_to_global_phase(
            ancilla_state_vector, expected_ancilla_state_vector, atol=1e-8
        )


def test_programmable_rotation_gate_array_consistent():
    with pytest.raises(ValueError, match='must be of same length'):
        _ = CustomProgrammableRotationGateArray([1, 2], [1], kappa=1, rotation_gate=cirq.X)
