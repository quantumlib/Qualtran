import cirq
import numpy as np
import pytest

from typing import Sequence, Type
from functools import cached_property

import cirq_qubitization as cq
from cirq_qubitization.bit_tools import iter_bits


class CustomProgrammableRotationGateArray(cq.ProgrammableRotationGateArrayBase):
    def interleaved_unitary(self, index: int, **qubit_regs: Sequence[cirq.Qid]) -> cirq.Operation:
        two_qubit_ops_factory = [
            cirq.X(*qubit_regs['unrelated_target']).controlled_by(*qubit_regs['rotations_target']),
            cirq.Z(*qubit_regs['unrelated_target']).controlled_by(*qubit_regs['rotations_target']),
        ]
        return two_qubit_ops_factory[index % 2]

    @cached_property
    def interleaved_unitary_target(self) -> cq.Registers:
        return cq.Registers.build(unrelated_target=1)


def construct_programmable_rotation_gate(
    gate_type: Type[cq.ProgrammableRotationGateArrayBase],
    angles: Sequence[Sequence[int]],
    kappa: int,
    rotation_gate: cirq.Gate,
) -> cq.ProgrammableRotationGateArrayBase:
    if gate_type == cq.ProgrammableRotationGateArray:
        return gate_type(
            *angles,
            kappa=kappa,
            rotation_gate=rotation_gate,
            interleaved_unitaries=[cirq.Z] * (len(angles) - 1),
        )
    if gate_type == CustomProgrammableRotationGateArray:
        return gate_type(*angles, kappa=kappa, rotation_gate=rotation_gate)


@pytest.mark.parametrize(
    "angles",
    [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[3, 4, 5], [10, 11, 12]]],
)
@pytest.mark.parametrize("kappa", [*range(1, 12)])
@pytest.mark.parametrize(
    "gate_type", [CustomProgrammableRotationGateArray, cq.ProgrammableRotationGateArray]
)
def test_programmable_rotation_gate_array(angles, kappa, gate_type):
    rotation_gate = cirq.X
    programmable_rotation_gate = construct_programmable_rotation_gate(
        gate_type, angles, kappa, rotation_gate
    )
    qubit_regs = programmable_rotation_gate.registers.get_named_qubits()
    all_qubits = programmable_rotation_gate.registers.merge_qubits(**qubit_regs)
    # Get interleaved unitaries.
    interleaved_unitaries = [
        programmable_rotation_gate.interleaved_unitary(i, **qubit_regs)
        for i in range(len(angles) - 1)
    ]
    # Get qubits on which rotations + unitaries act.
    rotations_and_unitary_registers = cq.Registers(
        [
            *programmable_rotation_gate.rotations_target,
            *programmable_rotation_gate.interleaved_unitary_target,
        ]
    )
    rotations_and_unitary_qubits = rotations_and_unitary_registers.merge_qubits(**qubit_regs)
    # Set qubit order s.t. rotations_and_unitary_qubits are at the beginning of the list.
    for q in rotations_and_unitary_qubits:
        all_qubits.insert(0, all_qubits.pop(all_qubits.index(q)))
    # Build circuit.
    circuit = cirq.Circuit(programmable_rotation_gate.on_registers(**qubit_regs))
    simulator = cirq.Simulator(dtype=np.complex128)

    selection = qubit_regs["selection"]

    def rotation_ops(theta: int) -> cirq.OP_TREE:
        # OP-TREE to apply rotation, by integer-approximated angle `theta`, on the target register.
        for i, b in enumerate(bin(theta)[2:][::-1]):
            if b == '1':
                yield cirq.pow(
                    rotation_gate.on(*qubit_regs['rotations_target']), (1 / 2 ** (1 + i))
                )

    for selection_integer in range(programmable_rotation_gate.iteration_length):
        # Set bits in initial_state s.t. selection register stores `selection_integer`.
        svals = list(iter_bits(selection_integer, len(selection)))
        qubit_vals = {x: 0 for x in all_qubits}
        qubit_vals.update({s: sval for s, sval in zip(selection, svals)})
        initial_state = [qubit_vals[x] for x in all_qubits]
        # Actual circuit simulation.
        result = simulator.simulate(circuit, initial_state=initial_state, qubit_order=all_qubits)
        ru_len = len(rotations_and_unitary_qubits)
        ru_state_vector = cirq.sub_state_vector(
            result.final_state_vector, keep_indices=[*range(ru_len)]
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
            expected_circuit, qubit_order=all_qubits[:ru_len]
        ).final_state_vector
        # Assert that actual and expected match.
        cirq.testing.assert_allclose_up_to_global_phase(
            ru_state_vector, expected_ru_state_vector, atol=1e-8
        )
        # Assert that all other qubits are returned to their original state.
        ancilla_state_vector = cirq.sub_state_vector(
            result.final_state_vector, keep_indices=[*range(ru_len, len(all_qubits))]
        )
        expected_ancilla_state_vector = cirq.quantum_state(
            initial_state[ru_len:], qid_shape=(2,) * (len(all_qubits) - ru_len)
        ).state_vector()
        cirq.testing.assert_allclose_up_to_global_phase(
            ancilla_state_vector, expected_ancilla_state_vector, atol=1e-8
        )
