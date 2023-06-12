import math
from functools import cached_property

import cirq
import numpy as np
import pytest
from attrs import frozen

import cirq_qubitization as cq
from cirq_qubitization.cirq_algos.mean_estimation.complex_phase_oracle import ComplexPhaseOracle
from cirq_qubitization.cirq_infra import testing as cq_testing


@frozen
class DummySelect(cq.cirq_algos.SelectOracle):
    bitsize: int

    @cached_property
    def control_registers(self) -> cq.Registers:
        return cq.Registers([])

    @cached_property
    def selection_registers(self) -> cq.SelectionRegisters:
        return cq.SelectionRegisters.build(selection=(self.bitsize, 2**self.bitsize))

    @cached_property
    def target_registers(self) -> cq.Registers:
        return cq.Registers.build(target=self.bitsize)

    def decompose_from_registers(self, context, selection, target):
        yield [cirq.CNOT(s, t) for s, t in zip(selection, target)]


@pytest.mark.parametrize('bitsize', [2, 3, 4, 5])
@pytest.mark.parametrize('arctan_bitsize', [5, 6, 7])
def test_phase_oracle(bitsize: int, arctan_bitsize: int):
    phase_oracle = ComplexPhaseOracle(DummySelect(bitsize), arctan_bitsize)
    g = cq_testing.GateHelper(phase_oracle)

    # Prepare uniform superposition state on selection register and apply phase oracle.
    circuit = cirq.Circuit(cirq.H.on_each(*g.quregs['selection']))
    circuit += cirq.Circuit(cirq.decompose_once(g.operation))

    # Simulate the circut and test output.
    qubit_order = cirq.QubitOrder.explicit(g.quregs['selection'], fallback=cirq.QubitOrder.DEFAULT)
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=qubit_order)
    state_vector = result.final_state_vector
    state_vector = state_vector.reshape(2**bitsize, len(state_vector) // 2**bitsize)
    prepared_state = state_vector.sum(axis=1)
    for x in range(2**bitsize):
        output_val = -2 * np.arctan(x, dtype=np.double) / np.pi
        output_bits = [*cq.bit_tools.iter_bits_fixed_point(np.abs(output_val), arctan_bitsize)]
        approx_val = np.sign(output_val) * math.fsum(
            [b * (1 / 2 ** (1 + i)) for i, b in enumerate(output_bits)]
        )

        assert math.isclose(output_val, approx_val, abs_tol=1 / 2**bitsize), output_bits

        y = np.exp(1j * approx_val * np.pi) / np.sqrt(2**bitsize)
        assert np.isclose(prepared_state[x], y)
