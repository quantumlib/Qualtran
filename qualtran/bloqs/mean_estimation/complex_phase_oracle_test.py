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

import math
from functools import cached_property
from typing import Optional, Tuple

import cirq
import numpy as np
import pytest
from attrs import frozen

from qualtran import QAny, QBit, QFxp, Register
from qualtran.bloqs.mean_estimation.complex_phase_oracle import ComplexPhaseOracle
from qualtran.bloqs.multiplexers.select_base import SelectOracle
from qualtran.cirq_interop import testing as cq_testing
from qualtran.testing import assert_valid_bloq_decomposition


@frozen
class ExampleSelect(SelectOracle):
    bitsize: int
    control_val: Optional[int] = None

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return () if self.control_val is None else (Register('control', QBit()),)

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (Register('selection', QAny(self.bitsize)),)

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register('target', QAny(self.bitsize)),)

    def decompose_from_registers(self, context, selection, target):
        yield [cirq.CNOT(s, t) for s, t in zip(selection, target)]


@pytest.mark.slow
@pytest.mark.parametrize('bitsize', [2, 3, 4, 5])
@pytest.mark.parametrize('arctan_bitsize', [5, 6, 7])
def test_phase_oracle(bitsize: int, arctan_bitsize: int):
    phase_oracle = ComplexPhaseOracle(ExampleSelect(bitsize), arctan_bitsize)
    g = cq_testing.GateHelper(phase_oracle)

    assert_valid_bloq_decomposition(phase_oracle)

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
        output_bits = QFxp(arctan_bitsize, arctan_bitsize).to_bits(
            QFxp(arctan_bitsize, arctan_bitsize).to_fixed_width_int(
                np.abs(output_val), require_exact=False
            )
        )
        approx_val = np.sign(output_val) * math.fsum(
            [b * (1 / 2 ** (1 + i)) for i, b in enumerate(output_bits)]
        )

        assert math.isclose(output_val, approx_val, abs_tol=1 / 2**bitsize), output_bits

        y = np.exp(1j * approx_val * np.pi) / np.sqrt(2**bitsize)
        assert np.isclose(prepared_state[x], y)


def test_phase_oracle_consistent_protocols():
    bitsize, arctan_bitsize = 3, 5
    gate = ComplexPhaseOracle(ExampleSelect(bitsize, 1), arctan_bitsize)
    expected_symbols = ('@',) + ('ROTy',) * bitsize
    assert cirq.circuit_diagram_info(gate).wire_symbols == expected_symbols
