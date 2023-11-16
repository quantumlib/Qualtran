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

from qualtran.bloqs.arithmetic import HammingWeightCompute
from qualtran.cirq_interop.bit_tools import iter_bits
from qualtran.cirq_interop.testing import (
    assert_circuit_inp_out_cirqsim,
    assert_decompose_is_consistent_with_t_complexity,
    GateHelper,
)
from qualtran.testing import assert_valid_bloq_decomposition


@pytest.mark.parametrize('bitsize', [3, 4, 5])
def test_hamming_weight_compute(bitsize: int):
    gate = HammingWeightCompute(bitsize=bitsize)
    gate_inv = gate**-1

    assert_decompose_is_consistent_with_t_complexity(gate)
    assert_decompose_is_consistent_with_t_complexity(gate_inv)
    assert_valid_bloq_decomposition(gate)
    assert_valid_bloq_decomposition(gate_inv)

    junk_bitsize = bitsize - bitsize.bit_count()
    out_bitsize = bitsize.bit_length()
    sim = cirq.Simulator()
    op = GateHelper(gate).operation
    circuit = cirq.Circuit(cirq.decompose_once(op))
    circuit_with_inv = circuit + cirq.Circuit(cirq.decompose_once(op**-1))
    qubit_order = sorted(circuit_with_inv.all_qubits())
    for inp in range(2**bitsize):
        input_state = [0] * (junk_bitsize + out_bitsize) + list(iter_bits(inp, bitsize))
        result = sim.simulate(circuit, initial_state=input_state).dirac_notation()
        actual_bits = result[1 + junk_bitsize : 1 + junk_bitsize + out_bitsize]
        assert actual_bits == f'{inp.bit_count():0{out_bitsize}b}'
        assert_circuit_inp_out_cirqsim(circuit_with_inv, qubit_order, input_state, input_state)
