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

from qualtran.bloqs.mean_estimation.arctan import ArcTan
from qualtran.cirq_interop.bit_tools import iter_bits_fixed_point
from qualtran.cirq_interop.t_complexity_protocol import t_complexity, TComplexity


@pytest.mark.parametrize('selection_bitsize', [3, 4])
@pytest.mark.parametrize('target_bitsize', [3, 5, 6])
def test_arctan(selection_bitsize, target_bitsize):
    gate = ArcTan(selection_bitsize, target_bitsize)
    maps = {}
    for x in range(2**selection_bitsize):
        inp = f'0b_{x:0{selection_bitsize}b}_0_{0:0{target_bitsize}b}'
        y = -2 * np.arctan(x) / np.pi
        bits = [*iter_bits_fixed_point(y, target_bitsize + 1, signed=True)]
        sign, y_bin = bits[0], bits[1:]
        y_bin_str = ''.join(str(b) for b in y_bin)
        out = f'0b_{x:0{selection_bitsize}b}_{sign}_{y_bin_str}'
        maps[int(inp, 2)] = int(out, 2)
    num_qubits = gate.num_qubits()
    op = gate.on(*cirq.LineQubit.range(num_qubits))
    circuit = cirq.Circuit(op)
    cirq.testing.assert_equivalent_computational_basis_map(maps, circuit)
    # missing cirq stubs
    circuit += op**-1  # type: ignore[operator]
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit.unitary(), np.diag([1] * 2**num_qubits), atol=1e-8
    )


def test_arctan_t_complexity():
    gate = ArcTan(4, 5)
    assert t_complexity(gate) == TComplexity(t=5)
