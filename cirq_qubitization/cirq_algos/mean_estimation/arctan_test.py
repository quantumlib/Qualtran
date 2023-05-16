import cirq
import numpy as np
import pytest

from cirq_qubitization.cirq_algos.mean_estimation.arctan import ArcTan
from cirq_qubitization.bit_tools import iter_bits_fixed_point


@pytest.mark.parametrize('selection_bitsize', [3, 4])
@pytest.mark.parametrize('target_bitsize', [3, 5, 6])
def test_arctan(selection_bitsize, target_bitsize):
    gate = ArcTan(selection_bitsize, target_bitsize)
    maps = {}
    for x in range(2**selection_bitsize):
        inp = f'0b_{x:0{selection_bitsize}b}_0_{0:0{target_bitsize}b}'
        y = -2 * np.arctan(x) / np.pi
        sign, y_bin = int(np.sign(y)), iter_bits_fixed_point(abs(y), target_bitsize)
        sign_str = '01'[sign < 0]
        y_bin_str = ''.join(str(b) for b in y_bin)
        out = f'0b_{x:0{selection_bitsize}b}_{sign_str}_{y_bin_str}'
        maps[int(inp, 2)] = int(out, 2)
    num_qubits = gate.num_qubits()
    op = gate.on(*cirq.LineQubit.range(num_qubits))
    circuit = cirq.Circuit(op)
    cirq.testing.assert_equivalent_computational_basis_map(maps, circuit)
    circuit += op**-1
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit.unitary(), np.diag([1] * 2**num_qubits), atol=1e-8
    )
