import itertools
from typing import List

import cirq
import pytest

import cirq_qubitization
import cirq_qubitization.cirq_infra.testing as cq_testing
from cirq_qubitization import bit_tools


def test_less_than_gate():
    circuit = cirq.Circuit(
        cirq_qubitization.LessThanGate([2, 2, 2], 5).on(*cirq.LineQubit.range(4))
    )
    maps = {
        0b_000_0: 0b_000_1,
        0b_000_1: 0b_000_0,
        0b_001_0: 0b_001_1,
        0b_001_1: 0b_001_0,
        0b_010_0: 0b_010_1,
        0b_010_1: 0b_010_0,
        0b_011_0: 0b_011_1,
        0b_011_1: 0b_011_0,
        0b_100_0: 0b_100_1,
        0b_100_1: 0b_100_0,
        0b_101_0: 0b_101_0,
        0b_101_1: 0b_101_1,
        0b_110_0: 0b_110_0,
        0b_110_1: 0b_110_1,
        0b_111_0: 0b_111_0,
        0b_111_1: 0b_111_1,
    }
    cirq.testing.assert_equivalent_computational_basis_map(maps, circuit)


def test_multi_in_less_equal_than_gate():
    circuit = cirq.Circuit(
        cirq_qubitization.LessThanEqualGate([2, 2, 2], [2, 2, 2]).on(*cirq.LineQubit.range(7))
    )
    maps = {}
    for in1, in2 in itertools.product(range(2**3), repeat=2):
        for target_reg_val in range(2):
            target_bin = bin(target_reg_val)[2:]
            in1_bin = format(in1, '03b')
            in2_bin = format(in2, '03b')
            out_bin = bin(target_reg_val ^ (in1 <= in2))[2:]
            true_out_int = target_reg_val ^ (in1 <= in2)
            input_int = int(in1_bin + in2_bin + target_bin, 2)
            output_int = int(in1_bin + in2_bin + out_bin, 2)
            assert true_out_int == int(out_bin, 2)
            maps[input_int] = output_int
    cirq.testing.assert_equivalent_computational_basis_map(maps, circuit)


@pytest.mark.parametrize("bits", [*range(8)])
@pytest.mark.parametrize("val", [3, 5, 7, 8, 9])
def test_decompose_less_than_gate(bits: List[int], val: int):
    qubit_states = list(bit_tools.iter_bits(bits, 3))
    circuit = cirq.Circuit(
        cirq.decompose_once(
            cirq_qubitization.LessThanGate([2, 2, 2], val).on(*cirq.LineQubit.range(4))
        )
    )
    if val < 8:
        initial_state = [0] * 4 + qubit_states + [0]
        output_state = [0] * 4 + qubit_states + [int(bits < val)]
    else:
        # When val >= 2**number_qubits the decomposition doesn't create any ancillas since the answer is always 1.
        initial_state = [0]
        output_state = [1]
    cq_testing.assert_circuit_inp_out_cirqsim(
        circuit, sorted(circuit.all_qubits()), initial_state, output_state
    )


@pytest.mark.parametrize("n", [*range(2, 5)])
@pytest.mark.parametrize("val", [3, 4, 5, 7, 8, 9])
def test_t_complexity(n: int, val: int):
    g = cirq_qubitization.LessThanGate(n * [2], val)
    cq_testing.assert_decompose_is_consistent_with_t_complexity(g)
