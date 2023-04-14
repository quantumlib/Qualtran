import itertools
from typing import List

import cirq
import pytest

import cirq_qubitization
import cirq_qubitization.cirq_infra.testing as cq_testing
from cirq_qubitization import bit_tools


def identity_map(n: int):
    return {i: i for i in range(2**n)}


def test_less_than_gate():
    qubits = cirq.LineQubit.range(4)
    op = cirq_qubitization.LessThanGate([2, 2, 2], 5).on(*qubits)
    circuit = cirq.Circuit(op)
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
    circuit += op**-1
    cirq.testing.assert_equivalent_computational_basis_map(identity_map(len(qubits)), circuit)


def test_multi_in_less_equal_than_gate():
    qubits = cirq.LineQubit.range(7)
    op = cirq_qubitization.LessThanEqualGate([2, 2, 2], [2, 2, 2]).on(*qubits)
    circuit = cirq.Circuit(op)
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
    circuit += op**-1
    cirq.testing.assert_equivalent_computational_basis_map(identity_map(len(qubits)), circuit)


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


@pytest.mark.parametrize("P,n", [(v, n) for n in range(1, 3) for v in range(1 << n)])
@pytest.mark.parametrize("Q,m", [(v, n) for n in range(1, 3) for v in range(1 << n)])
def test_decompose_less_than_gate(P: List[int], n: int, Q: List[int], m: int):
    qubit_states = list(bit_tools.iter_bits(P, n)) + list(bit_tools.iter_bits(Q, m))
    circuit = cirq.Circuit(
        cirq.decompose_once(
            cirq_qubitization.LessThanEqualGate([2] * n, [2] * m).on(
                *cirq.LineQubit.range(n + m + 1)
            )
        )
    )
    num_ancillas = len(circuit.all_qubits()) - n - m - 1
    initial_state = [0] * num_ancillas + qubit_states + [0]
    output_state = [0] * num_ancillas + qubit_states + [int(P <= Q)]
    cq_testing.assert_circuit_inp_out_cirqsim(
        circuit, sorted(circuit.all_qubits()), initial_state, output_state
    )


@pytest.mark.parametrize("n", [*range(2, 5)])
@pytest.mark.parametrize("val", [3, 4, 5, 7, 8, 9])
def test_t_complexity_less_than_gate(n: int, val: int):
    g = cirq_qubitization.LessThanGate(n * [2], val)
    cq_testing.assert_decompose_is_consistent_with_t_complexity(g)


@pytest.mark.parametrize("n", [*range(1, 5)])
@pytest.mark.parametrize("m", [*range(1, 5)])
def test_t_complexity_less_than_equal_gate(n: int, m: int):
    g = cirq_qubitization.LessThanEqualGate([2] * m, [2] * n)
    cq_testing.assert_decompose_is_consistent_with_t_complexity(g)


def test_contiguous_register_gate():
    circuit = cirq.Circuit(
        cirq_qubitization.ContiguousRegisterGate(3, 6).on(*cirq.LineQubit.range(12))
    )
    maps = {}
    for p in range(2**3):
        for q in range(p):
            inp = f'0b_{p:03b}_{q:03b}_{0:06b}'
            out = f'0b_{p:03b}_{q:03b}_{(p * (p - 1))//2 + q:06b}'
            maps[int(inp, 2)] = int(out, 2)

    cirq.testing.assert_equivalent_computational_basis_map(maps, circuit)


@pytest.mark.parametrize('n', [*range(1, 10)])
def test_contiguous_register_gate_t_complexity(n):
    gate = cirq_qubitization.ContiguousRegisterGate(n, 2 * n)
    toffoli_complexity = cirq_qubitization.t_complexity(cirq.CCNOT)
    assert cirq_qubitization.t_complexity(gate) == (n**2 + n - 1) * toffoli_complexity
