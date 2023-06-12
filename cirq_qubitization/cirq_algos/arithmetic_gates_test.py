import itertools
from typing import List

import cirq
import pytest

import cirq_qubitization
import cirq_qubitization.cirq_infra.testing as cq_testing
from cirq_qubitization import bit_tools, cirq_infra
from cirq_qubitization.cirq_algos.arithmetic_gates import AdditionGate


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


@pytest.mark.parametrize("n", [*range(2, 5)])
@pytest.mark.parametrize("val", [3, 4, 5, 7, 8, 9])
def test_t_complexity(n: int, val: int):
    g = cirq_qubitization.LessThanGate(n * [2], val)
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


@pytest.mark.parametrize('a,b,num_bits', itertools.product(range(4), range(4), range(3, 5)))
def test_add(a, b, num_bits):
    num_anc = num_bits - 1
    gate = AdditionGate(num_bits)
    qubits = cirq.LineQubit.range(2 * num_bits)
    greedy_mm = cirq_infra.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits), context=context))
    ancillas = sorted(circuit.all_qubits())[-num_anc:]
    initial_state = [0] * (2 * num_bits + num_anc)
    initial_state[:num_bits] = list(bit_tools.iter_bits(a, num_bits))[::-1]
    initial_state[num_bits : 2 * num_bits] = list(bit_tools.iter_bits(b, num_bits))[::-1]
    final_state = [0] * (2 * num_bits + num_bits - 1)
    final_state[:num_bits] = list(bit_tools.iter_bits(a, num_bits))[::-1]
    final_state[num_bits : 2 * num_bits] = list(bit_tools.iter_bits(a + b, num_bits))[::-1]
    cq_testing.assert_circuit_inp_out_cirqsim(
        circuit, qubits + ancillas, initial_state, final_state
    )


@pytest.mark.parametrize('bitsize', [3])
@pytest.mark.parametrize('mod', [5, 8])
@pytest.mark.parametrize('add_val', [1, 2])
@pytest.mark.parametrize('cv', [[], [0, 1], [1, 0], [1, 1]])
def test_add_mod_n(bitsize, mod, add_val, cv):
    gate = cirq_qubitization.AddMod(bitsize, mod, add_val=add_val, cv=cv)
    maps = {}
    num_cvs = len(cv)
    for x in range(2**bitsize):
        y = (x + add_val) % mod if x < mod else x
        if not num_cvs:
            maps[x] = y
            continue
        for cb in range(2**num_cvs):
            inp = f'0b_{cb:0{num_cvs}b}_{x:0{bitsize}b}'
            if tuple(int(x) for x in f'{cb:0{num_cvs}b}') == tuple(cv):
                out = f'0b_{cb:0{num_cvs}b}_{y:0{bitsize}b}'
                maps[int(inp, 2)] = int(out, 2)
            else:
                maps[int(inp, 2)] = int(inp, 2)

    num_qubits = gate.num_qubits()
    op = gate.on(*cirq.LineQubit.range(num_qubits))
    circuit = cirq.Circuit(op)
    cirq.testing.assert_equivalent_computational_basis_map(maps, circuit)
    circuit += op**-1
    cirq.testing.assert_equivalent_computational_basis_map(identity_map(num_qubits), circuit)


def test_add_mod_n_protocols():
    with pytest.raises(ValueError, match="must be between"):
        _ = cirq_qubitization.AddMod(3, 10)
    add_one = cirq_qubitization.AddMod(3, 5, 1)
    add_two = cirq_qubitization.AddMod(3, 5, 2, cv=[1, 0])

    assert add_one == cirq_qubitization.AddMod(3, 5, 1)
    assert add_one != add_two
    assert hash(add_one) != hash(add_two)
    assert add_two.cv == (1, 0)


def test_add_truncated():
    num_bits = 3
    num_anc = num_bits - 1
    gate = AdditionGate(num_bits)
    qubits = cirq.LineQubit.range(2 * num_bits)
    greedy_mm = cirq_infra.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits), context=context))
    ancillas = sorted(circuit.all_qubits())[-num_anc:]
    all_qubits = qubits + ancillas
    # Corresponds to 2^2 + 2^2 (4 + 4 = 8 = 2^3 (needs num_bits = 4 to work properly))
    initial_state = [0, 0, 1, 0, 0, 1, 0, 0]
    # Should be 1000 (or 0001 below) but bit falls off the end
    final_state = [0, 0, 1, 0, 0, 0, 0, 0]
    # increasing number of bits yields correct value
    num_bits = 4
    num_anc = num_bits - 1
    gate = AdditionGate(num_bits)
    qubits = cirq.LineQubit.range(2 * num_bits)
    greedy_mm = cirq_infra.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits), context=context))
    ancillas = sorted(circuit.all_qubits())[-num_anc:]
    all_qubits = qubits + ancillas
    initial_state = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    final_state = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    cq_testing.assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)
    cq_testing.assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)
    num_bits = 3
    num_anc = num_bits - 1
    gate = AdditionGate(num_bits)
    qubits = cirq.LineQubit.range(2 * num_bits)
    greedy_mm = cirq_infra.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits), context=context))
    ancillas = sorted(circuit.all_qubits())[-num_anc:]
    all_qubits = qubits + ancillas
    # Corresponds to 2^2 + (2^2 + 2^1 + 2^0)  (4 + 7 = 11 = 1011 (needs num_bits = 4 to work properly))
    initial_state = [0, 0, 1, 1, 1, 1, 0, 0]
    # Should be 1011 (or 1101 below) but last two bits are lost
    final_state = [0, 0, 1, 1, 1, 0, 0, 0]
    cq_testing.assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)


@pytest.mark.parametrize('a,b,num_bits', itertools.product(range(4), range(4), range(3, 5)))
def test_subtract(a, b, num_bits):
    num_anc = num_bits - 1
    gate = AdditionGate(num_bits)
    qubits = cirq.LineQubit.range(2 * num_bits)
    greedy_mm = cirq_infra.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits), context=context))
    ancillas = sorted(circuit.all_qubits())[-num_anc:]
    initial_state = [0] * (2 * num_bits + num_anc)
    initial_state[:num_bits] = list(bit_tools.iter_bits_twos_complement(a, num_bits))[::-1]
    initial_state[num_bits : 2 * num_bits] = list(
        bit_tools.iter_bits_twos_complement(-b, num_bits)
    )[::-1]
    final_state = [0] * (2 * num_bits + num_bits - 1)
    final_state[:num_bits] = list(bit_tools.iter_bits_twos_complement(a, num_bits))[::-1]
    final_state[num_bits : 2 * num_bits] = list(
        bit_tools.iter_bits_twos_complement(a - b, num_bits)
    )[::-1]
    all_qubits = qubits + ancillas
    cq_testing.assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)


@pytest.mark.parametrize("n", [*range(3, 10)])
def test_addition_gate_t_complexity(n: int):
    g = AdditionGate(n)
    cq_testing.assert_decompose_is_consistent_with_t_complexity(g)


@pytest.mark.parametrize('a,b', itertools.product(range(2**3), repeat=2))
def test_add_no_decompose(a, b):
    num_bits = 5
    qubits = cirq.LineQubit.range(2 * num_bits)
    op = AdditionGate(num_bits).on(*qubits)
    circuit = cirq.Circuit(op)
    maps = {}
    a_bin = format(a, f'0{num_bits}b')
    b_bin = format(b, f'0{num_bits}b')
    out_bin = format(a + b, f'0{num_bits}b')
    true_out_int = a + b
    input_int = int(a_bin + b_bin, 2)
    output_int = int(a_bin + out_bin, 2)
    assert true_out_int == int(out_bin, 2)
    maps[input_int] = output_int
    cirq.testing.assert_equivalent_computational_basis_map(maps, circuit)
