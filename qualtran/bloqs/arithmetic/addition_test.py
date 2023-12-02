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

import itertools

import cirq
import pytest

from qualtran import BloqBuilder
from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran.bloqs.arithmetic.addition import (
    Add,
    AddConstantMod,
    OutOfPlaceAdder,
    SimpleAddConstant,
)
from qualtran.bloqs.arithmetic.comparison_test import identity_map
from qualtran.cirq_interop.bit_tools import iter_bits, iter_bits_twos_complement
from qualtran.cirq_interop.testing import (
    assert_circuit_inp_out_cirqsim,
    assert_decompose_is_consistent_with_t_complexity,
    GateHelper,
)
from qualtran.testing import assert_valid_bloq_decomposition


def _make_add():
    from qualtran.bloqs.arithmetic import Add

    return Add(bitsize=4)


@pytest.mark.parametrize('a,b,num_bits', itertools.product(range(4), range(4), range(3, 5)))
def test_add_decomposition(a: int, b: int, num_bits: int):
    num_anc = num_bits - 1
    gate = Add(num_bits)
    qubits = cirq.LineQubit.range(2 * num_bits)
    op = gate.on_registers(a=qubits[:num_bits], b=qubits[num_bits:])
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(op, context=context))
    ancillas = sorted(circuit.all_qubits())[-num_anc:]
    initial_state = [0] * (2 * num_bits + num_anc)
    initial_state[:num_bits] = list(iter_bits(a, num_bits))[::-1]
    initial_state[num_bits : 2 * num_bits] = list(iter_bits(b, num_bits))[::-1]
    final_state = [0] * (2 * num_bits + num_bits - 1)
    final_state[:num_bits] = list(iter_bits(a, num_bits))[::-1]
    final_state[num_bits : 2 * num_bits] = list(iter_bits(a + b, num_bits))[::-1]
    assert_circuit_inp_out_cirqsim(circuit, qubits + ancillas, initial_state, final_state)
    # Test diagrams
    expected_wire_symbols = ("In(x)",) * num_bits + ("In(y)/Out(x+y)",) * num_bits
    assert cirq.circuit_diagram_info(gate).wire_symbols == expected_wire_symbols
    # Test with_registers
    assert gate.with_registers([2] * 6, [2] * 6) == Add(6)


def test_add_truncated():
    num_bits = 3
    num_anc = num_bits - 1
    gate = Add(num_bits)
    qubits = cirq.LineQubit.range(2 * num_bits)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits)))
    ancillas = sorted(circuit.all_qubits() - frozenset(qubits))
    assert len(ancillas) == num_anc
    all_qubits = qubits + ancillas
    # Corresponds to 2^2 + 2^2 (4 + 4 = 8 = 2^3 (needs num_bits = 4 to work properly))
    initial_state = [0, 0, 1, 0, 0, 1, 0, 0]
    # Should be 1000 (or 0001 below) but bit falls off the end
    final_state = [0, 0, 1, 0, 0, 0, 0, 0]
    # increasing number of bits yields correct value
    assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)

    num_bits = 4
    num_anc = num_bits - 1
    gate = Add(num_bits)
    qubits = cirq.LineQubit.range(2 * num_bits)
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits), context=context))
    ancillas = sorted(circuit.all_qubits() - frozenset(qubits))
    assert len(ancillas) == num_anc
    all_qubits = qubits + ancillas
    initial_state = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    final_state = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)

    num_bits = 3
    num_anc = num_bits - 1
    gate = Add(num_bits)
    qubits = cirq.LineQubit.range(2 * num_bits)
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits), context=context))
    ancillas = sorted(circuit.all_qubits() - frozenset(qubits))
    assert len(ancillas) == num_anc
    all_qubits = qubits + ancillas
    # Corresponds to 2^2 + (2^2 + 2^1 + 2^0) (4 + 7 = 11 = 1011 (need num_bits=4 to work properly))
    initial_state = [0, 0, 1, 1, 1, 1, 0, 0]
    # Should be 1011 (or 1101 below) but last two bits are lost
    final_state = [0, 0, 1, 1, 1, 0, 0, 0]
    assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)


@pytest.mark.parametrize('a,b,num_bits', itertools.product(range(4), range(4), range(3, 5)))
def test_subtract(a, b, num_bits):
    num_anc = num_bits - 1
    gate = Add(num_bits)
    qubits = cirq.LineQubit.range(2 * num_bits)
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits), context=context))
    ancillas = sorted(circuit.all_qubits())[-num_anc:]
    initial_state = [0] * (2 * num_bits + num_anc)
    initial_state[:num_bits] = list(iter_bits_twos_complement(a, num_bits))[::-1]
    initial_state[num_bits : 2 * num_bits] = list(iter_bits_twos_complement(-b, num_bits))[::-1]
    final_state = [0] * (2 * num_bits + num_bits - 1)
    final_state[:num_bits] = list(iter_bits_twos_complement(a, num_bits))[::-1]
    final_state[num_bits : 2 * num_bits] = list(iter_bits_twos_complement(a - b, num_bits))[::-1]
    all_qubits = qubits + ancillas
    assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)


@pytest.mark.parametrize("n", [*range(3, 10)])
def test_addition_gate_t_complexity(n: int):
    g = Add(n)
    assert_decompose_is_consistent_with_t_complexity(g)
    assert_valid_bloq_decomposition(g)


@pytest.mark.parametrize('a,b', itertools.product(range(2**3), repeat=2))
def test_add_no_decompose(a, b):
    num_bits = 5
    qubits = cirq.LineQubit.range(2 * num_bits)
    op = Add(num_bits).on(*qubits)
    circuit = cirq.Circuit(op)
    basis_map = {}
    a_bin = format(a, f'0{num_bits}b')
    b_bin = format(b, f'0{num_bits}b')
    out_bin = format(a + b, f'0{num_bits}b')
    true_out_int = a + b
    input_int = int(a_bin + b_bin, 2)
    output_int = int(a_bin + out_bin, 2)
    assert true_out_int == int(out_bin, 2)
    basis_map[input_int] = output_int
    cirq.testing.assert_equivalent_computational_basis_map(basis_map, circuit)


def test_add():
    bb = BloqBuilder()
    bitsize = 4
    q0 = bb.add_register('a', bitsize)
    q1 = bb.add_register('b', bitsize)
    a, b = bb.add(Add(bitsize), a=q0, b=q1)
    cbloq = bb.finalize(a=a, b=b)
    cbloq.t_complexity()


@pytest.mark.parametrize('bitsize', [3])
@pytest.mark.parametrize('mod', [5, 8])
@pytest.mark.parametrize('add_val', [1, 2])
@pytest.mark.parametrize('cvs', [[], [0, 1], [1, 0], [1, 1]])
def test_add_mod_n(bitsize, mod, add_val, cvs):
    gate = AddConstantMod(bitsize, mod, add_val=add_val, cvs=cvs)
    basis_map = {}
    num_cvs = len(cvs)
    for x in range(2**bitsize):
        y = (x + add_val) % mod if x < mod else x
        if not num_cvs:
            basis_map[x] = y
            continue
        for cb in range(2**num_cvs):
            inp = f'0b_{cb:0{num_cvs}b}_{x:0{bitsize}b}'
            if tuple(int(x) for x in f'{cb:0{num_cvs}b}') == tuple(cvs):
                out = f'0b_{cb:0{num_cvs}b}_{y:0{bitsize}b}'
                basis_map[int(inp, 2)] = int(out, 2)
            else:
                basis_map[int(inp, 2)] = int(inp, 2)

    op = gate.on_registers(**get_named_qubits(gate.signature))
    circuit = cirq.Circuit(op)
    cirq.testing.assert_equivalent_computational_basis_map(basis_map, circuit)
    circuit += op**-1
    cirq.testing.assert_equivalent_computational_basis_map(identity_map(gate.num_qubits()), circuit)


def test_add_mod_n_protocols():
    with pytest.raises(ValueError, match="must be between"):
        _ = AddConstantMod(3, 10)
    add_one = AddConstantMod(3, 5, 1)
    add_two = AddConstantMod(3, 5, 2, cvs=[1, 0])

    assert add_one == AddConstantMod(3, 5, 1)
    assert add_one != add_two
    assert hash(add_one) != hash(add_two)
    assert add_two.cvs == (1, 0)
    assert cirq.circuit_diagram_info(add_two).wire_symbols == ('@', '@(0)') + ('Add_2_Mod_5',) * 3


def test_out_of_place_adder():
    basis_map = {}
    gate = OutOfPlaceAdder(bitsize=3)
    for x in range(2**3):
        for y in range(2**3):
            basis_map[int(f'0b_{x:03b}_{y:03b}_0000', 2)] = int(f'0b_{x:03b}_{y:03b}_{x+y:04b}', 2)
            assert gate.call_classically(a=x, b=y, c=0) == (x, y, x + y)
    op = GateHelper(gate).operation
    op_inv = cirq.inverse(op)
    cirq.testing.assert_equivalent_computational_basis_map(basis_map, cirq.Circuit(op))
    cirq.testing.assert_equivalent_computational_basis_map(
        basis_map, cirq.Circuit(cirq.decompose_once(op))
    )
    # Check that inverse un-computes correctly
    qubit_order = op.qubits
    circuit = cirq.Circuit(cirq.decompose_once(op), cirq.decompose_once(op_inv))
    for x in range(2**6):
        bits = [*iter_bits(x, 10)][::-1]
        assert_circuit_inp_out_cirqsim(circuit, qubit_order, bits, bits)
    assert gate.t_complexity().t == 3 * 4
    assert (gate**-1).t_complexity().t == 0
    assert_decompose_is_consistent_with_t_complexity(gate**-1)
    assert_valid_bloq_decomposition(gate)
    assert_valid_bloq_decomposition(gate**-1)


@pytest.mark.parametrize('bitsize', [3])
@pytest.mark.parametrize('k', [5, 8])
@pytest.mark.parametrize('signed', [True, False])
@pytest.mark.parametrize('cvs', [[], [0, 1], [1, 0], [1, 1]])
def test_simple_add_constant_decomp(bitsize, k, signed, cvs):
    bloq = SimpleAddConstant(bitsize=bitsize, k=k, cvs=cvs, signed=signed)
    assert_valid_bloq_decomposition(bloq)
