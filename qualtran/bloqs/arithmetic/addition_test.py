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
import numpy as np
import pytest
import sympy

import qualtran.testing as qlt_testing
from qualtran import BloqBuilder, CtrlSpec, QInt, QUInt
from qualtran.bloqs.arithmetic.addition import (
    _add_diff_size_regs,
    _add_k,
    _add_k_large,
    _add_k_small,
    _add_large,
    _add_oop_large,
    _add_oop_small,
    _add_oop_symb,
    _add_small,
    _add_symb,
    Add,
    AddK,
    OutOfPlaceAdder,
)
from qualtran.bloqs.mcmt.and_bloq import And
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim, GateHelper
from qualtran.resource_counting import get_cost_value, QubitCount
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.simulation.classical_sim import (
    format_classical_truth_table,
    get_classical_truth_table,
)


@pytest.mark.parametrize(
    "bloq",
    [
        _add_symb,
        _add_small,
        _add_large,
        _add_diff_size_regs,
        _add_oop_symb,
        _add_oop_small,
        _add_oop_large,
        _add_k,
        _add_k_small,
        _add_k_large,
    ],
)
def test_examples(bloq_autotester, bloq):
    bloq_autotester(bloq)


@pytest.mark.parametrize('a,b,num_bits', itertools.product(range(4), range(4), range(3, 5)))
def test_add_decomposition(a: int, b: int, num_bits: int):
    num_anc = num_bits - 1
    gate = Add(QUInt(num_bits))
    qubits = cirq.LineQubit.range(2 * num_bits)
    op = gate.on_registers(a=qubits[:num_bits], b=qubits[num_bits:])
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(op, context=context))
    circuit0 = cirq.Circuit(op)
    ancillas = sorted(circuit.all_qubits())[-num_anc:]
    initial_state = [0] * (2 * num_bits + num_anc)
    initial_state[:num_bits] = QUInt(num_bits).to_bits(a)
    initial_state[num_bits : 2 * num_bits] = QUInt(num_bits).to_bits(b)
    final_state = [0] * (2 * num_bits + num_anc)
    final_state[:num_bits] = QUInt(num_bits).to_bits(a)
    final_state[num_bits : 2 * num_bits] = QUInt(num_bits).to_bits(a + b)
    assert_circuit_inp_out_cirqsim(circuit, qubits + ancillas, initial_state, final_state)
    assert_circuit_inp_out_cirqsim(
        circuit0, qubits, initial_state[:-num_anc], final_state[:-num_anc]
    )
    # Test diagrams
    expected_wire_symbols = ("In(x)",) * num_bits + ("In(y)/Out(x+y)",) * num_bits
    assert cirq.circuit_diagram_info(gate).wire_symbols == expected_wire_symbols


@pytest.mark.parametrize('a', [1, 2])
@pytest.mark.parametrize('b', [1, 2, 3])
@pytest.mark.parametrize('num_bits_a', [2, 3])
@pytest.mark.parametrize('num_bits_b', [4, 5])
def test_add_diff_size_registers(a, b, num_bits_a, num_bits_b):
    num_anc = num_bits_b - 1
    gate = Add(QUInt(num_bits_a), QUInt(num_bits_b))
    qubits = cirq.LineQubit.range(num_bits_a + num_bits_b)
    op = gate.on_registers(a=qubits[:num_bits_a], b=qubits[num_bits_a:])
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(op, context=context))
    circuit0 = cirq.Circuit(op)
    ancillas = sorted(circuit.all_qubits())[-num_anc:]
    initial_state = [0] * (num_bits_a + num_bits_b + num_anc)
    initial_state[:num_bits_a] = QUInt(num_bits_a).to_bits(a)
    initial_state[num_bits_a : num_bits_a + num_bits_b] = QUInt(num_bits_b).to_bits(b)
    final_state = [0] * (num_bits_a + num_bits_b + num_anc)
    final_state[:num_bits_a] = QUInt(num_bits_a).to_bits(a)
    final_state[num_bits_a : num_bits_a + num_bits_b] = QUInt(num_bits_b).to_bits(a + b)
    assert_circuit_inp_out_cirqsim(circuit, qubits + ancillas, initial_state, final_state)
    assert_circuit_inp_out_cirqsim(
        circuit0, qubits, initial_state[:-num_anc], final_state[:-num_anc]
    )
    # Test diagrams
    expected_wire_symbols = ("In(x)",) * num_bits_a + ("In(y)/Out(x+y)",) * num_bits_b
    assert cirq.circuit_diagram_info(gate).wire_symbols == expected_wire_symbols


def test_add_truncated():
    num_bits = 3
    num_anc = num_bits - 1
    gate = Add(QUInt(num_bits))
    qubits = cirq.LineQubit.range(2 * num_bits)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits)))
    ancillas = sorted(circuit.all_qubits() - frozenset(qubits))
    assert len(ancillas) == num_anc
    all_qubits = qubits + ancillas
    # Corresponds to 2^2 + 2^2 (4 + 4 = 8 = 2^3 (needs num_bits = 4 to work properly))
    initial_state = [1, 0, 0, 1, 0, 0, 0, 0]
    # Should be 1000 but bit falls off the end
    final_state = [1, 0, 0, 0, 0, 0, 0, 0]
    assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)

    # increasing number of bits yields correct value
    num_bits = 4
    num_anc = num_bits - 1
    gate = Add(QUInt(num_bits))
    qubits = cirq.LineQubit.range(2 * num_bits)
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits), context=context))
    ancillas = sorted(circuit.all_qubits() - frozenset(qubits))
    assert len(ancillas) == num_anc
    all_qubits = qubits + ancillas
    # 0100|0100|000
    initial_state = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    final_state = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)

    num_bits = 3
    num_anc = num_bits - 1
    gate = Add(QUInt(num_bits))
    qubits = cirq.LineQubit.range(2 * num_bits)
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits), context=context))
    ancillas = sorted(circuit.all_qubits() - frozenset(qubits))
    assert len(ancillas) == num_anc
    all_qubits = qubits + ancillas
    # Corresponds to 2^2 + (2^2 + 2^1 + 2^0) (4 + 7 = 11 = 1011 (need num_bits=4 to work properly))
    initial_state = [1, 0, 0, 1, 1, 1, 0, 0]
    # Should be 1011 but last bit is lost.
    final_state = [1, 0, 0, 0, 1, 1, 0, 0]
    assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)


@pytest.mark.parametrize('a,b,num_bits', itertools.product(range(4), range(4), range(3, 5)))
def test_subtract(a, b, num_bits):
    num_anc = num_bits - 1
    gate = Add(QInt(num_bits))
    qubits = cirq.LineQubit.range(2 * num_bits)
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits), context=context))
    ancillas = sorted(circuit.all_qubits())[-num_anc:]
    initial_state = [0] * (2 * num_bits + num_anc)
    initial_state[:num_bits] = QInt(num_bits).to_bits(a)
    initial_state[num_bits : 2 * num_bits] = QInt(num_bits).to_bits(-b)
    final_state = [0] * (2 * num_bits + num_bits - 1)
    final_state[:num_bits] = QInt(num_bits).to_bits(a)
    final_state[num_bits : 2 * num_bits] = QInt(num_bits).to_bits(a - b)
    all_qubits = qubits + ancillas
    assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)


def add_reference_t_complexity(b: Add):
    n = b.dtype.bitsize
    num_clifford = (n - 2) * 19 + 16
    num_toffoli = n - 1
    return TComplexity(t=4 * num_toffoli, clifford=num_clifford)


@pytest.mark.parametrize("n", [*range(3, 10)])
def test_addition_gate_counts(n: int):
    add = Add(QUInt(n))
    qlt_testing.assert_valid_bloq_decomposition(add)
    assert add.t_complexity() == add.decompose_bloq().t_complexity()
    assert add.t_complexity() == add_reference_t_complexity(add)
    qlt_testing.assert_equivalent_bloq_counts(add, ignore_split_join)


@pytest.mark.parametrize('a,b', itertools.product(range(2**3), repeat=2))
def test_add_tensor_contract(a, b):
    num_bits = 5
    bloq = Add(QUInt(num_bits))

    a_bin = format(a, f'0{num_bits}b')
    b_bin = format(b, f'0{num_bits}b')
    out_bin = format(a + b, f'0{num_bits}b')
    true_out_int = a + b
    input_int = int(a_bin + b_bin, 2)
    output_int = int(a_bin + out_bin, 2)
    assert true_out_int == int(out_bin, 2)

    unitary = bloq.tensor_contract()
    np.testing.assert_allclose(unitary[output_int, input_int], 1)


@pytest.mark.parametrize('a,b,num_bits', itertools.product(range(4), range(4), range(3, 5)))
def test_add_call_classically(a: int, b: int, num_bits: int):
    bloq = Add(QUInt(num_bits))
    ret = bloq.call_classically(a=a, b=b)
    assert ret == (a, a + b)


def test_add_call_classically_overflow():
    bloq = Add(QUInt(3))
    ret = bloq.call_classically(a=5, b=6)
    assert ret == (5, 3)  # 3 = 5+6 mod 8


def test_add_truth_table():
    bloq = Add(QUInt(2))
    classical_truth_table = format_classical_truth_table(*get_classical_truth_table(bloq))
    assert (
        classical_truth_table
        == """\
a  b  |  a  b
--------------
0, 0 -> 0, 0
0, 1 -> 0, 1
0, 2 -> 0, 2
0, 3 -> 0, 3
1, 0 -> 1, 1
1, 1 -> 1, 2
1, 2 -> 1, 3
1, 3 -> 1, 0
2, 0 -> 2, 2
2, 1 -> 2, 3
2, 2 -> 2, 0
2, 3 -> 2, 1
3, 0 -> 3, 3
3, 1 -> 3, 0
3, 2 -> 3, 1
3, 3 -> 3, 2"""
    )


def test_add_in_cbloq():
    bb = BloqBuilder()
    bitsize = 4
    q0 = bb.add_register('a', bitsize)
    q1 = bb.add_register('b', bitsize)
    a, b = bb.add(Add(QUInt(bitsize)), a=q0, b=q1)
    cbloq = bb.finalize(a=a, b=b)
    cbloq.t_complexity()


def test_add_classical():
    bloq = Add(QInt(bitsize=32))
    ret1 = bloq.call_classically(a=10, b=3)
    ret2 = bloq.decompose_bloq().call_classically(a=10, b=3)
    assert ret1 == ret2


def test_add_symb():
    bloq = _add_symb()
    assert bloq.signature.n_qubits() == sympy.sympify('2*n')
    assert get_cost_value(bloq, QubitCount()) == sympy.sympify('Max(3, 2*n)')


def test_out_of_place_adder():
    basis_map = {}
    gate = OutOfPlaceAdder(bitsize=3)
    cbloq = gate.decompose_bloq()
    for x in range(2**3):
        for y in range(2**3):
            basis_map[int(f'0b_{x:03b}_{y:03b}_0000', 2)] = int(f'0b_{x:03b}_{y:03b}_{x+y:04b}', 2)
            assert gate.call_classically(a=x, b=y, c=0) == (x, y, x + y)
            assert cbloq.call_classically(a=x, b=y, c=0) == (x, y, x + y)
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
        bits = QUInt(10).to_bits(x)[::-1]
        assert_circuit_inp_out_cirqsim(circuit, qubit_order, bits, bits)
    assert gate.t_complexity().t == 3 * 4
    assert (gate**-1).t_complexity().t == 0
    qlt_testing.assert_valid_bloq_decomposition(gate)
    qlt_testing.assert_valid_bloq_decomposition(gate**-1)
    and_t = And().t_complexity()
    assert gate.t_complexity() == TComplexity(t=3 * and_t.t, clifford=3 * (5 + and_t.clifford))


def test_controlled_add_k():
    n, k = sympy.symbols('n k')
    addk = AddK(n, k)
    assert addk.controlled() == AddK(n, k, cvs=(1,))
    assert addk.controlled(CtrlSpec(cvs=0)) == AddK(n, k, cvs=(0,))


@pytest.mark.parametrize('bitsize', [5])
@pytest.mark.parametrize('k', [5, 8])
@pytest.mark.parametrize('cvs', [[], [0, 1], [1, 0], [1, 1]])
def test_add_k_decomp_unsigned(bitsize, k, cvs):
    bloq = AddK(bitsize=bitsize, k=k, cvs=cvs, signed=False)
    qlt_testing.assert_valid_bloq_decomposition(bloq)


@pytest.mark.parametrize('bitsize', [5])
@pytest.mark.parametrize('k', [-5, 8])
@pytest.mark.parametrize('cvs', [[], [0, 1], [1, 0], [1, 1]])
def test_add_k_decomp_signed(bitsize, k, cvs):
    bloq = AddK(bitsize=bitsize, k=k, cvs=cvs, signed=True)
    qlt_testing.assert_valid_bloq_decomposition(bloq)


@pytest.mark.parametrize(
    'bitsize,k,x,cvs,ctrls,result',
    [
        (5, 1, 2, (), (), 3),
        (5, 3, 2, (1,), (1,), 5),
        (5, 2, 0, (1, 0), (1, 0), 2),
        (5, 1, 2, (1, 0, 1), (0, 0, 0), 2),
    ],
)
def test_classical_add_k_unsigned(bitsize, k, x, cvs, ctrls, result):
    bloq = AddK(bitsize=bitsize, k=k, cvs=cvs, signed=False)
    cbloq = bloq.decompose_bloq()
    bloq_classical = bloq.call_classically(ctrls=ctrls, x=x)
    cbloq_classical = cbloq.call_classically(ctrls=ctrls, x=x)

    assert len(bloq_classical) == len(cbloq_classical)
    for i in range(len(bloq_classical)):
        np.testing.assert_array_equal(bloq_classical[i], cbloq_classical[i])

    assert bloq_classical[-1] == result


@pytest.mark.parametrize('bitsize', range(2, 5))
def test_classical_add_signed_overflow(bitsize):
    bloq = Add(QInt(bitsize))
    mx = 2 ** (bitsize - 1) - 1
    assert bloq.call_classically(a=mx, b=mx) == (mx, -2)


@pytest.mark.parametrize(
    'bitsize,k,x,cvs,ctrls,result', [(5, 2, 0, (1, 0), (1, 0), 2), (6, -3, 2, (), (), -1)]
)
def test_classical_add_k_signed(bitsize, k, x, cvs, ctrls, result):
    bloq = AddK(bitsize=bitsize, k=k, cvs=cvs, signed=True)
    cbloq = bloq.decompose_bloq()
    bloq_classical = bloq.call_classically(ctrls=ctrls, x=x)
    cbloq_classical = cbloq.call_classically(ctrls=ctrls, x=x)

    assert len(bloq_classical) == len(cbloq_classical)
    for i in range(len(bloq_classical)):
        np.testing.assert_array_equal(bloq_classical[i], cbloq_classical[i])

    assert bloq_classical[-1] == result


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('addition')


@pytest.mark.parametrize('bitsize', range(1, 5))
def test_outofplaceadder_classical_action(bitsize):
    b = OutOfPlaceAdder(bitsize)
    cb = b.decompose_bloq()
    for x, y in itertools.product(range(2**bitsize), repeat=2):
        assert b.call_classically(a=x, b=y) == cb.call_classically(a=x, b=y)

    b = OutOfPlaceAdder(bitsize).adjoint()
    cb = b.decompose_bloq()
    for x, y in itertools.product(range(2**bitsize), repeat=2):
        assert b.call_classically(a=x, b=y, c=x + y) == cb.call_classically(a=x, b=y, c=x + y)
