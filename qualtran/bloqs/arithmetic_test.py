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

from qualtran import BloqBuilder, Register
from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran.bloqs.arithmetic import (
    Add,
    AddConstantMod,
    EqualsAConstant,
    GreaterThan,
    GreaterThanConstant,
    HammingWeightCompute,
    LessThanConstant,
    LessThanEqual,
    MultiplyTwoReals,
    OutOfPlaceAdder,
    Product,
    ScaleIntByReal,
    SignedIntegerToTwosComplement,
    Square,
    SquareRealNumber,
    SumOfSquares,
    ToContiguousIndex,
)
from qualtran.bloqs.basic_gates import TGate
from qualtran.cirq_interop.bit_tools import iter_bits, iter_bits_twos_complement
from qualtran.cirq_interop.testing import (
    assert_circuit_inp_out_cirqsim,
    assert_decompose_is_consistent_with_t_complexity,
    GateHelper,
)
from qualtran.testing import assert_valid_bloq_decomposition, execute_notebook


def _make_add():
    from qualtran.bloqs.arithmetic import Add

    return Add(bitsize=4)


def _make_square():
    from qualtran.bloqs.arithmetic import Square

    return Square(bitsize=8)


def _make_sum_of_squares():
    from qualtran.bloqs.arithmetic import SumOfSquares

    return SumOfSquares(bitsize=8, k=4)


def _make_product():
    from qualtran.bloqs.arithmetic import Product

    return Product(a_bitsize=4, b_bitsize=6)


def _make_greater_than():
    from qualtran.bloqs.arithmetic import GreaterThan

    return GreaterThan(a_bitsize=4, b_bitsize=4)


def _make_greater_than_constant():
    from qualtran.bloqs.arithmetic import GreaterThanConstant

    return GreaterThanConstant(bitsize=4, val=13)


def _make_equals_a_constant():
    from qualtran.bloqs.arithmetic import EqualsAConstant

    return EqualsAConstant(bitsize=4, val=13)


def _make_to_contiguous_index():
    from qualtran.bloqs.arithmetic import ToContiguousIndex

    return ToContiguousIndex(bitsize=4, s_bitsize=8)


def _make_scale_int_by_real():
    from qualtran.bloqs.arithmetic import ScaleIntByReal

    return ScaleIntByReal(r_bitsize=8, i_bitsize=12)


def _make_multiply_two_reals():
    from qualtran.bloqs.arithmetic import MultiplyTwoReals

    return MultiplyTwoReals(bitsize=10)


def _make_square_real_number():
    from qualtran.bloqs.arithmetic import SquareRealNumber

    return SquareRealNumber(bitsize=10)


def _make_signed_to_twos_complement():
    from qualtran.bloqs.arithmetic import SignedIntegerToTwosComplement

    return SignedIntegerToTwosComplement(bitsize=10)


def identity_map(n: int):
    """Returns a dict of size `2**n` mapping each integer in range [0, 2**n) to itself."""
    return {i: i for i in range(2**n)}


def test_less_than_gate():
    qubits = cirq.LineQubit.range(4)
    gate = LessThanConstant(3, 5)
    op = gate.on(*qubits)
    circuit = cirq.Circuit(op)
    basis_map = {
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
    cirq.testing.assert_equivalent_computational_basis_map(basis_map, circuit)
    circuit += op**-1
    cirq.testing.assert_equivalent_computational_basis_map(identity_map(len(qubits)), circuit)
    gate2 = LessThanConstant(4, 10)
    assert gate.with_registers(*gate2.registers()) == gate2
    assert cirq.circuit_diagram_info(gate).wire_symbols == ("In(x)",) * 3 + ("+(x < 5)",)
    assert (gate**1 is gate) and (gate**-1 is gate)
    assert gate.__pow__(2) is NotImplemented


@pytest.mark.parametrize("bits", [*range(8)])
@pytest.mark.parametrize("val", [3, 5, 7, 8, 9])
def test_decompose_less_than_gate(bits: int, val: int):
    qubit_states = list(iter_bits(bits, 3))
    circuit = cirq.Circuit(
        cirq.decompose_once(
            LessThanConstant(3, val).on_registers(x=cirq.LineQubit.range(3), target=cirq.q(4))
        )
    )
    if val < 8:
        initial_state = [0] * 4 + qubit_states + [0]
        output_state = [0] * 4 + qubit_states + [int(bits < val)]
    else:
        # When val >= 2**number_qubits the decomposition doesn't create any ancilla since the
        # answer is always 1.
        initial_state = [0]
        output_state = [1]
    assert_circuit_inp_out_cirqsim(
        circuit, sorted(circuit.all_qubits()), initial_state, output_state
    )


@pytest.mark.parametrize("n", [*range(2, 5)])
@pytest.mark.parametrize("val", [3, 4, 5, 7, 8, 9])
def test_less_than_consistent_protocols(n: int, val: int):
    g = LessThanConstant(n, val)
    assert_decompose_is_consistent_with_t_complexity(g)
    # Test the unitary is self-inverse
    u = cirq.unitary(g)
    np.testing.assert_allclose(u @ u, np.eye(2 ** (n + 1)))
    assert_valid_bloq_decomposition(g)


def test_multi_in_less_equal_than_gate():
    qubits = cirq.LineQubit.range(7)
    op = LessThanEqual(3, 3).on_registers(x=qubits[:3], y=qubits[3:6], target=qubits[-1])
    circuit = cirq.Circuit(op)
    basis_map = {}
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
            basis_map[input_int] = output_int

    cirq.testing.assert_equivalent_computational_basis_map(basis_map, circuit)
    circuit += op**-1
    cirq.testing.assert_equivalent_computational_basis_map(identity_map(len(qubits)), circuit)


@pytest.mark.parametrize("x_bitsize", [*range(1, 5)])
@pytest.mark.parametrize("y_bitsize", [*range(1, 5)])
def test_less_than_equal_consistent_protocols(x_bitsize: int, y_bitsize: int):
    g = LessThanEqual(x_bitsize, y_bitsize)
    assert_decompose_is_consistent_with_t_complexity(g)
    assert_valid_bloq_decomposition(g)

    # Decomposition works even when context is None.
    qubits = cirq.LineQid.range(x_bitsize + y_bitsize + 1, dimension=2)
    assert cirq.Circuit(g._decompose_with_context_(qubits=qubits)) == cirq.Circuit(
        cirq.decompose_once(
            g.on(*qubits), context=cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
        )
    )

    # Test the unitary is self-inverse
    assert g**-1 is g
    u = cirq.unitary(g)
    np.testing.assert_allclose(u @ u, np.eye(2 ** (x_bitsize + y_bitsize + 1)))
    # Test diagrams
    expected_wire_symbols = ("In(x)",) * x_bitsize + ("In(y)",) * y_bitsize + ("+(x <= y)",)
    assert cirq.circuit_diagram_info(g).wire_symbols == expected_wire_symbols
    # Test with_registers
    assert g.with_registers([2] * 4, [2] * 5, [2]) == LessThanEqual(4, 5)


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


def test_square():
    bb = BloqBuilder()
    bitsize = 4
    q0 = bb.add_register('a', bitsize)
    q0, q1 = bb.add(Square(bitsize), a=q0)
    cbloq = bb.finalize(a=q0, result=q1)
    cbloq.t_complexity()


def test_sum_of_squares():
    bb = BloqBuilder()
    bitsize = 4
    k = 3
    inp = bb.add_register(Register("input", bitsize=bitsize, shape=(k,)))
    inp, out = bb.add(SumOfSquares(bitsize, k), input=inp)
    cbloq = bb.finalize(input=inp, result=out)
    assert SumOfSquares(bitsize, k).signature[1].bitsize == 2 * bitsize + 2
    cbloq.t_complexity()


def test_product():
    bb = BloqBuilder()
    bitsize = 5
    mbits = 3
    q0 = bb.add_register('a', bitsize)
    q1 = bb.add_register('b', mbits)
    q0, q1, q2 = bb.add(Product(bitsize, mbits), a=q0, b=q1)
    cbloq = bb.finalize(a=q0, b=q1, result=q2)
    cbloq.t_complexity()


def test_scale_int_by_real():
    bb = BloqBuilder()
    q0 = bb.add_register('a', 15)
    q1 = bb.add_register('b', 8)
    q0, q1, q2 = bb.add(ScaleIntByReal(15, 8), real_in=q0, int_in=q1)
    cbloq = bb.finalize(a=q0, b=q1, result=q2)
    cbloq.t_complexity()


def test_multiply_two_reals():
    bb = BloqBuilder()
    q0 = bb.add_register('a', 15)
    q1 = bb.add_register('b', 15)
    q0, q1, q2 = bb.add(MultiplyTwoReals(15), a=q0, b=q1)
    cbloq = bb.finalize(a=q0, b=q1, result=q2)
    cbloq.t_complexity()


def test_square_real_number():
    bb = BloqBuilder()
    q0 = bb.add_register('a', 15)
    q1 = bb.add_register('b', 15)
    q0, q1, q2 = bb.add(SquareRealNumber(15), a=q0, b=q1)
    cbloq = bb.finalize(a=q0, b=q1, result=q2)


def test_greater_than():
    bb = BloqBuilder()
    bitsize = 5
    q0 = bb.add_register('a', bitsize)
    q1 = bb.add_register('b', bitsize)
    anc = bb.add_register('result', 1)
    q0, q1, anc = bb.add(GreaterThan(bitsize, bitsize), a=q0, b=q1, target=anc)
    cbloq = bb.finalize(a=q0, b=q1, result=anc)
    cbloq.t_complexity()


def test_greater_than_constant():
    bb = BloqBuilder()
    bitsize = 5
    q0 = bb.add_register('x', bitsize)
    anc = bb.add_register('result', 1)
    q0, anc = bb.add(GreaterThanConstant(bitsize, 17), x=q0, target=anc)
    cbloq = bb.finalize(x=q0, result=anc)
    cbloq.t_complexity()


def test_equals_a_constant():
    bb = BloqBuilder()
    bitsize = 5
    q0 = bb.add_register('x', bitsize)
    anc = bb.add_register('result', 1)
    q0, anc = bb.add(EqualsAConstant(bitsize, 17), x=q0, target=anc)
    cbloq = bb.finalize(x=q0, result=anc)
    cbloq.t_complexity()


def test_to_contiguous_index():
    bb = BloqBuilder()
    bitsize = 5
    q0 = bb.add_register('mu', bitsize)
    q1 = bb.add_register('nu', bitsize)
    out = bb.add_register('s', 1)
    q0, q1, out = bb.add(ToContiguousIndex(bitsize, 2 * bitsize), mu=q0, nu=q1, s=out)
    cbloq = bb.finalize(mu=q0, nu=q1, s=out)
    cbloq.t_complexity()


def test_signed_to_twos_complement():
    bb = BloqBuilder()
    bitsize = 5
    q0 = bb.add_register('x', bitsize)
    q0 = bb.add(SignedIntegerToTwosComplement(bitsize), x=q0)
    cbloq = bb.finalize(x=q0)
    _, sigma = cbloq.call_graph()
    assert sigma[TGate()] == 4 * (5 - 2)


def test_arithmetic_notebook():
    execute_notebook('arithmetic')


def test_comparison_gates_notebook():
    execute_notebook('comparison_gates')


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
