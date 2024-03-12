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

import qualtran.testing as qlt_testing
from qualtran import BloqBuilder
from qualtran.bloqs.arithmetic.comparison import (
    _eq_k,
    _greater_than,
    _gt_k,
    EqualsAConstant,
    GreaterThan,
    GreaterThanConstant,
    LessThanConstant,
    LessThanEqual,
    LinearDepthGreaterThan,
)
from qualtran.cirq_interop.bit_tools import iter_bits
from qualtran.cirq_interop.testing import (
    assert_circuit_inp_out_cirqsim,
    assert_decompose_is_consistent_with_t_complexity,
)


def test_greater_than(bloq_autotester):
    bloq_autotester(_greater_than)


def test_gt_k(bloq_autotester):
    bloq_autotester(_gt_k)


def test_eq_k(bloq_autotester):
    bloq_autotester(_eq_k)


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
    assert cirq.circuit_diagram_info(gate).wire_symbols == ("In(x)",) * 3 + ("⨁(x < 5)",)
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
    qlt_testing.assert_valid_bloq_decomposition(g)


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
    qlt_testing.assert_valid_bloq_decomposition(g)

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
    expected_wire_symbols = ("In(x)",) * x_bitsize + ("In(y)",) * y_bitsize + ("⨁(x <= y)",)
    assert cirq.circuit_diagram_info(g).wire_symbols == expected_wire_symbols
    # Test with_registers
    assert g.with_registers([2] * 4, [2] * 5, [2]) == LessThanEqual(4, 5)


def test_greater_than_manual():
    bb = BloqBuilder()
    bitsize = 5
    q0 = bb.add_register('a', bitsize)
    q1 = bb.add_register('b', bitsize)
    anc = bb.add_register('result', 1)
    q0, q1, anc = bb.add(GreaterThan(bitsize, bitsize), a=q0, b=q1, target=anc)
    cbloq = bb.finalize(a=q0, b=q1, result=anc)
    cbloq.t_complexity()
    qlt_testing.assert_wire_symbols_match_expected(
        GreaterThanConstant(bitsize, 17), ['In(x)', '⨁(x > 17)']
    )


@pytest.mark.parametrize('bitsize', [1, 2, 5])
@pytest.mark.parametrize('signed', [False, True])
@pytest.mark.parametrize('num_targets', [1, 2, 3, 4])
def test_linear_depth_greater_than_decomp(bitsize, signed, num_targets):
    bloq = LinearDepthGreaterThan(bitsize=bitsize, signed=signed, num_targets=num_targets)
    qlt_testing.assert_valid_bloq_decomposition(bloq)


# TODO: write tests for signed integer comparison
# https://github.com/quantumlib/Qualtran/issues/606
@pytest.mark.parametrize(
    'bitsize,signed,num_targets,a,b,targets,result',
    [
        (1, False, 1, 1, 0, (0,), (1,)),
        (2, False, 1, 2, 3, (0,), (0,)),
        (3, False, 2, 5, 3, (1, 1), (0, 0)),
        (4, False, 2, 8, 8, (0, 1), (0, 1)),
        (5, False, 3, 30, 16, (1, 1, 0), (0, 0, 1)),
        (1, True, 1, 1, 1, (0,), (0,)),
        (2, True, 1, 1, 0, (1,), (0,)),
        (3, True, 3, 2, 0, (0, 0, 0), (1, 1, 1)),
        (4, True, 4, 7, 7, (1, 1, 0, 1), (1, 1, 0, 1)),
        (5, True, 5, 13, 12, (1, 1, 1, 1, 1), (0, 0, 0, 0, 0)),
    ],
)
def test_classical_linear_depth_greater_than(bitsize, signed, num_targets, a, b, targets, result):
    bloq = LinearDepthGreaterThan(bitsize=bitsize, signed=signed, num_targets=num_targets)
    cbloq = bloq.decompose_bloq()
    print(targets)
    bloq_classical = bloq.call_classically(a=a, b=b, targets=targets)
    cbloq_classical = cbloq.call_classically(a=a, b=b, targets=targets)

    assert len(bloq_classical) == len(cbloq_classical)
    for i in range(len(bloq_classical)):
        np.testing.assert_array_equal(bloq_classical[i], cbloq_classical[i])

    assert (bloq_classical[-1] == result).all()


def test_greater_than_constant():
    bb = BloqBuilder()
    bitsize = 5
    q0 = bb.add_register('x', bitsize)
    anc = bb.add_register('result', 1)
    q0, anc = bb.add(GreaterThanConstant(bitsize, 17), x=q0, target=anc)
    cbloq = bb.finalize(x=q0, result=anc)
    cbloq.t_complexity()
    qlt_testing.assert_wire_symbols_match_expected(
        GreaterThanConstant(bitsize, 17), ['In(x)', '⨁(x > 17)']
    )


def test_equals_a_constant():
    bb = BloqBuilder()
    bitsize = 5
    q0 = bb.add_register('x', bitsize)
    anc = bb.add_register('result', 1)
    q0, anc = bb.add(EqualsAConstant(bitsize, 17), x=q0, target=anc)
    cbloq = bb.finalize(x=q0, result=anc)
    cbloq.t_complexity()
    qlt_testing.assert_wire_symbols_match_expected(
        EqualsAConstant(bitsize, 17), ['In(x)', '⨁(x = 17)']
    )


@pytest.mark.notebook
def test_t_complexity_of_comparison_gates_notebook():
    qlt_testing.execute_notebook('t_complexity_of_comparison_gates')


@pytest.mark.notebook
def test_comparison_notebook():
    qlt_testing.execute_notebook('comparison')
