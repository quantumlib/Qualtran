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

from qualtran import BloqBuilder
from qualtran.bloqs.arithmetic import (
    EqualsAConstant,
    GreaterThan,
    GreaterThanConstant,
    LessThanConstant,
    LessThanEqual,
)
from qualtran.cirq_interop.bit_tools import iter_bits
from qualtran.cirq_interop.testing import (
    assert_circuit_inp_out_cirqsim,
    assert_decompose_is_consistent_with_t_complexity,
)
from qualtran.testing import (
    assert_valid_bloq_decomposition,
    assert_wire_symbols_match_expected,
    execute_notebook,
)


def _make_greater_than():
    from qualtran.bloqs.arithmetic import GreaterThan

    return GreaterThan(a_bitsize=4, b_bitsize=4)


def _make_greater_than_constant():
    from qualtran.bloqs.arithmetic import GreaterThanConstant

    return GreaterThanConstant(bitsize=4, val=13)


def _make_equals_a_constant():
    from qualtran.bloqs.arithmetic import EqualsAConstant

    return EqualsAConstant(bitsize=4, val=13)


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
    expected_wire_symbols = ("In(x)",) * x_bitsize + ("In(y)",) * y_bitsize + ("⨁(x <= y)",)
    assert cirq.circuit_diagram_info(g).wire_symbols == expected_wire_symbols
    # Test with_registers
    assert g.with_registers([2] * 4, [2] * 5, [2]) == LessThanEqual(4, 5)


def test_greater_than():
    bb = BloqBuilder()
    bitsize = 5
    q0 = bb.add_register('a', bitsize)
    q1 = bb.add_register('b', bitsize)
    anc = bb.add_register('result', 1)
    q0, q1, anc = bb.add(GreaterThan(bitsize, bitsize), a=q0, b=q1, target=anc)
    cbloq = bb.finalize(a=q0, b=q1, result=anc)
    cbloq.t_complexity()
    assert_wire_symbols_match_expected(GreaterThanConstant(bitsize, 17), ['In(x)', '⨁(x > 17)'])


def test_greater_than_constant():
    bb = BloqBuilder()
    bitsize = 5
    q0 = bb.add_register('x', bitsize)
    anc = bb.add_register('result', 1)
    q0, anc = bb.add(GreaterThanConstant(bitsize, 17), x=q0, target=anc)
    cbloq = bb.finalize(x=q0, result=anc)
    cbloq.t_complexity()
    assert_wire_symbols_match_expected(GreaterThanConstant(bitsize, 17), ['In(x)', '⨁(x > 17)'])


def test_equals_a_constant():
    bb = BloqBuilder()
    bitsize = 5
    q0 = bb.add_register('x', bitsize)
    anc = bb.add_register('result', 1)
    q0, anc = bb.add(EqualsAConstant(bitsize, 17), x=q0, target=anc)
    cbloq = bb.finalize(x=q0, result=anc)
    cbloq.t_complexity()
    assert_wire_symbols_match_expected(EqualsAConstant(bitsize, 17), ['In(x)', '⨁(x = 17)'])


def test_comparison_gates_notebook():
    execute_notebook('comparison_gates')


def test_arithmetic_notebook():
    execute_notebook('arithmetic')
