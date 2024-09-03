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
from qualtran import BloqBuilder, QInt, QMontgomeryUInt, QUInt
from qualtran.bloqs.arithmetic.comparison import (
    _clineardepthgreaterthan_example,
    _eq_k,
    _greater_than,
    _gt_k,
    _leq_symb,
    _lt_k_symb,
    BiQubitsMixer,
    CLinearDepthGreaterThan,
    EqualsAConstant,
    GreaterThan,
    GreaterThanConstant,
    LessThanConstant,
    LessThanEqual,
    LinearDepthGreaterThan,
    SingleQubitCompare,
)
from qualtran.cirq_interop.t_complexity_protocol import t_complexity, TComplexity
from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join


def test_clineardepthgreaterthan_example(bloq_autotester):
    bloq_autotester(_clineardepthgreaterthan_example)


def test_greater_than(bloq_autotester):
    bloq_autotester(_greater_than)


def test_gt_k(bloq_autotester):
    bloq_autotester(_gt_k)


def test_lt_k_symb(bloq_autotester):
    bloq_autotester(_lt_k_symb)


def test_leq_symb(bloq_autotester):
    bloq_autotester(_leq_symb)


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
    # Missing cirq stubs
    circuit += op**-1  # type: ignore[operator]
    cirq.testing.assert_equivalent_computational_basis_map(identity_map(len(qubits)), circuit)
    gate2 = LessThanConstant(4, 10)
    assert gate.with_registers(*gate2.registers()) == gate2
    assert cirq.circuit_diagram_info(gate).wire_symbols == ("In(x)",) * 3 + ("⨁(x < 5)",)
    assert (gate**1 is gate) and (gate**-1 is gate)
    assert gate.__pow__(2) is NotImplemented


@pytest.mark.parametrize("bits", [*range(8)])
@pytest.mark.parametrize("val", [3, 5, 7, 8, 9])
def test_decompose_less_than_gate(bits: int, val: int):
    qubit_states = QUInt(3).to_bits(bits)
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
    expected_t_complexity = (
        TComplexity(clifford=1)
        if val >= 2**n
        else TComplexity(t=4 * n, clifford=15 * n + 3 * bin(val).count("1") + 2)
    )
    assert g.t_complexity() == expected_t_complexity
    # Test the unitary is self-inverse
    u = cirq.unitary(g)
    np.testing.assert_allclose(u @ u, np.eye(2 ** (n + 1)))
    qlt_testing.assert_valid_bloq_decomposition(g)


def test_bi_qubits_mixer_t_complexity():
    g = BiQubitsMixer()
    assert g.t_complexity() == TComplexity(t=8, clifford=28)
    assert g.adjoint().t_complexity() == TComplexity(clifford=18)


def test_single_qubit_compare_t_complexity():
    g = SingleQubitCompare()
    assert g.t_complexity() == TComplexity(t=4, clifford=14)
    assert g.adjoint().t_complexity() == TComplexity(clifford=9)


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
    # Missing cirq stubs
    circuit += op**-1  # type: ignore[operator]
    cirq.testing.assert_equivalent_computational_basis_map(identity_map(len(qubits)), circuit)


def _less_than_equal_expected_t_complexity(gate: LessThanEqual):
    n = min(gate.x_bitsize, gate.y_bitsize)
    d = max(gate.x_bitsize, gate.y_bitsize) - n
    is_second_longer = gate.y_bitsize > gate.x_bitsize
    if d == 0:
        # When both registers are of the same size the T complexity is
        # 8n - 4 same as in the second reference.
        return TComplexity(t=8 * n - 4, clifford=46 * n - 21)
    else:
        # When the registers differ in size and `n` is the size of the smaller one and
        # `d` is the difference in size. The T complexity is the sum of the tree
        # decomposition as before giving 8n + O(1) and the T complexity of an `And` gate
        # over `d` registers giving 4d + O(1) totaling 8n + 4d + O(1).
        # From the decomposition we get that the constant is -4 as well as the clifford counts.
        if d == 1:
            return TComplexity(t=8 * n, clifford=46 * n - 1 + 2 * is_second_longer)
        else:
            return TComplexity(
                t=8 * n + 4 * d - 4, clifford=46 * n + 17 * d - 18 + 2 * is_second_longer
            )


@pytest.mark.parametrize("x_bitsize", [*range(1, 5)])
@pytest.mark.parametrize("y_bitsize", [*range(1, 5)])
def test_less_than_equal_consistent_protocols(x_bitsize: int, y_bitsize: int):
    g = LessThanEqual(x_bitsize, y_bitsize)
    assert g.t_complexity() == _less_than_equal_expected_t_complexity(g)
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
    qlt_testing.assert_wire_symbols_match_expected(
        GreaterThanConstant(bitsize, 17), ['In(x)', '⨁(x > 17)']
    )
    assert cbloq.t_complexity() == (
        t_complexity(LessThanEqual(bitsize, bitsize)) + TComplexity(clifford=1)
    )


@pytest.mark.parametrize('bitsize', [1, 2, 5])
@pytest.mark.parametrize('signed', [False, True])
def test_linear_depth_greater_than_decomp(bitsize, signed):
    bloq = LinearDepthGreaterThan(bitsize=bitsize, signed=signed)
    qlt_testing.assert_valid_bloq_decomposition(bloq)
    qlt_testing.assert_equivalent_bloq_counts(bloq, [ignore_alloc_free, ignore_split_join])


# TODO: write tests for signed integer comparison
# https://github.com/quantumlib/Qualtran/issues/606
@pytest.mark.parametrize(
    'bitsize,signed,a,b,target,result',
    [
        (1, False, 1, 0, 0, 1),
        (2, False, 2, 3, 0, 0),
        (3, False, 5, 3, 1, 0),
        (4, False, 8, 8, 0, 0),
        (5, False, 30, 16, 1, 0),
        (1, True, 1, 1, 0, 0),
        (2, True, 1, 0, 1, 0),
        (3, True, 2, 0, 0, 1),
        (4, True, 7, 7, 1, 1),
        (5, True, 13, 12, 1, 0),
    ],
)
def test_classical_linear_depth_greater_than(bitsize, signed, a, b, target, result):
    bloq = LinearDepthGreaterThan(bitsize=bitsize, signed=signed)
    cbloq = bloq.decompose_bloq()
    bloq_classical = bloq.call_classically(a=a, b=b, target=target)
    cbloq_classical = cbloq.call_classically(a=a, b=b, target=target)

    assert len(bloq_classical) == len(cbloq_classical)
    for i in range(len(bloq_classical)):
        np.testing.assert_array_equal(bloq_classical[i], cbloq_classical[i])

    assert bloq_classical[-1] == result


def test_greater_than_constant():
    bb = BloqBuilder()
    bitsize = 5
    q0 = bb.add_register('x', bitsize)
    anc = bb.add_register('result', 1)
    q0, anc = bb.add(GreaterThanConstant(bitsize, 17), x=q0, target=anc)
    cbloq = bb.finalize(x=q0, result=anc)
    qlt_testing.assert_wire_symbols_match_expected(
        GreaterThanConstant(bitsize, 17), ['In(x)', '⨁(x > 17)']
    )
    assert t_complexity(GreaterThanConstant(bitsize, 17)) == t_complexity(
        LessThanConstant(bitsize, 17)
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
    assert t_complexity(EqualsAConstant(bitsize, 17)) == TComplexity(
        t=4 * (bitsize - 1), clifford=65
    )


@pytest.mark.notebook
def test_t_complexity_of_comparison_gates_notebook():
    qlt_testing.execute_notebook('t_complexity_of_comparison_gates')


@pytest.mark.notebook
def test_comparison_notebook():
    qlt_testing.execute_notebook('comparison')


@pytest.mark.parametrize('gate', [LessThanConstant(3, 3), LessThanEqual(3, 3)])
def test_decomposition_frees_ancilla(gate):
    op = gate(*cirq.LineQid.for_gate(gate))
    qubit_manager = cirq.ops.GreedyQubitManager(prefix='_test')
    _ = cirq.decompose(op, context=cirq.DecompositionContext(qubit_manager))
    assert len(qubit_manager._used_qubits) == 0


@pytest.mark.parametrize('ctrl', range(2))
@pytest.mark.parametrize('dtype', [QUInt, QMontgomeryUInt])
@pytest.mark.parametrize('bitsize', range(1, 5))
def test_clineardepthgreaterthan_classical_action_unsigned(ctrl, dtype, bitsize):
    b = CLinearDepthGreaterThan(dtype(bitsize), ctrl)
    cb = b.decompose_bloq()
    for c, target in itertools.product(range(2), repeat=2):
        for (x, y) in itertools.product(range(2**bitsize), repeat=2):
            assert b.call_classically(ctrl=c, a=x, b=y, target=target) == cb.call_classically(
                ctrl=c, a=x, b=y, target=target
            )


@pytest.mark.parametrize('ctrl', range(2))
@pytest.mark.parametrize('bitsize', range(2, 5))
def test_clineardepthgreaterthan_classical_action_signed(ctrl, bitsize):
    b = CLinearDepthGreaterThan(QInt(bitsize), ctrl)
    cb = b.decompose_bloq()
    for c, target in itertools.product(range(2), repeat=2):
        for (x, y) in itertools.product(range(-(2 ** (bitsize - 1)), 2 ** (bitsize - 1)), repeat=2):
            assert b.call_classically(ctrl=c, a=x, b=y, target=target) == cb.call_classically(
                ctrl=c, a=x, b=y, target=target
            )


@pytest.mark.parametrize('ctrl', range(2))
@pytest.mark.parametrize('dtype', [QInt, QUInt, QMontgomeryUInt])
@pytest.mark.parametrize('bitsize', range(2, 5))
def test_clineardepthgreaterthan_decomposition(ctrl, dtype, bitsize):
    b = CLinearDepthGreaterThan(dtype(bitsize), ctrl)
    qlt_testing.assert_valid_bloq_decomposition(b)


@pytest.mark.parametrize('ctrl', range(2))
@pytest.mark.parametrize('dtype', [QInt, QUInt, QMontgomeryUInt])
@pytest.mark.parametrize('bitsize', range(2, 5))
def test_clineardepthgreaterthan_bloq_counts(ctrl, dtype, bitsize):
    b = CLinearDepthGreaterThan(dtype(bitsize), ctrl)
    qlt_testing.assert_equivalent_bloq_counts(b, [ignore_alloc_free, ignore_split_join])


@pytest.mark.parametrize('ctrl', range(2))
@pytest.mark.parametrize('dtype', [QInt, QUInt, QMontgomeryUInt])
def test_clineardepthgreaterthan_tcomplexity(ctrl, dtype):
    n = sympy.Symbol('n')
    c = CLinearDepthGreaterThan(dtype(n), ctrl).t_complexity()
    assert c.t == 4 * (n + 2)
    assert c.rotations == 0
