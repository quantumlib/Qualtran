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

import random
from typing import Dict, Tuple

import cirq
import numpy as np
import pytest
import sympy

import qualtran.cirq_interop.testing as cq_testing
from qualtran import Bloq, BloqBuilder, SelectionRegister
from qualtran.bloqs.basic_gates import CSwap, TGate
from qualtran.bloqs.basic_gates.z_basis import IntState
from qualtran.bloqs.swap_network import (
    _approx_cswap_large,
    _approx_cswap_small,
    _approx_cswap_symb,
    _multiplexed_cswap,
    _swz,
    _swz_small,
    CSwapApprox,
    MultiplexedCSwap,
    SwapWithZero,
)
from qualtran.bloqs.util_bloqs import ArbitraryClifford
from qualtran.cirq_interop.bit_tools import iter_bits
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim, GateHelper
from qualtran.simulation.tensor import flatten_for_tensor_contraction
from qualtran.testing import assert_valid_bloq_decomposition, execute_notebook

random.seed(12345)


def _make_CSwapApprox():
    from qualtran.bloqs.swap_network import CSwapApprox

    return CSwapApprox(bitsize=64)


def _make_SwapWithZero():
    from qualtran.bloqs.swap_network import SwapWithZero

    return SwapWithZero(selection_bitsize=3, target_bitsize=64, n_target_registers=5)


def test_cswap_approx_decomp():
    csa = CSwapApprox(10)
    assert_valid_bloq_decomposition(csa)


@pytest.mark.parametrize('n', [5, 32])
def test_approx_cswap_t_count(n):
    cswap = CSwapApprox(bitsize=n)
    cswap_d = cswap.decompose_bloq()

    assert cswap.t_complexity() == cswap_d.t_complexity()


def test_swap_with_zero_decomp():
    swz = SwapWithZero(selection_bitsize=3, target_bitsize=64, n_target_registers=5)
    assert_valid_bloq_decomposition(swz)


@pytest.mark.parametrize(
    "selection_bitsize, target_bitsize, n_target_registers",
    [[3, 5, 1], [2, 2, 3], [2, 3, 4], [3, 2, 5], [4, 1, 10]],
)
def test_swap_with_zero_bloq(selection_bitsize, target_bitsize, n_target_registers):
    swz = SwapWithZero(selection_bitsize, target_bitsize, n_target_registers)
    data = [random.randint(0, 2**target_bitsize - 1) for _ in range(n_target_registers)]

    expected_state_vector = np.zeros(2**target_bitsize)
    # Iterate on every selection integer.
    for selection_integer in range(len(data)):
        bb = BloqBuilder()
        sel = bb.add(IntState(val=selection_integer, bitsize=selection_bitsize))
        trgs = []
        for i in range(n_target_registers):
            trg = bb.add(IntState(val=data[i], bitsize=target_bitsize))
            trgs.append(trg)
        sel, trgs = bb.add(swz, selection=sel, targets=np.array(trgs))
        circuit = bb.finalize(sel=sel, trgs=trgs)
        flat_circuit = flatten_for_tensor_contraction(circuit)
        full_state_vector = flat_circuit.tensor_contract()
        result_state_vector = cirq.sub_state_vector(
            full_state_vector,
            keep_indices=list(range(selection_bitsize, selection_bitsize + target_bitsize)),
        )
        # Expected state vector should correspond to data[selection_integer] due to the swap.
        expected_state_vector[data[selection_integer]] = 1
        # Assert that result and expected state vectors are equal; reset and continue.
        assert cirq.equal_up_to_global_phase(result_state_vector, expected_state_vector)
        expected_state_vector[data[selection_integer]] = 0


def test_swap_with_zero_cirq_gate_diagram():
    gate = SwapWithZero(3, 2, 4)
    gh = cq_testing.GateHelper(gate)
    cirq.testing.assert_has_diagram(
        cirq.Circuit(gh.operation, cirq.decompose_once(gh.operation)),
        """
selection0: ──────@(r⇋0)───────────────────────────────────────
                  │
selection1: ──────@(r⇋0)───────────────────────────@(approx)───
                  │                                │
selection2: ──────@(r⇋0)───@(approx)───@(approx)───┼───────────
                  │        │           │           │
targets[0][0]: ───swap_0───×(x)────────┼───────────×(x)────────
                  │        │           │           │
targets[0][1]: ───swap_0───×(x)────────┼───────────×(x)────────
                  │        │           │           │
targets[1][0]: ───swap_1───×(y)────────┼───────────┼───────────
                  │        │           │           │
targets[1][1]: ───swap_1───×(y)────────┼───────────┼───────────
                  │                    │           │
targets[2][0]: ───swap_2───────────────×(x)────────×(y)────────
                  │                    │           │
targets[2][1]: ───swap_2───────────────×(x)────────×(y)────────
                  │                    │
targets[3][0]: ───swap_3───────────────×(y)────────────────────
                  │                    │
targets[3][1]: ───swap_3───────────────×(y)────────────────────
""",
    )


def test_swap_with_zero_classically():
    data = np.array([131, 255, 92, 2])
    swz = SwapWithZero(selection_bitsize=2, target_bitsize=8, n_target_registers=4)

    for sel in range(2**2):
        sel, out_data = swz.call_classically(selection=sel, targets=data)
        print(sel, out_data)


def get_t_count_and_clifford(bc: Dict[Bloq, int]) -> Tuple[int, int]:
    """Get the t count and clifford cost from bloq count."""
    cliff_cost = sum([v for k, v in bc.items() if isinstance(k, ArbitraryClifford)])
    t_cost = sum([v for k, v in bc.items() if isinstance(k, TGate)])
    return t_cost, cliff_cost


@pytest.mark.parametrize("n", [*range(1, 6)])
def test_t_complexity(n):
    cq_testing.assert_decompose_is_consistent_with_t_complexity(CSwap(n))
    cq_testing.assert_decompose_is_consistent_with_t_complexity(CSwapApprox(n))


@pytest.mark.parametrize("n", [*range(2, 6)])
def test_cswap_approx_bloq_counts(n):
    csa = CSwapApprox(n)
    bc = csa.bloq_counts()
    t_cost, cliff_cost = get_t_count_and_clifford(bc)
    assert csa.t_complexity().clifford == cliff_cost
    assert csa.t_complexity().t == t_cost


@pytest.mark.parametrize(
    "selection_bitsize, target_bitsize, n_target_registers, want",
    [
        [3, 5, 1, TComplexity(t=0, clifford=0)],
        [2, 2, 3, TComplexity(t=16, clifford=86)],
        [2, 3, 4, TComplexity(t=36, clifford=195)],
        [3, 2, 5, TComplexity(t=32, clifford=172)],
        [4, 1, 10, TComplexity(t=36, clifford=189)],
    ],
)
def test_swap_with_zero_bloq_counts(selection_bitsize, target_bitsize, n_target_registers, want):
    gate = SwapWithZero(selection_bitsize, target_bitsize, n_target_registers)

    n = sympy.Symbol('n')

    def _gen_clif(bloq: Bloq) -> Bloq:
        if isinstance(bloq, ArbitraryClifford):
            return ArbitraryClifford(n)
        return bloq

    _, sigma = gate.call_graph(generalizer=_gen_clif)
    assert sigma[TGate()] == want.t
    assert sigma[ArbitraryClifford(n)] == want.clifford


@pytest.mark.parametrize(
    "selection_bitsize, target_bitsize, n_target_registers, want",
    [
        [3, 5, 1, TComplexity(t=0, clifford=0)],
        [2, 2, 3, TComplexity(t=16, clifford=86)],
        [2, 3, 4, TComplexity(t=36, clifford=195)],
        [3, 2, 5, TComplexity(t=32, clifford=172)],
        [4, 1, 10, TComplexity(t=36, clifford=189)],
    ],
)
def test_swap_with_zero_t_complexity(selection_bitsize, target_bitsize, n_target_registers, want):
    gate = SwapWithZero(selection_bitsize, target_bitsize, n_target_registers)
    assert want == gate.t_complexity()


@pytest.mark.parametrize(
    "selection_bitsize,iteration_length,target_bitsize", [[2, 3, 2], [3, 2, 3]]
)
def test_cswap_lth_reg(selection_bitsize, iteration_length, target_bitsize):
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    gate = MultiplexedCSwap(
        SelectionRegister('selection', selection_bitsize, iteration_length),
        target_bitsize=target_bitsize,
    )
    g = GateHelper(gate, context=cirq.DecompositionContext(greedy_mm))
    for n in range(iteration_length):
        # Initial qubit values
        qubit_vals = {q: 0 for q in g.all_qubits}
        # Set selection according to `n`
        qubit_vals.update(zip(g.quregs['selection'], iter_bits(n, selection_bitsize)))
        final_state = [qubit_vals[x] for x in g.all_qubits]

        # swap the nth register (x{n}) with the ancilla (y)
        # put some non-zero numbers in the registers for comparison.
        qubit_vals.update(zip(g.quregs['targets'][n], iter_bits(n + 1, target_bitsize)))
        initial_state = [qubit_vals[x] for x in g.all_qubits]
        qubit_vals.update(zip(g.quregs['targets'][n], [0] * len(g.quregs['targets'][n])))
        qubit_vals.update(zip(g.quregs['output'], iter_bits(n + 1, target_bitsize)))
        final_state = [qubit_vals[x] for x in g.all_qubits]
        assert_circuit_inp_out_cirqsim(
            g.decomposed_circuit, g.all_qubits, initial_state, final_state
        )


@pytest.mark.parametrize(
    "selection_bitsize,iteration_length,target_bitsize", [[2, 3, 2], [3, 2, 3]]
)
def test_multiplexed_cswap_bloq_has_consistent_decomposition(
    selection_bitsize, iteration_length, target_bitsize
):
    bloq = MultiplexedCSwap(
        SelectionRegister('selection', selection_bitsize, iteration_length),
        target_bitsize=target_bitsize,
    )
    assert_valid_bloq_decomposition(bloq)


@pytest.mark.parametrize(
    "selection_bitsize,iteration_length,target_bitsize", [[3, 8, 2], [4, 9, 3]]
)
def test_multiplexed_cswap_t_counts(selection_bitsize, iteration_length, target_bitsize):
    bloq = MultiplexedCSwap(
        SelectionRegister('selection', selection_bitsize, iteration_length),
        target_bitsize=target_bitsize,
    )
    expected = 4 * (iteration_length - 2) + 7 * (iteration_length * target_bitsize)
    assert bloq.t_complexity().t == expected
    assert bloq.call_graph()[1][TGate()] == expected


def test_multiplexed_cswap(bloq_autotester):
    bloq_autotester(_multiplexed_cswap)


def test_approx_cswap_small(bloq_autotester):
    bloq_autotester(_approx_cswap_small)


def test_approx_cswap_symb(bloq_autotester):
    bloq_autotester(_approx_cswap_symb)


def test_approx_cswap_large(bloq_autotester):
    bloq_autotester(_approx_cswap_large)


def test_swz_small(bloq_autotester):
    bloq_autotester(_swz_small)


def test_swz(bloq_autotester):
    bloq_autotester(_swz)


def test_notebook():
    execute_notebook('swap_network')
