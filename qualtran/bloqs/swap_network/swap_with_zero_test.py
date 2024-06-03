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

import cirq
import numpy as np
import pytest
import sympy

import qualtran.cirq_interop.testing as cq_testing
from qualtran import Bloq, BloqBuilder
from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.basic_gates.z_basis import IntState
from qualtran.bloqs.bookkeeping import ArbitraryClifford
from qualtran.bloqs.swap_network.swap_with_zero import _swz, _swz_small, SwapWithZero
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.simulation.tensor import flatten_for_tensor_contraction
from qualtran.testing import assert_valid_bloq_decomposition

random.seed(12345)


def test_swap_with_zero_decomp():
    swz = SwapWithZero(selection_bitsizes=3, target_bitsize=64, n_target_registers=5)
    assert_valid_bloq_decomposition(swz)


@pytest.mark.slow
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


def test_swap_with_zero_cirq_gate_diagram_multi_dim():
    gate = SwapWithZero((2, 1), 2, (3, 2))
    gh = cq_testing.GateHelper(gate)
    cirq.testing.assert_has_diagram(
        cirq.Circuit(gh.operation, cirq.decompose_once(gh.operation)),
        """
                                                        ┌──────────────────┐
selection0_0: ───────@(r⇋0)────────────────────────────────────────────────────@(approx)───
                     │                                                         │
selection0_1: ───────@(r⇋0)──────────────────────────────@(approx)─────────────┼───────────
                     │                                   │                     │
selection1_: ────────@(r⇋0)─────@(approx)───@(approx)────┼────────@(approx)────┼───────────
                     │          │           │            │        │            │
targets[0, 0][0]: ───swap_0_0───×(x)────────┼────────────×(x)─────┼────────────×(x)────────
                     │          │           │            │        │            │
targets[0, 0][1]: ───swap_0_0───×(x)────────┼────────────×(x)─────┼────────────×(x)────────
                     │          │           │            │        │            │
targets[0, 1][0]: ───swap_0_1───×(y)────────┼────────────┼────────┼────────────┼───────────
                     │          │           │            │        │            │
targets[0, 1][1]: ───swap_0_1───×(y)────────┼────────────┼────────┼────────────┼───────────
                     │                      │            │        │            │
targets[1, 0][0]: ───swap_1_0───────────────×(x)─────────×(y)─────┼────────────┼───────────
                     │                      │            │        │            │
targets[1, 0][1]: ───swap_1_0───────────────×(x)─────────×(y)─────┼────────────┼───────────
                     │                      │                     │            │
targets[1, 1][0]: ───swap_1_1───────────────×(y)──────────────────┼────────────┼───────────
                     │                      │                     │            │
targets[1, 1][1]: ───swap_1_1───────────────×(y)──────────────────┼────────────┼───────────
                     │                                            │            │
targets[2, 0][0]: ───swap_2_0─────────────────────────────────────×(x)─────────×(y)────────
                     │                                            │            │
targets[2, 0][1]: ───swap_2_0─────────────────────────────────────×(x)─────────×(y)────────
                     │                                            │
targets[2, 1][0]: ───swap_2_1─────────────────────────────────────×(y)─────────────────────
                     │                                            │
targets[2, 1][1]: ───swap_2_1─────────────────────────────────────×(y)─────────────────────
                                                        └──────────────────┘
""",
    )


def test_swap_with_zero_classically():
    data = np.array([131, 255, 92, 2])
    swz = SwapWithZero(selection_bitsizes=2, target_bitsize=8, n_target_registers=4)

    for sel in range(2**2):
        sel, out_data = swz.call_classically(selection=sel, targets=data)  # type: ignore[assignment]
        print(sel, out_data)


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


def test_t_complexiy_for_multi_dimensional_swap_with_zero():
    np.random.seed(10)
    selection_bitsize = np.array([*range(2, 5)])
    n_target_registers = np.random.randint(1, 2**selection_bitsize - 1)
    target_bitsize = 3
    bloq = SwapWithZero(selection_bitsize, target_bitsize, n_target_registers)
    assert bloq.decompose_bloq().t_complexity() == bloq.t_complexity()


def test_bloq_counts_symbolic():
    (p, q, r) = sympy.symbols("p q r")
    b = sympy.Symbol("b")
    P, Q, R = sympy.symbols("P Q R")
    swz_multi_symbolic = SwapWithZero(
        selection_bitsizes=(p, q, r), target_bitsize=b, n_target_registers=(P, Q, R)
    )
    g, sigma = swz_multi_symbolic.call_graph()
    assert sigma[TGate()] == 4 * b * (P * Q * R - 1)


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


def test_swz_small(bloq_autotester):
    bloq_autotester(_swz_small)


def test_swz(bloq_autotester):
    bloq_autotester(_swz)
