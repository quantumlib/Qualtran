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
import cirq_ft
import cirq_ft.infra.testing as cq_testing
import numpy as np
import pytest

from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates.z_basis import IntState
from qualtran.bloqs.swap_network import CSwapApprox, SwapWithZero
from qualtran.simulation.quimb_sim import flatten_for_tensor_contraction
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
        full_state_vector = flatten_for_tensor_contraction(circuit).tensor_contract()

        result_state_vector = cirq.sub_state_vector(
            full_state_vector,
            keep_indices=list(range(selection_bitsize, selection_bitsize + target_bitsize)),
        )
        # Expected state vector should correspond to data[selection_integer] due to the swap.
        expected_state_vector[data[selection_integer]] = 1
        # Assert that result and expected state vectors are equal; reset and continue.
        assert cirq.equal_up_to_global_phase(result_state_vector, expected_state_vector)
        expected_state_vector[data[selection_integer]] = 0


def test_swap_with_zero_classically():
    data = np.array([131, 255, 92, 2])
    swz = SwapWithZero(selection_bitsize=2, target_bitsize=8, n_target_registers=4)

    for sel in range(2**2):
        sel, out_data = swz.call_classically(selection=sel, targets=data)
        print(sel, out_data)


@pytest.mark.parametrize("n", [*range(1, 6)])
def test_t_complexity(n):
    g = cirq_ft.MultiTargetCSwap(n)
    cq_testing.assert_decompose_is_consistent_with_t_complexity(g)

    g = cirq_ft.MultiTargetCSwapApprox(n)
    cq_testing.assert_decompose_is_consistent_with_t_complexity(g)


@pytest.mark.parametrize(
    "selection_bitsize, target_bitsize, n_target_registers, want",
    [
        [3, 5, 1, cirq_ft.TComplexity(t=0, clifford=0)],
        [2, 2, 3, cirq_ft.TComplexity(t=16, clifford=86)],
        [2, 3, 4, cirq_ft.TComplexity(t=36, clifford=195)],
        [3, 2, 5, cirq_ft.TComplexity(t=32, clifford=172)],
        [4, 1, 10, cirq_ft.TComplexity(t=36, clifford=189)],
    ],
)
def test_swap_with_zero_t_complexity(selection_bitsize, target_bitsize, n_target_registers, want):
    gate = cirq_ft.SwapWithZeroGate(selection_bitsize, target_bitsize, n_target_registers)
    assert want == cirq_ft.t_complexity(gate)


def test_notebook():
    execute_notebook('swap_network')
