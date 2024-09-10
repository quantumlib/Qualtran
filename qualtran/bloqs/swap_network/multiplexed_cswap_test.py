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
import pytest

from qualtran import BQUInt, QUInt, Register
from qualtran.bloqs.swap_network.multiplexed_cswap import _multiplexed_cswap, MultiplexedCSwap
from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim, GateHelper
from qualtran.resource_counting import GateCounts, get_cost_value, QECGatesCost
from qualtran.testing import assert_valid_bloq_decomposition

random.seed(12345)


@pytest.mark.parametrize(
    "selection_bitsize,iteration_length,target_bitsize", [[2, 3, 2], [3, 2, 3]]
)
def test_cswap_lth_reg(selection_bitsize, iteration_length, target_bitsize):
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    gate = MultiplexedCSwap(
        Register('selection', BQUInt(selection_bitsize, iteration_length)),
        target_bitsize=target_bitsize,
    )
    g = GateHelper(gate, context=cirq.DecompositionContext(greedy_mm))
    for n in range(iteration_length):
        # Initial qubit values
        qubit_vals = {q: 0 for q in g.all_qubits}
        # Set selection according to `n`
        qubit_vals.update(zip(g.quregs['selection'], QUInt(selection_bitsize).to_bits(n)))

        # swap the nth register (x{n}) with the ancilla (y)
        # put some non-zero numbers in the registers for comparison.
        qubit_vals.update(zip(g.quregs['targets'][n], QUInt(target_bitsize).to_bits(n + 1)))
        initial_state = [qubit_vals[x] for x in g.all_qubits]
        qubit_vals.update(zip(g.quregs['targets'][n], [0] * len(g.quregs['targets'][n])))
        qubit_vals.update(zip(g.quregs['output'], QUInt(target_bitsize).to_bits(n + 1)))
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
        Register('selection', BQUInt(selection_bitsize, iteration_length)),
        target_bitsize=target_bitsize,
    )
    assert_valid_bloq_decomposition(bloq)


@pytest.mark.parametrize(
    "selection_bitsize,iteration_length,target_bitsize", [[3, 8, 2], [4, 9, 3]]
)
def test_multiplexed_cswap_t_counts(selection_bitsize, iteration_length, target_bitsize):
    bloq = MultiplexedCSwap(
        Register('selection', BQUInt(selection_bitsize, iteration_length)),
        target_bitsize=target_bitsize,
    )
    expected_t = 4 * (iteration_length - 2) + 7 * (iteration_length * target_bitsize)
    assert bloq.t_complexity().t == expected_t
    gc = get_cost_value(bloq, QECGatesCost())
    assert gc == GateCounts(
        and_bloq=iteration_length - 2,
        measurement=iteration_length - 2,  # and^dag,
        cswap=iteration_length * target_bitsize,
        clifford=gc.clifford,  # don't test this
    )


def test_multiplexed_cswap(bloq_autotester):
    bloq_autotester(_multiplexed_cswap)
