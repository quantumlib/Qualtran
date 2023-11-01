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

import cirq
import pytest

from qualtran import SelectionRegister
from qualtran.bloqs.and_bloq import And
from qualtran.bloqs.basic_gates import TGate, Toffoli
from qualtran.bloqs.cswap_lth_reg import ApplyCSwapToLthReg
from qualtran.cirq_interop.bit_tools import iter_bits
from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim, GateHelper
from qualtran.testing import assert_valid_bloq_decomposition, execute_notebook


@pytest.mark.parametrize(
    "selection_bitsize,iteration_length,target_bitsize", [[2, 3, 2], [3, 2, 3]]
)
def test_cswap_lth_reg(selection_bitsize, iteration_length, target_bitsize):
    greedy_mm = cirq.GreedyQubitManager(prefix="_a", maximize_reuse=True)
    gate = ApplyCSwapToLthReg(
        SelectionRegister('selection', selection_bitsize, iteration_length), bitsize=target_bitsize
    )
    g = GateHelper(gate, context=cirq.DecompositionContext(greedy_mm))
    # Upper bounded because not all ancillas may be used as part of unary iteration.
    print(len(g.all_qubits))

    for n in range(iteration_length):
        # Initial qubit values
        qubit_vals = {q: 0 for q in g.all_qubits}
        # Set selection according to `n`
        qubit_vals.update(zip(g.quregs['selection'], iter_bits(n, selection_bitsize)))
        final_state = [qubit_vals[x] for x in g.all_qubits]

        # swap the nth register (x{n}) with the ancilla (y)
        # put some non-zero numbers in the registers for comparison.
        qubit_vals.update(zip(g.quregs[f'x{n}'], iter_bits(n + 1, target_bitsize)))
        initial_state = [qubit_vals[x] for x in g.all_qubits]
        qubit_vals.update(zip(g.quregs[f'x{n}'], [0] * len(g.quregs[f'x{n}'])))
        qubit_vals.update(zip(g.quregs['y'], iter_bits(n + 1, target_bitsize)))
        final_state = [qubit_vals[x] for x in g.all_qubits]
        assert_circuit_inp_out_cirqsim(
            g.decomposed_circuit, g.all_qubits, initial_state, final_state
        )


@pytest.mark.parametrize(
    "selection_bitsize,iteration_length,target_bitsize", [[2, 3, 2], [3, 2, 3]]
)
def test_bloq_has_consistent_decomposition(selection_bitsize, iteration_length, target_bitsize):
    bloq = ApplyCSwapToLthReg(
        SelectionRegister('selection', selection_bitsize, iteration_length), bitsize=target_bitsize
    )
    assert_valid_bloq_decomposition(bloq)


@pytest.mark.parametrize(
    "selection_bitsize,iteration_length,target_bitsize", [[3, 8, 2], [4, 9, 3]]
)
def test_t_counts(selection_bitsize, iteration_length, target_bitsize):
    bloq = ApplyCSwapToLthReg(
        SelectionRegister('selection', selection_bitsize, iteration_length), bitsize=target_bitsize
    )
    expected = 4 * (iteration_length - 2) + 7 * (iteration_length * target_bitsize)
    assert bloq.t_complexity().t == expected
    assert bloq.call_graph()[1][TGate()] == expected
