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

from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran.bloqs.hubbard_model import PrepareHubbard, SelectHubbard
from qualtran.cirq_interop.t_complexity_protocol import t_complexity
from qualtran.testing import assert_valid_bloq_decomposition, execute_notebook


@pytest.mark.parametrize('dim', [*range(2, 10)])
def test_select_t_complexity(dim):
    select = SelectHubbard(x_dim=dim, y_dim=dim, control_val=1)
    cost = t_complexity(select)
    N = 2 * dim * dim
    logN = 2 * (dim - 1).bit_length() + 1
    assert cost.t == 10 * N + 14 * logN - 8
    assert cost.rotations == 0


@pytest.mark.parametrize('dim', [*range(3, 10)])
def test_prepare_t_complexity(dim):
    prepare = PrepareHubbard(x_dim=dim, y_dim=dim, t=2, mu=8)
    cost = t_complexity(prepare)
    logN = 2 * (dim - 1).bit_length() + 1
    assert cost.t <= 32 * logN
    # TODO(#233): The rotation count should reduce to a constant once cost for Controlled-H
    # gates is recognized as $2$ T-gates instead of $2$ rotations.
    assert cost.rotations <= 2 * logN + 9


def test_hubbard_model_consistent_protocols():
    select_gate = SelectHubbard(x_dim=2, y_dim=2)
    prepare_gate = PrepareHubbard(x_dim=2, y_dim=2, t=1, mu=2)

    assert_valid_bloq_decomposition(select_gate)
    assert_valid_bloq_decomposition(prepare_gate)

    # Build controlled SELECT gate
    select_op = select_gate.on_registers(**get_named_qubits(select_gate.signature))
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.add_equality_group(
        select_gate.controlled(),
        select_gate.controlled(num_controls=1),
        select_gate.controlled(control_values=(1,)),
        select_op.controlled_by(cirq.q("control")).gate,
    )
    equals_tester.add_equality_group(
        select_gate.controlled(control_values=(0,)),
        select_gate.controlled(num_controls=1, control_values=(0,)),
        select_op.controlled_by(cirq.q("control"), control_values=(0,)).gate,
    )
    with pytest.raises(NotImplementedError, match="Cannot create a controlled version"):
        _ = select_gate.controlled(num_controls=2)

    # Test diagrams
    expected_symbols = ['U', 'V', 'p_x', 'p_y', 'alpha', 'q_x', 'q_y', 'beta']
    expected_symbols += ['target'] * 8
    expected_symbols[0] = 'SelectHubbard'
    assert cirq.circuit_diagram_info(select_gate).wire_symbols == tuple(expected_symbols)


def test_hubbard_model_notebook():
    execute_notebook('hubbard_model')


def test_phase_estimation_of_qubitized_hubbard_model():
    execute_notebook('phase_estimation_of_quantum_walk')
