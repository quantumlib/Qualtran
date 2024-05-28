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
from qualtran.bloqs.chemistry.hubbard_model.qubitization import PrepareHubbard, SelectHubbard
from qualtran.testing import assert_valid_bloq_decomposition, execute_notebook


def test_hubbard_model_consistent_protocols():
    select_gate = SelectHubbard(x_dim=2, y_dim=2)
    prepare_gate = PrepareHubbard(x_dim=2, y_dim=2, t=1, u=2)

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

    # Test diagrams
    expected_symbols = ['U', 'V', 'p_x', 'p_y', 'alpha', 'q_x', 'q_y', 'beta']
    expected_symbols += ['target'] * 8
    expected_symbols[0] = 'SelectHubbard'
    assert cirq.circuit_diagram_info(select_gate).wire_symbols == tuple(expected_symbols)


@pytest.mark.notebook
def test_hubbard_model_notebook():
    execute_notebook('hubbard_model')
