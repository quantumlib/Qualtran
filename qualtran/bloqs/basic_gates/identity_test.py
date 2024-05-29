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
import numpy as np

from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import OneState
from qualtran.bloqs.basic_gates.identity import _identity, Identity
from qualtran.simulation.classical_sim import (
    format_classical_truth_table,
    get_classical_truth_table,
)


def test_to_cirq():
    bb = BloqBuilder()
    q = bb.add(OneState())
    q = bb.add(Identity(), q=q)
    cbloq = bb.finalize(q=q)
    circuit = cbloq.to_cirq_circuit()
    cirq.testing.assert_has_diagram(circuit, "_c(0): ───X───I───")
    vec1 = cbloq.tensor_contract()
    vec2 = cirq.final_state_vector(circuit)
    np.testing.assert_allclose(vec1, vec2)


def test_identity(bloq_autotester):
    bloq_autotester(_identity)


def test_unitary_vs_cirq():
    i = Identity()
    unitary = i.tensor_contract()
    cirq_unitary = cirq.unitary(cirq.I)
    np.testing.assert_allclose(unitary, cirq_unitary)


def test_i_truth_table():
    classical_truth_table = format_classical_truth_table(*get_classical_truth_table(Identity()))
    assert (
        classical_truth_table
        == """\
q  |  q
--------
0 -> 0
1 -> 1"""
    )
