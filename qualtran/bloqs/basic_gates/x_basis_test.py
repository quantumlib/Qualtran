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
from qualtran.bloqs.basic_gates import MinusState, PlusEffect, PlusState, XGate
from qualtran.resource_counting import GateCounts, get_cost_value, QECGatesCost
from qualtran.simulation.classical_sim import (
    format_classical_truth_table,
    get_classical_truth_table,
)


def test_plus_state():
    bloq = PlusState()
    vector = bloq.tensor_contract()
    should_be = np.array([1, 1]) / np.sqrt(2)
    np.testing.assert_allclose(should_be, vector)


def _make_plus_effect():
    from qualtran.bloqs.basic_gates import PlusEffect

    return PlusEffect()


def test_plus_effect():
    bloq = PlusEffect()
    vector = bloq.tensor_contract()

    # Note: we don't do "column vectors" or anything for kets.
    # Everything is squeezed. Keep track manually or use compositebloq.
    should_be = np.array([1, 1]) / np.sqrt(2)
    np.testing.assert_allclose(should_be, vector)

    assert get_cost_value(bloq, QECGatesCost()) == GateCounts()


def test_plus_state_effect():
    bb = BloqBuilder()

    q0 = bb.add(PlusState())
    bb.add(PlusEffect(), q=q0)
    cbloq = bb.finalize()
    val = cbloq.tensor_contract()

    should_be = 1
    np.testing.assert_allclose(should_be, val)


def test_to_cirq():
    bb = BloqBuilder()
    q = bb.add(PlusState())
    q = bb.add(XGate(), q=q)
    cbloq = bb.finalize(q=q)
    circuit = cbloq.to_cirq_circuit()
    cirq.testing.assert_has_diagram(circuit, "_c(0): ───H───X───")
    vec1 = cbloq.tensor_contract()
    vec2 = cirq.final_state_vector(circuit)
    np.testing.assert_allclose(vec1, vec2)

    bb = BloqBuilder()
    q = bb.add(MinusState())
    q = bb.add(XGate(), q=q)
    cbloq = bb.finalize(q=q)
    circuit = cbloq.to_cirq_circuit()
    cirq.testing.assert_has_diagram(circuit, "_c(0): ───[ _c(0): ───X───H─── ]───X───")
    vec1 = cbloq.tensor_contract()
    vec2 = cirq.final_state_vector(circuit)
    np.testing.assert_allclose(vec1, vec2)


def test_x_truth_table():
    classical_truth_table = format_classical_truth_table(*get_classical_truth_table(XGate()))
    assert (
        classical_truth_table
        == """\
q  |  q
--------
0 -> 1
1 -> 0"""
    )
