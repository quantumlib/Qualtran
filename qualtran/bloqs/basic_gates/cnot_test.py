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

from qualtran import BloqBuilder, Signature
from qualtran.bloqs.basic_gates import CNOT, PlusState, ZeroState
from qualtran.bloqs.basic_gates.cnot import _cnot
from qualtran.drawing import get_musical_score_data


def _make_CNOT():
    from qualtran.bloqs.basic_gates import CNOT

    return CNOT()


def test_cnot_tensor():
    bloq = CNOT()
    matrix = bloq.tensor_contract()
    # fmt: off
    should_be = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]])
    # fmt: on
    np.testing.assert_allclose(should_be, matrix)


def test_cnot_cbloq_tensor_vs_cirq():
    bb, soqs = BloqBuilder.from_signature(Signature.build(c=1, t=1))
    c, t = bb.add(CNOT(), ctrl=soqs['c'], target=soqs['t'])
    cbloq = bb.finalize(c=c, t=t)
    matrix = cbloq.tensor_contract()

    c_qs = cirq.LineQubit.range(2)
    c_circ = cirq.Circuit(cirq.CNOT(c_qs[0], c_qs[1]))
    c_matrix = c_circ.unitary(qubit_order=c_qs)

    np.testing.assert_allclose(c_matrix, matrix)


def test_bell_statevector():
    bb = BloqBuilder()

    q0 = bb.add(PlusState())
    q1 = bb.add(ZeroState())

    q0, q1 = bb.add(CNOT(), ctrl=q0, target=q1)

    cbloq = bb.finalize(q0=q0, q1=q1)
    matrix = cbloq.tensor_contract()

    should_be = np.array([1, 0, 0, 1]) / np.sqrt(2)
    np.testing.assert_allclose(should_be, matrix)


def test_classical_truth_table():
    truth_table = []
    for c, t in itertools.product([0, 1], repeat=2):
        out_c, out_t = CNOT().call_classically(ctrl=c, target=t)
        truth_table.append(((c, t), (out_c, out_t)))

    assert truth_table == [((0, 0), (0, 0)), ((0, 1), (0, 1)), ((1, 0), (1, 1)), ((1, 1), (1, 0))]

    with pytest.raises(ValueError):
        CNOT().call_classically(ctrl=2, target=0)


def test_cnot_musical_score():
    cbloq = CNOT().as_composite_bloq()
    msd = get_musical_score_data(cbloq)
    # soq[0] and [1] are the dangling symbols
    assert msd.soqs[2].json_dict()['symb_cls'] == 'Circle'
    assert msd.soqs[3].json_dict()['symb_cls'] == 'ModPlus'


def test_cnot(bloq_autotester):
    bloq_autotester(_cnot)
