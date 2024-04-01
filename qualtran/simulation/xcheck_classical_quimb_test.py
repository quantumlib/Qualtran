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
import pytest

import qualtran.testing as qlt_testing
from qualtran import QUInt
from qualtran.bloqs.arithmetic import Add
from qualtran.bloqs.basic_gates import CNOT, Toffoli, XGate
from qualtran.simulation.xcheck_classical_quimb import flank_with_classical_vectors


def test_xgate():
    x = XGate()
    x_tt = flank_with_classical_vectors(x, {'q': 0})
    assert x_tt.tensor_contract() == 1.0

    result = flank_with_classical_vectors(x, in_vals={'q': 0}, out_vals={'q': 0}).tensor_contract()
    assert result == 0.0


def test_cnot():
    cnot = CNOT()
    cnot_tt = flank_with_classical_vectors(cnot, {'ctrl': 1, 'target': 0})
    assert cnot_tt.tensor_contract() == 1.0


def test_toffoli():
    tof = Toffoli()
    tof_tt = flank_with_classical_vectors(tof, {'ctrl': [1, 1], 'target': 0})
    assert tof_tt.tensor_contract() == 1.0


def test_add():
    add = Add(QUInt(bitsize=5))
    add_tt = flank_with_classical_vectors(add, {'a': 2, 'b': 3})
    assert add_tt.tensor_contract() == 1.0


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('xcheck_classical_quimb')
