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

from qualtran.bloqs.chemistry.trotter.grid_ham.qvr import _qvr, QuantumVariableRotation


def test_kinetic_energy(bloq_autotester):
    bloq_autotester(_qvr)


@pytest.mark.parametrize('bitsize', [8, 16, 32])
def test_qvr_t_complexity(bitsize: int):
    bloq = QuantumVariableRotation(bitsize)
    assert bloq.t_complexity().rotations == bitsize
