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

from qualtran.cirq_interop._interop_qubit_manager import InteropQubitManager


@pytest.mark.parametrize('base_qm', [cirq.GreedyQubitManager('anc'), cirq.SimpleQubitManager()])
def test_interop_qubit_manager(base_qm):
    qm = InteropQubitManager(base_qm)
    # qm delegates allocation / de-allocation requests to base_qm.
    q = qm.qalloc(1)
    qm.qfree(q)
    q = cirq.q('junk')
    # q is not managed and was not allocated by base_qm.
    with pytest.raises((ValueError, AssertionError)):
        qm.qfree([q])
    # You can delegate qubits to be "managed" by the InteropQubitManager.
    qm.manage_qubits([q])
    qm.qfree([q])
    # q was already deallocated.
    with pytest.raises((ValueError, AssertionError)):
        qm.qfree([q])
