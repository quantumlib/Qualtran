#  Copyright 2024 Google LLC
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


import numpy as np
import pytest

from qualtran import BQUInt
from qualtran.bloqs.swap_network.one_hot_encoding import OneHotLinearDepth, OneHotLogDepth
from qualtran.resource_counting import get_cost_value, QECGatesCost, QubitCount


@pytest.mark.parametrize('n, ilen', [(3, 8), (4, 14)])
def test_one_hot_linear_depth_classical_action(n, ilen):
    bloq = OneHotLinearDepth(BQUInt(n, ilen))
    for x in range(ilen):
        x_out, out = bloq.call_classically(x=x)
        assert x == x_out
        assert out[x] == 1 and np.all(out[:x] == 0) and np.all(out[x + 1 :] == 0)


@pytest.mark.parametrize('n, ilen', [(3, 8), (4, 14)])
def test_one_hot_log_depth_classical_action(n, ilen):
    bloq = OneHotLogDepth(BQUInt(n, ilen))
    for x in range(ilen):
        x_out, out = bloq.call_classically(x=x)
        assert x == x_out
        assert out[x] == 1 and np.all(out[:x] == 0) and np.all(out[x + 1 :] == 0)


@pytest.mark.parametrize('n, ilen', [(3, 8), (4, 14), (5, 30), (6, 60), (7, 120)])
def test_one_hot_linear_depth_gate_counts(n, ilen):
    bloq = OneHotLinearDepth(BQUInt(n, ilen))
    # N - 1 AND gates.
    assert get_cost_value(bloq, QECGatesCost()).and_bloq == ilen - 1
    assert get_cost_value(bloq, QECGatesCost()).total_t_count() == 4 * ilen - 4
    # Linear depth.
    assert ilen // 2 < len(bloq.decompose_bloq().to_cirq_circuit()) < ilen
    # Qubit Counts
    assert get_cost_value(bloq, QubitCount()) == n + ilen


@pytest.mark.parametrize('n, ilen', [(3, 8), (4, 14), (5, 30), (6, 60), (7, 120)])
def test_one_hot_log_depth_gate_counts(n, ilen):
    bloq = OneHotLogDepth(BQUInt(n, ilen))
    # N - 1 AND gates.
    assert get_cost_value(bloq, QECGatesCost()).and_bloq == ilen - 1
    assert get_cost_value(bloq, QECGatesCost()).total_t_count() == 4 * ilen - 4
    # Log depth.
    assert len(bloq.decompose_bloq().to_cirq_circuit()) == n + 2
    # O(N) additional qubits help achieve log depth
    assert n + ilen < get_cost_value(bloq, QubitCount()) < n + 2 * ilen
