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

import numpy as np
import pytest
from openfermion.resource_estimates.utils import QI, QR

from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.data_loading.qroam import _qroam_small, QROAM
from qualtran.testing import assert_wire_symbols_match_expected


def test_qroam_small(bloq_autotester):
    bloq_autotester(_qroam_small)


def test_qroam_classical():
    rs = np.random.RandomState()
    data = rs.randint(0, 2**3, size=10)
    qrom = QROAM([data], target_bitsizes=(3,), block_size=2)
    for i in range(len(data)):
        i_out, data_out = qrom.call_classically(selection=i, target0_=0)
        assert i_out == i
        assert data_out == data[i]

        decomp_ret = qrom.call_classically(selection=i, target0_=0)
        assert decomp_ret == (i_out, data_out)


def test_qroam_1d_multitarget_classical():
    rs = np.random.RandomState()
    n = 10
    data_sets = [rs.randint(0, 2**3, size=n) for _ in range(3)]
    qroam = QROAM.build(*data_sets, target_bitsizes=(3, 3, 3), block_size=2)
    for i in range(n):
        init = {f'target{i}_': 0 for i in range(3)}
        i_out, *data_out = qroam.call_classically(selection=i, **init)
        assert i_out == i
        assert data_out == [data[i] for data in data_sets]


def test_qroam_wire_symbols():
    n = 10
    rs = np.random.RandomState()
    data_sets = [rs.randint(0, 2**3, size=n) for _ in range(3)]
    qroam = QROAM.build(*data_sets, target_bitsizes=(3, 3, 3), block_size=2)
    assert_wire_symbols_match_expected(qroam, ['In', 'data_a', 'data_b', 'data_c'])


@pytest.mark.parametrize("n", range(40, 200, 20))
def test_t_complexity(n):
    rs = np.random.RandomState()
    data_sets = [rs.randint(0, 2**6, size=n) for _ in range(3)]
    qroam = QROAM.build(*data_sets)
    _, sigma = qroam.call_graph()
    toff_count = sigma.get(TGate()) // 4
    assert toff_count == QR(n, sum(qroam.target_bitsizes))[1]
    qroam = QROAM.build(*data_sets).adjoint()
    _, sigma = qroam.call_graph()
    toff_count = sigma.get(TGate()) // 4
    assert toff_count == QI(n)[1]
