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
from cirq_ft import QROM as CirqQROM
from cirq_ft.infra import t_complexity

import qualtran.testing as qlt_testing
from qualtran.bloqs.qrom import QROM


def test_qrom_decomp():
    data = np.ones((10, 10))
    sel_bitsizes = tuple((s - 1).bit_length() for s in data.shape)
    qrom = QROM(data=[data], data_bitsizes=(4,), selection_bitsizes=sel_bitsizes)
    qlt_testing.assert_valid_bloq_decomposition(qrom)


def test_qrom_decomp_with_control():
    data = np.ones((10, 10))
    sel_bitsizes = tuple((s - 1).bit_length() for s in data.shape)
    qrom = QROM(data=[data], data_bitsizes=(4,), selection_bitsizes=sel_bitsizes, num_controls=1)
    qlt_testing.assert_valid_bloq_decomposition(qrom)


def test_tcomplexity():
    data = np.ones((10, 10))
    sel_bitsizes = tuple((s - 1).bit_length() for s in data.shape)
    qrom = QROM([data], selection_bitsizes=sel_bitsizes, data_bitsizes=(4,))
    cbloq = qrom.decompose_bloq()
    cqrom = CirqQROM(data=[data], selection_bitsizes=sel_bitsizes, target_bitsizes=(4,))
    assert cbloq.t_complexity() == t_complexity(cqrom)


def test_hashing():
    data = np.ones((10, 10))
    sel_bitsizes = tuple((s - 1).bit_length() for s in data.shape)
    qrom = QROM([data], selection_bitsizes=sel_bitsizes, data_bitsizes=(4,))
    assert hash(qrom) == hash(qrom)
    qrom_2 = QROM([data], selection_bitsizes=sel_bitsizes, data_bitsizes=(5,))
    assert qrom == qrom
    assert qrom_2 != qrom
