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

from qualtran.bloqs.qrom import QROM


def test_qrom():
    data = np.zeros((10, 10))
    sel_bitsizes = tuple((s - 1).bit_length() for s in data.shape)
    qrom = QROM(data_bitsizes=(4,), selection_bitsizes=sel_bitsizes, data_ranges=data.shape)
    cbloq = qrom.decompose_bloq()
    cqrom = CirqQROM.build(data)
    assert cbloq.t_complexity() == t_complexity(cqrom)
