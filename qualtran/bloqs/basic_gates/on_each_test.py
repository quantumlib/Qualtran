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
from functools import reduce

import numpy as np

import qualtran.testing as qlt_testing
from qualtran.bloqs.basic_gates import Hadamard, XGate
from qualtran.bloqs.basic_gates.on_each import OnEach


def test_valid_bloq():
    on_each = OnEach(10, XGate())
    qlt_testing.assert_valid_bloq_decomposition(on_each)


def test_tensor_contract():
    bloq = OnEach(5, Hadamard())
    tensor = bloq.tensor_contract()
    single_had = Hadamard().tensor_contract()
    np.testing.assert_allclose(tensor, reduce(np.kron, (single_had,) * 5))
