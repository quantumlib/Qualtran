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

from qualtran import Bloq
from qualtran.bloqs.basic_gates import TwoBitCSwap, XGate

from .tensor_from_classical import bloq_to_dense_via_classical_action


@pytest.mark.parametrize("bloq", [XGate(), TwoBitCSwap()], ids=str)
def test_tensor_consistent_with_classical(bloq: Bloq):
    from_classical = bloq_to_dense_via_classical_action(bloq)
    from_tensor = bloq.tensor_contract()

    np.testing.assert_allclose(from_classical, from_tensor)
