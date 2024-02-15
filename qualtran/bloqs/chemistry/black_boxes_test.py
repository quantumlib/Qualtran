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
from openfermion.resource_estimates.utils import QI, QR

from qualtran.bloqs.chemistry.black_boxes import get_qroam_cost


@pytest.mark.parametrize("data_size, bitsize", ((100, 10), (100, 3), (1_000, 13), (1_000_000, 20)))
def test_qroam_factors(data_size, bitsize):
    assert get_qroam_cost(data_size, bitsize) == QR(data_size, bitsize)[-1]
    assert get_qroam_cost(data_size, bitsize, adjoint=True) == QI(data_size)[-1]
