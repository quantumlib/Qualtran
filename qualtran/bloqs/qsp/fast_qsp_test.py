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

from .fast_qsp import fast_complementary_polynomial
from .generalized_qsp_test import (
    check_polynomial_pair_on_random_points_on_unit_circle,
    random_qsp_polynomial,
)


# @pytest.mark.parametrize("degree", [4, 5])
@pytest.mark.parametrize("degree", [5])
def test_complementary_polynomial_quick(degree: int):
    random_state = np.random.RandomState(42)

    for _ in range(2):
        P = random_qsp_polynomial(degree, random_state=random_state)
        Q = fast_complementary_polynomial(P, verify=True)
        check_polynomial_pair_on_random_points_on_unit_circle(P, Q, random_state=random_state)
