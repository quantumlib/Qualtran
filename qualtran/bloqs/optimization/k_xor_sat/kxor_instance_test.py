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

from qualtran.bloqs.optimization.k_xor_sat.kxor_instance import KXorInstance


@pytest.mark.slow
@pytest.mark.parametrize("rho", [0, 0.8, 0.9])
def test_max_rhs(rho: float):
    rng = np.random.default_rng(402)

    rhs = []
    for i in range(100):
        inst = KXorInstance.random_instance(n=100, m=1000, k=4, planted_advantage=rho, rng=rng)
        rhs.append(inst.max_rhs)

    assert max(rhs) == 2
