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

from qualtran.surface_code.formulae import code_distance_from_budget, error_at


def test_invert_error_at():
    phys_err = 1e-3
    budgets = np.logspace(-1, -18)
    for budget in budgets:
        d = code_distance_from_budget(phys_err=phys_err, budget=budget)
        assert d % 2 == 1
        assert error_at(phys_err=phys_err, d=d) <= budget
        if d > 3:
            assert error_at(phys_err=phys_err, d=d - 2) > budget
