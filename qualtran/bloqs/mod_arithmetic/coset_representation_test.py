#  Copyright 2025 Google LLC
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
import sympy

import qualtran.testing as qlt_testing
from qualtran.bloqs.mod_arithmetic.coset_representation import (
    _init_coset_representation,
    InitCosetRepresntation,
)
from qualtran.resource_counting import get_cost_value, QECGatesCost


def test_init_cost_representation_cost():
    sym_vars = sympy.symbols('c n k N')
    c, n, k, N = sym_vars
    blq = InitCosetRepresntation(c, n, k, N)
    cost = get_cost_value(blq, QECGatesCost())
    assert cost.total_toffoli_only() == 0
    resolver = {v: 10**4 for v in sym_vars}
    upper_bound = (c + n) * (c + n - 1) + c
    assert cost.clifford.subs(resolver) <= upper_bound.subs(resolver)


def test_init_coset_representation(bloq_autotester):
    bloq_autotester(_init_coset_representation)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('coset_representation')
