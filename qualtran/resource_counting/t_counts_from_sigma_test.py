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
import sympy

from qualtran import Bloq
from qualtran.bloqs.basic_gates import (
    CZPowGate,
    Rx,
    Ry,
    Rz,
    TGate,
    Toffoli,
    XPowGate,
    YPowGate,
    ZPowGate,
)
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.resource_counting.t_counts_from_sigma import t_counts_from_sigma
from qualtran.symbolics import SymbolicFloat

EPS: SymbolicFloat = sympy.Symbol("eps")


@pytest.mark.parametrize(
    ("bloq", "t_count"), [(TGate(), 1), pytest.param(Toffoli(), 4, marks=pytest.mark.xfail)]
)
def test_t_counts_from_sigma_known(bloq: Bloq, t_count: int):
    assert t_counts_from_sigma({bloq: 1}) == t_count


@pytest.mark.parametrize(
    "bloq",
    [
        ZPowGate(0.01, eps=EPS),
        Rz(0.01, eps=EPS),
        Rx(0.01, eps=EPS),
        XPowGate(0.01, eps=EPS),
        Ry(0.01, eps=EPS),
        YPowGate(0.01, eps=EPS),
        CZPowGate(0.01, eps=EPS),
    ],
)
def test_t_counts_from_sigma_for_rotation_with_eps(bloq: Bloq):
    expected_t_count = TComplexity.rotation_cost(EPS)
    assert t_counts_from_sigma({bloq: 1}) == expected_t_count


@pytest.mark.parametrize(
    "bloq", [ZPowGate(eps=EPS), XPowGate(eps=EPS), YPowGate(eps=EPS), CZPowGate(eps=EPS)]
)
def test_t_counts_from_sigma_zero(bloq: Bloq):
    assert t_counts_from_sigma({bloq: 1}) == 0
