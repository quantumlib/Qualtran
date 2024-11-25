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

import sympy

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


def test_t_counts_from_sigma():
    z_eps1, z_eps2, x_eps, y_eps, cz_eps = sympy.symbols('z_eps1, z_eps2, x_eps, y_eps, cz_eps')
    sigma = {
        ZPowGate(eps=z_eps1): 1,
        ZPowGate(eps=z_eps2): 2,
        ZPowGate(0.01, eps=z_eps1): 1,
        ZPowGate(0.01, eps=z_eps2): 2,
        Rz(0.01, eps=z_eps2): 3,
        Rx(0.01, eps=x_eps): 4,
        XPowGate(eps=x_eps): 5,
        XPowGate(0.01, eps=x_eps): 5,
        Ry(0.01, eps=y_eps): 6,
        YPowGate(eps=y_eps): 7,
        YPowGate(0.01, eps=y_eps): 7,
        CZPowGate(eps=cz_eps): 20,
        CZPowGate(0.01, eps=cz_eps): 20,
        TGate(): 100,
        Toffoli(): 200,
    }
    expected_t_count = (
        +100
        + 200 * 4
        + 1 * TComplexity.rotation_cost(z_eps1)
        + 5 * TComplexity.rotation_cost(z_eps2)
        + 9 * TComplexity.rotation_cost(x_eps)
        + 13 * TComplexity.rotation_cost(y_eps)
        + 20 * TComplexity.rotation_cost(cz_eps)
    )
    assert t_counts_from_sigma(sigma) == expected_t_count
