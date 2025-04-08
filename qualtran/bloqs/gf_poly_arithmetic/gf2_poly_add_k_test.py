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
import numpy as np
from galois import Poly

from qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add_k import (
    _gf2_poly_4_8_add_k,
    _gf2_poly_add_k_symbolic,
)
from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.testing import assert_consistent_classical_action


def test_gf2_poly_4_8_add_k(bloq_autotester):
    bloq_autotester(_gf2_poly_4_8_add_k)


def test_gf2_poly_symbolic_add_k(bloq_autotester):
    bloq_autotester(_gf2_poly_add_k_symbolic)


def test_gf2_poly_add_k_resource():
    bloq = _gf2_poly_4_8_add_k.make()
    assert get_cost_value(bloq, QECGatesCost()).total_t_count() == 0
    assert get_cost_value(bloq, QECGatesCost()).clifford == sum(
        np.sum(bloq.qgf_poly.qgf.to_bits(x)) for x in bloq.g_x.coeffs
    )

    bloq = _gf2_poly_add_k_symbolic.make()
    assert get_cost_value(bloq, QECGatesCost()).total_t_count() == 0
    assert get_cost_value(bloq, QECGatesCost()).clifford == bloq.qgf_poly.bitsize


def test_gf2_poly_add_k_classical_sim():
    bloq = _gf2_poly_4_8_add_k.make()
    f_x = Poly(bloq.qgf_poly.qgf.gf_type([0, 1, 2, 3, 4]))
    assert bloq.call_classically(f_x=f_x)[0] == f_x + bloq.g_x  # type: ignore[arg-type]

    f_x_range = np.asarray(
        [
            Poly(bloq.qgf_poly.qgf.gf_type([0, 0, 0, 0, 0])),
            Poly(bloq.qgf_poly.qgf.gf_type([7, 7, 7, 7, 7])),
            Poly(bloq.qgf_poly.qgf.gf_type([0, 0, 3, 5, 7])),
            Poly(bloq.qgf_poly.qgf.gf_type([2, 3, 5, 0, 0])),
        ]
    )
    assert_consistent_classical_action(bloq, f_x=f_x_range)
