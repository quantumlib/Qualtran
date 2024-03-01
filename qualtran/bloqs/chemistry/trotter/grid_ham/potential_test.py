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

import qualtran.testing as qlt_testing
from qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt import (
    build_qrom_data_for_poly_fit,
    get_inverse_square_root_poly_coeffs,
)
from qualtran.bloqs.chemistry.trotter.grid_ham.potential import (
    _pair_potential,
    _potential_energy,
    PairPotential,
    PotentialEnergy,
)


def test_pair_potential(bloq_autotester):
    bloq_autotester(_pair_potential)


def test_potential_energy(bloq_autotester):
    bloq_autotester(_potential_energy)


@pytest.mark.parametrize("nelec, nx", ((2, 10), (6, 8), (8, 12)))
def test_potential_bloq(nelec, nx):
    ngrid_x = 2 * nx + 1
    bitsize = (ngrid_x - 1).bit_length() + 1
    poly_bitsize = 15
    pe = PotentialEnergy(nelec, ngrid_x)
    qlt_testing.assert_valid_bloq_decomposition(pe)
    poly_coeffs = get_inverse_square_root_poly_coeffs()
    qrom_data = build_qrom_data_for_poly_fit(2 * bitsize + 2, poly_bitsize, poly_coeffs)
    qrom_data = tuple(tuple(int(k) for k in d) for d in qrom_data)
    pp = PairPotential(bitsize=bitsize, qrom_data=qrom_data, poly_bitsize=pe.poly_bitsize)
    qlt_testing.assert_valid_bloq_decomposition(pp)
    fac = nelec * (nelec - 1) // 2
    assert fac * pp.t_complexity().t == pe.t_complexity().t
