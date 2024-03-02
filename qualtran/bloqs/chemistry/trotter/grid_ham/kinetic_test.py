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
from qualtran.bloqs.chemistry.trotter.grid_ham.kinetic import _kinetic_energy, KineticEnergy


@pytest.mark.parametrize("nelec, nx", ((2, 10), (6, 8), (8, 12)))
def test_kinetic_bloq(nelec, nx):
    ngrid_x = 2 * nx + 1
    ke = KineticEnergy(nelec, ngrid_x)
    qlt_testing.assert_valid_bloq_decomposition(ke)


def test_kinetic_energy(bloq_autotester):
    bloq_autotester(_kinetic_energy)
