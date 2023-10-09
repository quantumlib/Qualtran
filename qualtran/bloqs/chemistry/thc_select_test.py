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

import qualtran.testing as qlt_testing
from qualtran.bloqs.chemistry.thc_select import SelectTHC


def _make_select():
    from qualtran.bloqs.chemistry.thc_select import SelectTHC

    num_spat = 4
    num_mu = 8
    return SelectTHC(num_mu=num_mu, num_spat=num_spat)


def test_select_thc():
    num_mu = 10
    num_spin_orb = 2 * 4
    angles = ((0.5,) * num_spin_orb // 2,) * num_mu
    select = SelectTHC(
        num_mu=num_mu, num_spin_orb=num_spin_orb, rotation_angles=angles, num_bits_theta=12
    )
    qlt_testing.assert_valid_bloq_decomposition(select)
