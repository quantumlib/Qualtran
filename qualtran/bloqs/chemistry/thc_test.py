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
from qualtran.bloqs.chemistry.thc import PrepareTHC, UniformSuperpositionTHC
from qualtran.testing import execute_notebook


def _make_uniform_superposition():
    from qualtran.bloqs.chemistry.thc import UniformSuperpositionTHC

    num_mu = 10
    num_spin_orb = 4
    return UniformSuperpositionTHC(num_mu=num_mu, num_spin_orb=num_spin_orb)


def _make_prepare():
    from qualtran.bloqs.chemistry.thc import PrepareTHC

    num_mu = 10
    num_spin_orb = 4

    return PrepareTHC(num_mu=num_mu, num_spin_orb=num_spin_orb, keep_bitsize=8)


def test_uniform_superposition():
    num_mu = 10
    num_spin_orb = 4
    usup = UniformSuperpositionTHC(num_mu=num_mu, num_spin_orb=num_spin_orb)
    qlt_testing.assert_valid_bloq_decomposition(usup)


def test_prepare():
    num_mu = 10
    num_spin_orb = 4
    prep = PrepareTHC(num_mu=num_mu, num_spin_orb=num_spin_orb, keep_bitsize=10)
    qlt_testing.assert_valid_bloq_decomposition(prep)


def test_notebook():
    execute_notebook('thc')
