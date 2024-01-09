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

from qualtran.bloqs.chemistry.chem_tutorials import fit_linear, gen_random_chem_ham
from qualtran.testing import execute_notebook


def test_resource_estimation_notebook():
    execute_notebook('resource_estimation')


def test_writing_algorithms_notebook():
    execute_notebook('writing_algorithms')


def test_rand_ham():
    tpq, eris = gen_random_chem_ham(4)
    np.testing.assert_allclose(tpq, tpq.T)
    np.testing.assert_allclose(eris, eris.transpose((0, 1, 3, 2)))
    np.testing.assert_allclose(eris, eris.transpose((1, 0, 2, 3)))
    np.testing.assert_allclose(eris, eris.transpose((1, 0, 3, 2)))
    np.testing.assert_allclose(eris, eris.transpose((2, 3, 0, 1)))
    np.testing.assert_allclose(eris, eris.transpose((3, 2, 0, 1)))
    np.testing.assert_allclose(eris, eris.transpose((2, 3, 1, 0)))
    np.testing.assert_allclose(eris, eris.transpose((2, 3, 0, 1)))


def test_fit_linear():
    xs = np.linspace(1, 200, 10)
    ys = 17 * xs**2.0
    slope, _ = fit_linear(np.log(xs), np.log(ys))
    assert np.isclose(slope, 2.0)
    ys = 17 * xs**4.5
    slope, _ = fit_linear(np.log(xs), np.log(ys))
    assert np.isclose(slope, 4.5)
