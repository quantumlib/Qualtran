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

from qualtran.bloqs.chemistry.single_factorization import (
    SingleFactorization,
    #SingleFactorizationOneBody,
)
from qualtran.testing import assert_valid_bloq_decomposition, execute_notebook


def _make_single_factorization():
    from qualtran.bloqs.chemistry.single_factorization import SingleFactorization

    return SingleFactorization(10, 20, 8)


# def test_single_factorization_one_body():
#     sf_one_body = SingleFactorizationOneBody(10, 12, 8)
#     assert_valid_bloq_decomposition(sf_one_body)


def test_single_factorization():
    sf = SingleFactorization(10, 12, 8)
    assert_valid_bloq_decomposition(sf)


def test_notebook():
    execute_notebook("single_factorization")
