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

from qualtran.bloqs.factoring.mod_sub import MontgomeryModNeg, MontgomeryModSub
from qualtran.testing import assert_valid_bloq_decomposition


@pytest.mark.parametrize('bitsize,p', [(1, 1), (2, 3), (5, 8)])
def test_montgomery_mod_neg_decomp(bitsize, p):
    bloq = MontgomeryModNeg(bitsize=bitsize, p=p)
    assert_valid_bloq_decomposition(bloq)


@pytest.mark.parametrize('bitsize,p', [(1, 1), (2, 3), (5, 8)])
def test_montgomery_mod_sub_decomp(bitsize, p):
    bloq = MontgomeryModSub(bitsize=bitsize, p=p)
    assert_valid_bloq_decomposition(bloq)
