#  Copyright 2024 Google LLC
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
import itertools

import numpy as np
import pytest

import qualtran.testing as qlt_testing
from qualtran import QInt, QUInt
from qualtran.bloqs.arithmetic.lists.has_duplicates import (
    _has_duplicates,
    _has_duplicates_symb,
    _has_duplicates_symb_len,
    HasDuplicates,
)


@pytest.mark.parametrize(
    "bloq_ex",
    [_has_duplicates, _has_duplicates_symb, _has_duplicates_symb_len],
    ids=lambda b: b.name,
)
def test_examples(bloq_autotester, bloq_ex):
    bloq_autotester(bloq_ex)


@pytest.mark.parametrize("bloq_ex", [_has_duplicates, _has_duplicates_symb], ids=lambda b: b.name)
def test_counts(bloq_ex):
    qlt_testing.assert_equivalent_bloq_counts(bloq_ex())


@pytest.mark.parametrize("l", [2, 3, pytest.param(4, marks=pytest.mark.slow)])
@pytest.mark.parametrize(
    "dtype", [QUInt(2), QInt(2), pytest.param(QUInt(3), marks=pytest.mark.slow)]
)
def test_classical_action(l, dtype):
    bloq = HasDuplicates(l, dtype)
    cbloq = bloq.decompose_bloq()

    for xs_t in itertools.product(dtype.get_classical_domain(), repeat=l):
        xs = np.sort(xs_t)
        for flag in [0, 1]:
            np.testing.assert_equal(
                cbloq.call_classically(xs=xs, flag=flag), bloq.call_classically(xs=xs, flag=flag)
            )
