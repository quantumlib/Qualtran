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
from qualtran import BloqExample


def assert_bloq_example_make_for_pytest(bloq_ex: BloqExample):
    """Wrap `assert_bloq_example_make`.

    Anything other than PASS is a test failure.
    """
    try:
        qlt_testing.assert_bloq_example_make(bloq_ex)
    except qlt_testing.BloqCheckException as bce:
        # No special skip logic
        raise bce from bce


def assert_bloq_example_decompose_for_pytest(bloq_ex: BloqExample):
    """Wrap `assert_bloq_example_decompose`.

    `NA` or `MISSING` result in the test being skipped.
    """
    try:
        qlt_testing.assert_bloq_example_decompose(bloq_ex)
    except qlt_testing.BloqCheckException as bce:
        if bce.check_result is qlt_testing.BloqCheckResult.NA:
            pytest.skip(bce.msg)
        if bce.check_result is qlt_testing.BloqCheckResult.MISSING:
            pytest.skip(bce.msg)

        raise bce from bce


def assert_equivalent_bloq_example_counts_for_pytest(bloq_ex: BloqExample):
    try:
        qlt_testing.assert_equivalent_bloq_example_counts(bloq_ex)
    except qlt_testing.BloqCheckException as bce:
        if bce.check_result in [
            qlt_testing.BloqCheckResult.UNVERIFIED,
            qlt_testing.BloqCheckResult.NA,
            qlt_testing.BloqCheckResult.MISSING,
        ]:
            pytest.skip(bce.msg)

        if bce.check_result == qlt_testing.BloqCheckResult.FAIL:
            pytest.xfail("We are not yet enforcing the 'counts' check.")

        raise bce from bce


_TESTFUNCS = [
    ('make', assert_bloq_example_make_for_pytest),
    ('decompose', assert_bloq_example_decompose_for_pytest),
    ('counts', assert_equivalent_bloq_example_counts_for_pytest),
]


@pytest.fixture(scope="module", params=_TESTFUNCS, ids=[name for name, func in _TESTFUNCS])
def bloq_autotester(request):
    name, func = request.param
    func.check_name = name
    return func
