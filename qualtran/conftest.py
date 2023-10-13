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


def assert_bloq_example_make(bloq_ex: BloqExample):
    status, err = qlt_testing.check_bloq_example_make(bloq_ex)
    if status is qlt_testing.BloqCheckResult.PASS:
        return

    raise AssertionError(err)


def assert_bloq_example_decompose(bloq_ex: BloqExample):
    status, err = qlt_testing.check_bloq_example_decompose(bloq_ex)
    if status is qlt_testing.BloqCheckResult.PASS:
        return
    if status is qlt_testing.BloqCheckResult.NA:
        pytest.skip(err)
    if status is qlt_testing.BloqCheckResult.MISSING:
        pytest.skip(err)

    raise AssertionError(err)


_TESTFUNCS = [('make', assert_bloq_example_make), ('decompose', assert_bloq_example_decompose)]


@pytest.fixture(
    scope="module",
    params=[func for name, func in _TESTFUNCS],
    ids=[name for name, func in _TESTFUNCS],
)
def bloq_autotester(request):
    return request.param
