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
from qualtran.bloqs.for_testing.with_decomposition import (
    TestIndependentParallelCombo,
    TestParallelCombo,
    TestSerialCombo,
)


def test_test_serial_combo():
    qlt_testing.assert_valid_bloq_decomposition(TestSerialCombo())


def test_test_parallel_combo():
    qlt_testing.assert_valid_bloq_decomposition(TestParallelCombo())


def test_test_indep_parallel_combo():
    qlt_testing.assert_valid_bloq_decomposition(TestIndependentParallelCombo())
