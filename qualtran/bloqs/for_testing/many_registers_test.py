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
from qualtran.bloqs.for_testing.many_registers import TestMultiRegister


def test_test_multi_register():
    bloq = TestMultiRegister()
    assert [r.name for r in bloq.signature] == ['xx', 'yy', 'zz']
    assert sum(r.total_bits() for r in bloq.signature) == 12

    qlt_testing.assert_valid_bloq_decomposition(bloq)
