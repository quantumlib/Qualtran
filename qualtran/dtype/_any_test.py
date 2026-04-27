#  Copyright 2026 Google LLC
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


from qualtran.dtype import assert_to_and_from_bits_array_consistent, QAny


def test_qany_to_and_from_bits():
    assert list(QAny(4).to_bits(10)) == [1, 0, 1, 0]

    assert_to_and_from_bits_array_consistent(QAny(4), range(16))
