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

import cirq

from qualtran.cirq_interop.decompose_protocol import _decompose_once_considering_known_decomposition


def test_known_decomposition_empty_unitary():
    class DecomposeEmptyList(cirq.testing.SingleQubitGate):
        def _decompose_(self, _):
            return []

    gate = DecomposeEmptyList()
    assert _decompose_once_considering_known_decomposition(gate) == []
