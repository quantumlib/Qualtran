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

from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.serialization import annotations


def test_t_complexity_to_proto():
    t_complexity = TComplexity(t=10, clifford=100, rotations=1000)
    proto = annotations.t_complexity_to_proto(t_complexity)
    assert (proto.t, proto.clifford, proto.rotations) == (10, 100, 1000)
    assert annotations.t_complexity_from_proto(proto) == t_complexity
