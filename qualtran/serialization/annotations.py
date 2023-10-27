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
from qualtran.protos import annotations_pb2


def t_complexity_to_proto(t_complexity: TComplexity) -> annotations_pb2.TComplexity:
    return annotations_pb2.TComplexity(
        clifford=t_complexity.clifford, rotations=t_complexity.rotations, t=t_complexity.t
    )


def t_complexity_from_proto(t_complexity: annotations_pb2.TComplexity) -> TComplexity:
    return TComplexity(
        clifford=t_complexity.clifford, t=t_complexity.t, rotations=t_complexity.rotations
    )
