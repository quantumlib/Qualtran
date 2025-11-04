#  Copyright 2025 Google LLC
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

from qualtran.rotation_synthesis.math_config import NumpyConfig, with_dps
from qualtran.rotation_synthesis.matrix import to_cirq, to_quirk, to_sequence
from qualtran.rotation_synthesis.protocols.clifford_t_synthesis import (
    diagonal_unitary_approx,
    fallback_protocol,
    mixed_diagonal_protocol,
    mixed_fallback_protocol,
)
