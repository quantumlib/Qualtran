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

"""Bi-directional interop between Qualtran & Cirq using Cirq-FT.

isort:skip_file
"""

from ._cirq_to_bloq import (
    CirqQuregT,
    CirqGateAsBloq,
    CirqGateAsBloqBase,
    cirq_optree_to_cbloq,
    cirq_gate_to_bloq,
    decompose_from_cirq_style_method,
)

from ._bloq_to_cirq import BloqAsCirqGate
