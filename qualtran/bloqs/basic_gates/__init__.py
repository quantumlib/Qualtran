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

"""Basic quantum gates.

The bloqs in this module encode gates you'd expect to find in any quantum computing
framework. It includes single-qubit unitary gates like rotations, bit- and phase-flip;
basic multi-qubit unitary gates; states and effects in the Pauli basis; and non-Clifford
gates `TGate` and `Toffoli` which are commonly counted to estimate algorithm resource
requirements.
"""

from .cnot import CNOT
from .hadamard import Hadamard
from .on_each import OnEach
from .rotation import CZPowGate, Rx, Ry, Rz, XPowGate, YPowGate, ZPowGate
from .swap import CSwap, TwoBitCSwap, TwoBitSwap
from .t_gate import TGate
from .toffoli import Toffoli
from .x_basis import MinusEffect, MinusState, PlusEffect, PlusState, XGate
from .y_gate import YGate
from .z_basis import IntEffect, IntState, OneEffect, OneState, ZeroEffect, ZeroState, ZGate
