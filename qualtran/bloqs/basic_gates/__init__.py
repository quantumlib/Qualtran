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

from qualtran.bloqs.basic_gates.cnot import CNOT
from qualtran.bloqs.basic_gates.global_phase import GlobalPhase
from qualtran.bloqs.basic_gates.hadamard import Hadamard
from qualtran.bloqs.basic_gates.on_each import OnEach
from qualtran.bloqs.basic_gates.rotation import CZPowGate, Rx, Ry, Rz, XPowGate, YPowGate, ZPowGate
from qualtran.bloqs.basic_gates.s_gate import SGate
from qualtran.bloqs.basic_gates.su2_rotation import SU2RotationGate
from qualtran.bloqs.basic_gates.swap import CSwap, Swap, TwoBitCSwap, TwoBitSwap
from qualtran.bloqs.basic_gates.t_gate import TGate
from qualtran.bloqs.basic_gates.toffoli import Toffoli
from qualtran.bloqs.basic_gates.x_basis import MinusEffect, MinusState, PlusEffect, PlusState, XGate
from qualtran.bloqs.basic_gates.y_gate import YGate
from qualtran.bloqs.basic_gates.z_basis import (
    IntEffect,
    IntState,
    OneEffect,
    OneState,
    ZeroEffect,
    ZeroState,
    ZGate,
)
