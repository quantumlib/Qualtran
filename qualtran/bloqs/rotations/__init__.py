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

from qualtran.bloqs.rotations.hamming_weight_phasing import (
    HammingWeightPhasing,
    HammingWeightPhasingViaPhaseGradient,
)
from qualtran.bloqs.rotations.phase_gradient import (
    AddIntoPhaseGrad,
    AddScaledValIntoPhaseReg,
    PhaseGradientState,
)
from qualtran.bloqs.rotations.phasing_via_cost_function import PhasingViaCostFunction
from qualtran.bloqs.rotations.programmable_rotation_gate_array import ProgrammableRotationGateArray
from qualtran.bloqs.rotations.quantum_variable_rotation import QvrPhaseGradient, QvrZPow
