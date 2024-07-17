#  Copyright 2024 Google LLC
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
import sympy

from qualtran import QUInt
from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.rotations.rz_via_phase_gradient import (
    _rz_via_phase_gradient,
    RzViaPhaseGradient,
)
from qualtran.resource_counting import BloqCount, get_cost_value


def test_examples(bloq_autotester):
    bloq_autotester(_rz_via_phase_gradient)


def test_costs():
    n = sympy.Symbol("n")
    dtype = QUInt(n)
    bloq = RzViaPhaseGradient(angle_dtype=dtype, phasegrad_dtype=dtype)
    # TODO need to improve this to `4 * n - 8` (i.e. Toffoli cost of `n - 2`)
    assert get_cost_value(bloq, BloqCount.for_gateset('t')) == {TGate(): 4 * n - 4}
