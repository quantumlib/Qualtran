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
import attrs

from qualtran import Bloq
from qualtran.bloqs.basic_gates import Rz
from qualtran.bloqs.basic_gates.rotation import ZPowGate
from qualtran.bloqs.bookkeeping import ArbitraryClifford
from qualtran.bloqs.chemistry.trotter.hubbard.interaction import _interaction, _interaction_hwp
from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.resource_counting.generalizers import PHI


def catch_rotations(bloq) -> Bloq:
    if isinstance(bloq, Rz):
        if isinstance(bloq.angle, float) and abs(bloq.angle) < 1e-12:
            return ArbitraryClifford(1)
        else:
            return attrs.evolve(bloq, angle=PHI)
    if isinstance(bloq, ZPowGate):
        if isinstance(bloq.exponent, float) and abs(bloq.exponent) < 1e-12:
            return ArbitraryClifford(1)
        else:
            return attrs.evolve(bloq, exponent=PHI, global_shift=0)
    return bloq


def test_hopping_tile(bloq_autotester):
    bloq_autotester(_interaction)


def test_interaction_hwp(bloq_autotester):
    bloq_autotester(_interaction_hwp)


def test_interaction_hwp_bloq_counts():
    bloq = _interaction_hwp()
    costs = get_cost_value(bloq, QECGatesCost(), generalizer=catch_rotations)
    n_rot_par = bloq.length**2 // 2
    assert costs.rotation == 2 * n_rot_par.bit_length()
    assert costs.total_t_count(ts_per_rotation=0) == 2 * 4 * (n_rot_par - n_rot_par.bit_count())


def test_interaction_bloq_counts():
    bloq = _interaction()
    costs = get_cost_value(bloq, QECGatesCost())
    n_rot = bloq.length**2
    assert costs.rotation == n_rot
