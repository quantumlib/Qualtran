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
from unittest.mock import ANY

import numpy as np
import pytest

from qualtran import QAny, QUInt
from qualtran.bloqs.chemistry.hubbard_model.qubitization import HubbardSpinUpZ, SelectHubbard
from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
    _hubb_majoranna,
    _hubb_majoranna_small,
    _hubb_spin_up_z,
    _hubb_spin_up_z_small,
    _sel_hubb,
)
from qualtran.resource_counting import GateCounts, get_cost_value, QECGatesCost
from qualtran.simulation.classical_sim import do_phased_classical_simulation


def test_sel_hubb_auto(bloq_autotester):
    bloq_autotester(_sel_hubb)


@pytest.mark.parametrize('dim', [*range(2, 10)])
def test_select_t_complexity(dim):
    select = SelectHubbard(x_dim=dim, y_dim=dim, control_val=1)
    cost = get_cost_value(select, QECGatesCost())
    N = 2 * dim * dim
    logN = 2 * (dim - 1).bit_length() + 1
    assert cost == GateCounts(
        cswap=2 * logN, and_bloq=5 * (N // 2) - 2, measurement=5 * (N // 2) - 2, clifford=ANY
    )
    assert cost.total_t_count() == 10 * N + 8 * logN - 8


def test_adjoint_controlled():
    bloq = _sel_hubb()

    adj_ctrl_bloq = bloq.controlled().adjoint()
    ctrl_adj_bloq = bloq.adjoint().controlled()

    assert adj_ctrl_bloq == ctrl_adj_bloq


def test_hubbard_majoranna_symb(bloq_autotester):
    bloq_autotester(_hubb_majoranna)


def test_hubbard_majoranna_small(bloq_autotester):
    bloq_autotester(_hubb_majoranna_small)


def test_hubbard_spin_up_z_symb(bloq_autotester):
    bloq_autotester(_hubb_spin_up_z)


def test_hubbard_spin_up_z(bloq_autotester):
    bloq_autotester(_hubb_spin_up_z_small)


def test_hubbard_spin_up_z_classical():
    rng = np.random.default_rng(52)

    M = 5
    N = M * M
    hsuz = HubbardSpinUpZ(x_dim=M, y_dim=M).decompose_bloq()
    n_trials = 10

    for _ in range(n_trials):
        the_x, the_y = rng.integers(0, M, size=2)

        # The "spin down" part should not affect the simulation.
        all_ones = QUInt(N).to_bits(2**N - 1)  # [1, 1, 1, ..., 1]

        # The "spin up" part will cause a phase only for one selection index
        # [0, .. 1_{the_x, the_y}, .. 0]
        onehot = [0] * N
        onehot[the_x + M * the_y] = 1

        # The bloqs deal with one monolithic target register.
        system = QAny(2 * N).from_bits(all_ones + onehot)

        # Go through all possible x,y selection indices and see if a phase is applied.
        negative_phases = []
        for x in range(M):
            for y in range(M):
                out_vals, phase = do_phased_classical_simulation(
                    hsuz, dict(V=1, x=x, y=y, target=system)
                )
                if phase == 1.0:
                    pass
                elif phase == -1.0:
                    negative_phases.append((x, y))
                else:
                    raise AssertionError(phase)

        # The only place a phase should be applied is at our chosen coordinate.
        assert negative_phases == [(the_x, the_y)]
