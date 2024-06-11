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
import cirq
import pytest

from qualtran.bloqs import basic_gates, mcmt, rotations
from qualtran.bloqs.basic_gates import Hadamard, TGate, Toffoli
from qualtran.bloqs.for_testing.costing import make_example_costing_bloqs
from qualtran.resource_counting import BloqCount, GateCounts, get_cost_value, QECGatesCost


def test_bloq_count():
    algo = make_example_costing_bloqs()

    cost = BloqCount([Toffoli()], 'toffoli')
    tof_count = get_cost_value(algo, cost)

    # `make_example_costing_bloqs` has `func` and `func2`. `func2` has 100 Tof
    assert tof_count == {Toffoli(): 100}

    t_and_tof_count = get_cost_value(algo, BloqCount.for_gateset('t+tof'))
    assert t_and_tof_count == {Toffoli(): 100, TGate(): 2 * 10, TGate().adjoint(): 2 * 10}

    g, _ = algo.call_graph()
    leaf = BloqCount.for_call_graph_leaf_bloqs(g)
    # Note: Toffoli has a decomposition in terms of T gates.
    assert set(leaf.gateset_bloqs) == {Hadamard(), TGate(), TGate().adjoint()}

    t_count = get_cost_value(algo, leaf)
    assert t_count == {TGate(): 2 * 10 + 100 * 4, TGate().adjoint(): 2 * 10, Hadamard(): 2 * 10}

    # count things other than leaf bloqs
    top_level = get_cost_value(algo, BloqCount([bloq for bloq, n in algo.callees], 'top'))
    assert sorted(f'{k}: {v}' for k, v in top_level.items()) == ['Func1: 2', 'Func2: 1']


def test_gate_counts():
    gc = GateCounts(t=100, toffoli=13)
    assert str(gc) == 't: 100, toffoli: 13'

    assert GateCounts(t=10) * 2 == GateCounts(t=20)
    assert 2 * GateCounts(t=10) == GateCounts(t=20)

    assert GateCounts(toffoli=1, cswap=1, and_bloq=1).total_t_count() == 4 + 7 + 4


def test_qec_gates_cost():
    algo = make_example_costing_bloqs()
    gc = get_cost_value(algo, QECGatesCost())
    assert gc == GateCounts(toffoli=100, t=2 * 2 * 10, clifford=2 * 10, depth=2)


@pytest.mark.parametrize(
    ['bloq', 'counts'],
    [
        # T Gate
        [basic_gates.TGate(is_adjoint=False), GateCounts(t=1)],
        # Toffoli
        [basic_gates.Toffoli(), GateCounts(toffoli=1)],
        # CSwap
        [basic_gates.TwoBitCSwap(), GateCounts(cswap=1)],
        # And
        [mcmt.And(), GateCounts(and_bloq=1)],
        # Rotations
        [basic_gates.ZPowGate(exponent=0.1, global_shift=0.0, eps=1e-11), GateCounts(rotation=1)],
        [
            rotations.phase_gradient.PhaseGradientUnitary(
                bitsize=10, exponent=1, is_controlled=False, eps=1e-10
            ),
            GateCounts(rotation=10, depth=1),
        ],
        # Recursive
        [
            mcmt.MultiControlPauli(cvs=(1, 1, 1), target_gate=cirq.X),
            GateCounts(and_bloq=2, depth=2, measurement=2, clifford=3),
        ],
    ],
)
def test_algorithm_summary_counts(bloq, counts):
    assert get_cost_value(bloq, QECGatesCost()) == counts
