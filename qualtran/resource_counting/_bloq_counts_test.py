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
import pytest
import sympy

from qualtran.bloqs import basic_gates, mcmt, rotations
from qualtran.bloqs.basic_gates import Hadamard, TGate, Toffoli
from qualtran.bloqs.basic_gates._shims import Measure
from qualtran.bloqs.for_testing.costing import make_example_costing_bloqs
from qualtran.bloqs.mcmt import MultiAnd, MultiTargetCNOT
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
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
    assert set(leaf.gateset_bloqs) == {Hadamard(), Toffoli(), TGate(), TGate().adjoint()}

    t_count = get_cost_value(algo, leaf)
    assert t_count == {
        Toffoli(): 100,
        TGate(): 2 * 10,
        TGate().adjoint(): 2 * 10,
        Hadamard(): 2 * 10,
    }

    # count things other than leaf bloqs
    top_level = get_cost_value(algo, BloqCount([bloq for bloq, n in algo.callees], 'top'))
    assert sorted(f'{k}: {v}' for k, v in top_level.items()) == ['Func1: 2', 'Func2: 1']


def test_gate_counts():
    gc = GateCounts(t=100, toffoli=13)
    assert str(gc) == 't: 100, toffoli: 13'
    assert gc.asdict() == {'t': 100, 'toffoli': 13}

    assert GateCounts(t=10) * 2 == GateCounts(t=20)
    assert 2 * GateCounts(t=10) == GateCounts(t=20)

    assert GateCounts(toffoli=1, cswap=1, and_bloq=1).total_t_count() == 4 + 7 + 4

    gc2 = GateCounts(t=sympy.Symbol('n'), toffoli=sympy.sympify('0'), cswap=2)
    assert str(gc2) == 't: n, cswap: 2'


def test_gate_counts_toffoli_only():
    gc = GateCounts(toffoli=10, cswap=10, and_bloq=10)
    assert gc.total_toffoli_only() == 30

    gc += GateCounts(t=1)
    with pytest.raises(ValueError):
        _ = gc.total_toffoli_only()

    gc = GateCounts(toffoli=sympy.Symbol('n'))
    assert gc.total_toffoli_only() == sympy.Symbol('n')


def test_qec_gates_cost():
    algo = make_example_costing_bloqs()
    gc = get_cost_value(algo, QECGatesCost())
    assert gc == GateCounts(toffoli=100, t=2 * 2 * 10, clifford=2 * 10)


def test_qec_gates_cost_cbloq():
    bloq = MultiAnd(cvs=(1,) * 5)
    cbloq = bloq.decompose_bloq()
    assert get_cost_value(bloq, QECGatesCost()) == get_cost_value(cbloq, QECGatesCost())


@pytest.mark.parametrize(
    ['bloq', 'counts'],
    [
        # T Gate
        [basic_gates.TGate(is_adjoint=False), GateCounts(t=1)],
        # Toffoli
        [basic_gates.Toffoli(), GateCounts(toffoli=1)],
        # Measure
        [Measure(), GateCounts(measurement=1)],
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
            GateCounts(clifford=2, t=1, rotation=7),
        ],
        # Recursive
        [mcmt.MultiControlX(cvs=(1, 1, 1)), GateCounts(and_bloq=2, measurement=2, clifford=3)],
    ],
)
def test_get_cost_value_qec_gates_cost(bloq, counts):
    assert get_cost_value(bloq, QECGatesCost()) == counts


def test_count_multi_target_cnot():
    b = MultiTargetCNOT(bitsize=12)

    # MultiTargetCNOT can be done in one clifford cycle on the surface code.
    assert get_cost_value(b, QECGatesCost()) == GateCounts(clifford=1)

    # And/or we could respect its decomposition.
    # TODO: https://github.com/quantumlib/Qualtran/issues/1318
    assert get_cost_value(b, QECGatesCost(legacy_shims=True)) == GateCounts(clifford=23)
    assert b.t_complexity() == TComplexity(clifford=23)
