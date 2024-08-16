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
from qref.schema_v1 import RoutineV1
from qref.verification import verify_topology

from qualtran import Bloq, BloqBuilder
from qualtran.bloqs.arithmetic.addition import _add_oop_large
from qualtran.bloqs.arithmetic.comparison import LessThanEqual
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.block_encoding.lcu_block_encoding import _black_box_lcu_block, _lcu_block
from qualtran.bloqs.chemistry.df.double_factorization import _df_block_encoding, _df_one_body
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.bloqs.state_preparation import StatePreparationAliasSampling
from qualtran.qref_interop import bloq_to_qref


# This function could be replaced by `get_bloq_examples` from `dev_tools.qualtran_dev_tools.bloq_finder`
# to run tests on all the available bloq examples, rather than a subset defined here.
def get_bloq_examples():
    return [_add_oop_large, _black_box_lcu_block, _lcu_block, _df_one_body, _df_block_encoding]


@pytest.mark.parametrize("bloq_example", get_bloq_examples())
def test_bloq_examples_can_be_converted_to_qualtran(bloq_example):
    bloq = bloq_example.make()
    qref_routine = bloq_to_qref(bloq)
    assert verify_topology(qref_routine)


@pytest.mark.parametrize("bloq_example", get_bloq_examples())
def test_bloq_examples_can_be_converted_to_qualtran_when_decomposed(bloq_example):
    bloq = bloq_example.make().decompose_bloq()
    qref_routine = bloq_to_qref(bloq)
    assert verify_topology(qref_routine)


def _cnot_routine(name: str) -> RoutineV1:
    return RoutineV1(
        name=name,
        type="CNOT",
        ports=[
            {"name": "in_ctrl", "size": 1, "direction": "input"},
            {"name": "in_target", "size": 1, "direction": "input"},
            {"name": "out_ctrl", "size": 1, "direction": "output"},
            {"name": "out_target", "size": 1, "direction": "output"},
        ],
        resources=[
            {"name": "clifford", "value": 1, "type": "additive"},
            {"name": "rotations", "value": 0, "type": "additive"},
            {"name": "t", "value": 0, "type": "additive"},
        ],
    )


def _two_connected_cnots() -> tuple[Bloq, RoutineV1, str]:
    cnot = CNOT()
    bb = BloqBuilder()
    q0 = bb.add_register("q0", 1)
    q1 = bb.add_register("q1", 1)
    q0, q1 = bb.add(cnot, ctrl=q0, target=q1)
    q0, q1 = bb.add(cnot, ctrl=q0, target=q1)
    cbloq = bb.finalize(q0=q0, q1=q1)

    routine = RoutineV1(
        name="CompositeBloq",
        type="CompositeBloq",
        children=[_cnot_routine("CNOT_0"), _cnot_routine("CNOT_1")],
        ports=sorted(
            [
                {"name": "in_q0", "size": 1, "direction": "input"},
                {"name": "in_q1", "size": 1, "direction": "input"},
                {"name": "out_q0", "size": 1, "direction": "output"},
                {"name": "out_q1", "size": 1, "direction": "output"},
            ],
            key=lambda p: p["name"],
        ),
        connections=[
            {"source": "in_q0", "target": "CNOT_0.in_ctrl"},
            {"source": "in_q1", "target": "CNOT_0.in_target"},
            {"source": "CNOT_0.out_ctrl", "target": "CNOT_1.in_ctrl"},
            {"source": "CNOT_0.out_target", "target": "CNOT_1.in_target"},
            {"source": "CNOT_1.out_target", "target": "out_q1"},
            {"source": "CNOT_1.out_ctrl", "target": "out_q0"},
        ],
        resources=[
            {"name": "clifford", "value": 2, "type": "additive"},
            {"name": "rotations", "value": 0, "type": "additive"},
            {"name": "t", "value": 0, "type": "additive"},
        ],
    )

    return cbloq, routine, "two connected CNOTs"


def _two_cross_connected_cnots() -> tuple[Bloq, RoutineV1, str]:
    cnot = CNOT()
    bb = BloqBuilder()
    q0 = bb.add_register("q0", 1)
    q1 = bb.add_register("q1", 1)
    q0, q1 = bb.add(cnot, ctrl=q0, target=q1)
    q0, q1 = bb.add(cnot, ctrl=q1, target=q0)
    cbloq = bb.finalize(q0=q0, q1=q1)

    routine = RoutineV1(
        name="CompositeBloq",
        type="CompositeBloq",
        children=[_cnot_routine("CNOT_0"), _cnot_routine("CNOT_1")],
        ports=[
            {"name": "in_q0", "size": 1, "direction": "input"},
            {"name": "in_q1", "size": 1, "direction": "input"},
            {"name": "out_q0", "size": 1, "direction": "output"},
            {"name": "out_q1", "size": 1, "direction": "output"},
        ],
        connections=[
            {"source": "in_q0", "target": "CNOT_0.in_ctrl"},
            {"source": "in_q1", "target": "CNOT_0.in_target"},
            {"source": "CNOT_0.out_ctrl", "target": "CNOT_1.in_target"},
            {"source": "CNOT_0.out_target", "target": "CNOT_1.in_ctrl"},
            {"source": "CNOT_1.out_target", "target": "out_q1"},
            {"source": "CNOT_1.out_ctrl", "target": "out_q0"},
        ],
        resources=[
            {"name": "clifford", "value": 2, "type": "additive"},
            {"name": "rotations", "value": 0, "type": "additive"},
            {"name": "t", "value": 0, "type": "additive"},
        ],
    )

    return cbloq, routine, "two cross connected CNOTs"


def _less_than_equal() -> tuple[Bloq, RoutineV1, str]:
    a = 10
    b = 15
    bloq = LessThanEqual(a, b)
    routine = RoutineV1(
        name="LessThanEqual",
        type="LessThanEqual",
        ports=[
            {"name": "in_target", "direction": "input", "size": 1},
            {"name": "in_x", "direction": "input", "size": a},
            {"name": "in_y", "direction": "input", "size": b},
            {"name": "out_target", "direction": "output", "size": 1},
            {"name": "out_x", "direction": "output", "size": a},
            {"name": "out_y", "direction": "output", "size": b},
        ],
        resources=[
            {"name": "clifford", "type": "additive", "value": 529},
            {"name": "rotations", "type": "additive", "value": 0},
            {"name": "t", "type": "additive", "value": 96},
        ],
    )

    return bloq, routine, "less than equal (numeric)"


def _less_than_equal_symbolic() -> tuple[Bloq, RoutineV1, str]:
    a, b = sympy.symbols("a b")
    bloq = LessThanEqual(a, b)
    routine = RoutineV1(
        name="LessThanEqual",
        type="LessThanEqual",
        ports=[
            {"name": "in_target", "direction": "input", "size": 1},
            {"name": "in_x", "direction": "input", "size": "a"},
            {"name": "in_y", "direction": "input", "size": "b"},
            {"name": "out_target", "direction": "output", "size": 1},
            {"name": "out_x", "direction": "output", "size": "a"},
            {"name": "out_y", "direction": "output", "size": "b"},
        ],
        resources=[
            {"name": "clifford", "type": "additive", "value": str(bloq.t_complexity().clifford)},
            {"name": "rotations", "type": "additive", "value": str(bloq.t_complexity().rotations)},
            {"name": "t", "type": "additive", "value": str(bloq.t_complexity().t)},
        ],
        input_params=["a", "b"],
    )
    return bloq, routine, "less than equal (symbolic)"


def _qrom_symbolic() -> tuple[Bloq, RoutineV1, str]:
    N, M, b1, b2, c = sympy.symbols("N M b1 b2 c")
    bloq = QROM.build_from_bitsize((N, M), (b1, b2), num_controls=c)
    routine = RoutineV1(
        name="QROM",
        type="QROM",
        ports=[
            {"name": "in_control", "direction": "input", "size": "c"},
            {"name": "in_selection0", "direction": "input", "size": "in_selection0_size"},
            {"name": "in_selection1", "direction": "input", "size": "in_selection1_size"},
            {"name": "in_target0_", "direction": "input", "size": "b1"},
            {"name": "in_target1_", "direction": "input", "size": "b2"},
            {"name": "out_control", "direction": "output", "size": "c"},
            {"name": "out_selection0", "direction": "output", "size": "out_selection0_size"},
            {"name": "out_selection1", "direction": "output", "size": "out_selection1_size"},
            {"name": "out_target0_", "direction": "output", "size": "b1"},
            {"name": "out_target1_", "direction": "output", "size": "b2"},
        ],
        resources=[
            {"name": "clifford", "type": "additive", "value": str(bloq.t_complexity().clifford)},
            {"name": "rotations", "type": "additive", "value": str(bloq.t_complexity().rotations)},
            {"name": "t", "type": "additive", "value": str(bloq.t_complexity().t)},
        ],
        input_params=["M", "N", "b1", "b2", "c"],
        local_variables={
            'in_selection0_size': 'ceiling(log2(floor(N)))',
            'out_selection0_size': 'ceiling(log2(floor(N)))',
            'in_selection1_size': 'ceiling(log2(floor(M)))',
            'out_selection1_size': 'ceiling(log2(floor(M)))',
        },
    )
    return bloq, routine, "qrom (symbolic)"


def _undecomposed_alias_sampling() -> tuple[Bloq, RoutineV1, str]:
    bloq = StatePreparationAliasSampling.from_probabilities([0.25, 0.5, 0.25])

    routine = RoutineV1(
        name="StatePreparationAliasSampling",
        type="StatePreparationAliasSampling",
        ports=[
            {"name": "in_selection", "size": 2, "direction": "input"},
            {"name": "in_sigma_mu", "size": 16, "direction": "input"},
            {"name": "in_alt", "size": 2, "direction": "input"},
            {"name": "in_keep", "size": 16, "direction": "input"},
            {"name": "in_less_than_equal", "size": 1, "direction": "input"},
            {"name": "out_selection", "size": 2, "direction": "output"},
            {"name": "out_sigma_mu", "size": 16, "direction": "output"},
            {"name": "out_alt", "size": 2, "direction": "output"},
            {"name": "out_keep", "size": 16, "direction": "output"},
            {"name": "out_less_than_equal", "size": 1, "direction": "output"},
        ],
        resources=[
            {"name": "clifford", "value": 879, "type": "additive"},
            {"name": "rotations", "value": 2, "type": "additive"},
            {"name": "t", "value": 162, "type": "additive"},
        ],
    )

    return bloq, routine, "alias sampling (not decomposed)"


@pytest.mark.parametrize(
    "qualtran_object, expected_routine",
    [
        pytest.param(operation, bloq, id=case_id)
        for operation, bloq, case_id in [
            _two_connected_cnots(),
            _two_cross_connected_cnots(),
            _less_than_equal(),
            _less_than_equal_symbolic(),
            _qrom_symbolic(),
            _undecomposed_alias_sampling(),
        ]
    ],
)
def test_importing_qualtran_object_gives_expected_routine_object(qualtran_object, expected_routine):
    assert bloq_to_qref(qualtran_object).program == expected_routine
