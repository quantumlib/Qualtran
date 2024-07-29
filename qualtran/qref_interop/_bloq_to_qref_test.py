from dev_tools.qualtran_dev_tools.bloq_finder import get_bloq_examples
from qualtran.qref_interop import bloq_to_qref
import pytest

import sympy
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.arithmetic.comparison import LessThanEqual
from qualtran.bloqs.state_preparation import StatePreparationAliasSampling
from qualtran.bloqs.data_loading.qrom import QROM

from qualtran import DecomposeTypeError, DecomposeNotImplementedError
from qref.verification import verify_topology

from qualtran import Bloq, BloqBuilder
from qref.schema_v1 import RoutineV1


@pytest.mark.parametrize("bloq_example", get_bloq_examples())
def test_bloq_examples_can_be_converted_to_qualtran(bloq_example):
    bloq = bloq_example.make()
    try:
        qref_routine = bloq_to_qref(bloq)
        verify_topology(qref_routine)
    except:
        pytest.xfail(f"QREF conversion failing for {bloq}")


@pytest.mark.parametrize("bloq_example", get_bloq_examples())
def test_bloq_examples_can_be_converted_to_qualtran_when_decomposed(bloq_example):
    try:
        bloq = bloq_example.make().decompose_bloq()
    except (DecomposeTypeError, DecomposeNotImplementedError, ValueError) as e:
        pytest.skip(f"QREF conversion not attempted, as bloq decomposition failed with {e}")
    try:
        qref_routine = bloq_to_qref(bloq)
    except:
        pytest.xfail(f"QREF conversion failing for {bloq}")


@pytest.mark.parametrize("bloq_example", get_bloq_examples())
def test_bloq_examples_can_be_converted_to_qualtran_through_call_graph(bloq_example):
    try:
        bloq = bloq_example.make()
    except (DecomposeTypeError, DecomposeNotImplementedError, ValueError) as e:
        pytest.skip(f"QREF conversion not attempted, as extracting callgraph failed with {e}")
    try:
        qref_routine = bloq_to_qref_from_call_graph(bloq)
    except:
        pytest.xfail(f"QREF conversion failing for {bloq}")


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
            'in_selection0_size': 'ceiling(log2(N - 1))',
            'out_selection0_size': 'ceiling(log2(N - 1))',
            'in_selection1_size': 'ceiling(log2(M - 1))',
            'out_selection1_size': 'ceiling(log2(M - 1))',
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
            {"name": "in_sigma_mu", "size": 3, "direction": "input"},
            {"name": "in_alt", "size": 2, "direction": "input"},
            {"name": "in_keep", "size": 3, "direction": "input"},
            {"name": "in_less_than_equal", "size": 1, "direction": "input"},
            {"name": "out_selection", "size": 2, "direction": "output"},
            {"name": "out_sigma_mu", "size": 3, "direction": "output"},
            {"name": "out_alt", "size": 2, "direction": "output"},
            {"name": "out_keep", "size": 3, "direction": "output"},
            {"name": "out_less_than_equal", "size": 1, "direction": "output"},
        ],
        resources=[
            {"name": "clifford", "value": 268, "type": "additive"},
            {"name": "rotations", "value": 2, "type": "additive"},
            {"name": "t", "value": 58, "type": "additive"},
        ],
    )

    return bloq, routine, "alias sampling (not decomposed)"


def _decomposed_alias_sampling() -> tuple[Bloq, RoutineV1, str]:
    bloq = StatePreparationAliasSampling.from_probabilities([0.25, 0.5, 0.25]).decompose_bloq()

    routine = RoutineV1(
        name="CompositeBloq",
        type="CompositeBloq",
        ports=[
            {"name": "in_selection", "size": 2, "direction": "input"},
            {"name": "in_sigma_mu", "size": 3, "direction": "input"},
            {"name": "in_alt", "size": 2, "direction": "input"},
            {"name": "in_keep", "size": 3, "direction": "input"},
            {"name": "in_less_than_equal", "size": 1, "direction": "input"},
            {"name": "out_selection", "size": 2, "direction": "output"},
            {"name": "out_sigma_mu", "size": 3, "direction": "output"},
            {"name": "out_alt", "size": 2, "direction": "output"},
            {"name": "out_keep", "size": 3, "direction": "output"},
            {"name": "out_less_than_equal", "size": 1, "direction": "output"},
        ],
        children=[
            RoutineV1(
                name="Split_1",
                type="Split",
                ports=[
                    {"name": "in_reg", "direction": "input", "size": 3},
                    {"name": "out_reg_0", "direction": "output", "size": 1},
                    {"name": "out_reg_1", "direction": "output", "size": 1},
                    {"name": "out_reg_2", "direction": "output", "size": 1},
                ],
                resources=[
                    {"name": "clifford", "value": 0, "type": "additive"},
                    {"name": "rotations", "value": 0, "type": "additive"},
                    {"name": "t", "value": 0, "type": "additive"},
                ],
            ),
            RoutineV1(
                name="Hadamard_2",
                type="Hadamard",
                ports=[
                    {"name": "in_q", "direction": "input", "size": 1},
                    {"name": "out_q", "direction": "output", "size": 1},
                ],
                resources=[
                    {"name": "clifford", "value": 1, "type": "additive"},
                    {"name": "rotations", "value": 0, "type": "additive"},
                    {"name": "t", "value": 0, "type": "additive"},
                ],
            ),
            RoutineV1(
                name="Hadamard_3",
                type="Hadamard",
                ports=[
                    {"name": "in_q", "direction": "input", "size": 1},
                    {"name": "out_q", "direction": "output", "size": 1},
                ],
                resources=[
                    {"name": "clifford", "value": 1, "type": "additive"},
                    {"name": "rotations", "value": 0, "type": "additive"},
                    {"name": "t", "value": 0, "type": "additive"},
                ],
            ),
            RoutineV1(
                name="Hadamard_4",
                type="Hadamard",
                ports=[
                    {"name": "in_q", "direction": "input", "size": 1},
                    {"name": "out_q", "direction": "output", "size": 1},
                ],
                resources=[
                    {"name": "clifford", "value": 1, "type": "additive"},
                    {"name": "rotations", "value": 0, "type": "additive"},
                    {"name": "t", "value": 0, "type": "additive"},
                ],
            ),
            RoutineV1(
                name="CSwap_8",
                type="CSwap",
                ports=[
                    {"name": "in_ctrl", "direction": "input", "size": 1},
                    {"name": "out_ctrl", "direction": "output", "size": 1},
                    {"name": "in_x", "direction": "input", "size": 2},
                    {"name": "out_x", "direction": "output", "size": 2},
                    {"name": "in_y", "direction": "input", "size": 2},
                    {"name": "out_y", "direction": "output", "size": 2},
                ],
                resources=[
                    {"name": "clifford", "value": 20, "type": "additive"},
                    {"name": "rotations", "value": 0, "type": "additive"},
                    {"name": "t", "value": 14, "type": "additive"},
                ],
            ),
            RoutineV1(
                name="Join_6",
                type="Join",
                ports=[
                    {"name": "out_reg", "direction": "output", "size": 3},
                    {"name": "in_reg_0", "direction": "input", "size": 1},
                    {"name": "in_reg_1", "direction": "input", "size": 1},
                    {"name": "in_reg_2", "direction": "input", "size": 1},
                ],
                resources=[
                    {"name": "clifford", "value": 0, "type": "additive"},
                    {"name": "rotations", "value": 0, "type": "additive"},
                    {"name": "t", "value": 0, "type": "additive"},
                ],
            ),
            RoutineV1(
                name="LessThanEqual_7",
                type="LessThanEqual",
                ports=[
                    {"name": "in_x", "direction": "input", "size": 3},
                    {"name": "out_x", "direction": "output", "size": 3},
                    {"name": "in_y", "direction": "input", "size": 3},
                    {"name": "out_y", "direction": "output", "size": 3},
                    {"name": "in_target", "direction": "input", "size": 1},
                    {"name": "out_target", "direction": "output", "size": 1},
                ],
                resources=[
                    {"name": "clifford", "value": 117, "type": "additive"},
                    {"name": "rotations", "value": 0, "type": "additive"},
                    {"name": "t", "value": 20, "type": "additive"},
                ],
            ),
            RoutineV1(
                name="PrepareUniformSuperposition_0",
                type="PrepareUniformSuperposition",
                ports=[
                    {"name": "in_target", "direction": "input", "size": 2},
                    {"name": "out_target", "direction": "output", "size": 2},
                ],
                resources=[
                    {"name": "clifford", "value": 103, "type": "additive"},
                    {"name": "rotations", "value": 2, "type": "additive"},
                    {"name": "t", "value": 20, "type": "additive"},
                ],
            ),
            RoutineV1(
                name="QROM_5",
                type="QROM",
                ports=[
                    {"name": "in_selection", "direction": "input", "size": 2},
                    {"name": "out_selection", "direction": "output", "size": 2},
                    {"name": "in_target0_", "direction": "input", "size": 2},
                    {"name": "out_target0_", "direction": "output", "size": 2},
                    {"name": "in_target1_", "direction": "input", "size": 3},
                    {"name": "out_target1_", "direction": "output", "size": 3},
                ],
                resources=[
                    {"name": "clifford", "value": 25, "type": "additive"},
                    {"name": "rotations", "value": 0, "type": "additive"},
                    {"name": "t", "value": 4, "type": "additive"},
                ],
            ),
        ],
        connections=[
            {"source": in_, "target": out_}
            for in_, out_ in [
                ("CSwap_8.out_ctrl", "out_less_than_equal"),
                ("CSwap_8.out_x", "out_alt"),
                ("CSwap_8.out_y", "out_selection"),
                ("Hadamard_2.out_q", "Join_6.in_reg_0"),
                ("Hadamard_3.out_q", "Join_6.in_reg_1"),
                ("Hadamard_4.out_q", "Join_6.in_reg_2"),
                ("Join_6.out_reg", "LessThanEqual_7.in_y"),
                ("LessThanEqual_7.out_target", "CSwap_8.in_ctrl"),
                ("LessThanEqual_7.out_x", "out_keep"),
                ("LessThanEqual_7.out_y", "out_sigma_mu"),
                ("PrepareUniformSuperposition_0.out_target", "QROM_5.in_selection"),
                ("QROM_5.out_selection", "CSwap_8.in_y"),
                ("QROM_5.out_target0_", "CSwap_8.in_x"),
                ("QROM_5.out_target1_", "LessThanEqual_7.in_x"),
                ("Split_1.out_reg_0", "Hadamard_2.in_q"),
                ("Split_1.out_reg_1", "Hadamard_3.in_q"),
                ("Split_1.out_reg_2", "Hadamard_4.in_q"),
                ("in_alt", "QROM_5.in_target0_"),
                ("in_keep", "QROM_5.in_target1_"),
                ("in_less_than_equal", "LessThanEqual_7.in_target"),
                ("in_selection", "PrepareUniformSuperposition_0.in_target"),
                ("in_sigma_mu", "Split_1.in_reg"),
            ]
        ],
        resources=[
            {"name": "clifford", "value": 268, "type": "additive"},
            {"name": "rotations", "value": 2, "type": "additive"},
            {"name": "t", "value": 58, "type": "additive"},
        ],
    )

    return bloq, routine, "alias sampling (decomposed)"


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
            _decomposed_alias_sampling(),
        ]
    ],
)
def test_importing_qualtran_object_gives_expected_routine_object(qualtran_object, expected_routine):
    assert bloq_to_qref(qualtran_object).program == expected_routine
