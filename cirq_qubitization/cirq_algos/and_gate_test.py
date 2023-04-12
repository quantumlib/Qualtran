import itertools
import random
from typing import List, Tuple

import cirq
import numpy as np
import pytest

import cirq_qubitization
import cirq_qubitization.cirq_infra.testing as cq_testing
from cirq_qubitization.cirq_algos.and_gate import And

random.seed(12345)


@pytest.mark.parametrize("decomp", [False, True])
@pytest.mark.parametrize("cv", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_and_gate(cv: Tuple[int, int], decomp: bool):
    c1, c2, t = cirq.LineQubit.range(3)
    input_states = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]
    output_states = [inp[:2] + (1 if inp[:2] == cv else 0,) for inp in input_states]

    operation = cirq_qubitization.And(cv).on(c1, c2, t)
    if decomp:
        circuit = cirq.Circuit(cirq.decompose_once(operation))
    else:
        circuit = cirq.Circuit(operation)
    for inp, out in zip(input_states, output_states):
        cq_testing.assert_circuit_inp_out_cirqsim(circuit, [c1, c2, t], inp, out)


def random_cv(n: int) -> List[int]:
    return [random.randint(0, 1) for _ in range(n)]


@pytest.mark.parametrize("cv", [[1] * 3, random_cv(5), random_cv(6), random_cv(7)])
def test_multi_controlled_and_gate(cv: List[int]):
    gate = cirq_qubitization.And(cv)
    r = gate.registers
    assert r['ancilla'].bitsize == r['control'].bitsize - 2
    quregs = r.get_named_qubits()
    and_op = gate.on_registers(**quregs)
    circuit = cirq.Circuit(and_op)

    input_controls = [cv] + [random_cv(len(cv)) for _ in range(10)]
    qubit_order = gate.registers.merge_qubits(**quregs)

    for input_control in input_controls:
        initial_state = input_control + [0] * (r['ancilla'].bitsize + 1)
        result = cirq.Simulator().simulate(
            circuit, initial_state=initial_state, qubit_order=qubit_order
        )
        expected_output = np.asarray([0, 1] if input_control == cv else [1, 0])
        assert cirq.equal_up_to_global_phase(
            cirq.sub_state_vector(
                result.final_state_vector, keep_indices=[cirq.num_qubits(gate) - 1]
            ),
            expected_output,
        )

        # Test adjoint.
        cq_testing.assert_circuit_inp_out_cirqsim(
            circuit + cirq.Circuit(and_op**-1),
            qubit_order=qubit_order,
            inputs=initial_state,
            outputs=initial_state,
        )


def test_and_gate_diagram():
    gate = cirq_qubitization.And((1, 0, 1, 0, 1, 0))
    qubit_regs = gate.registers.get_named_qubits()
    op = gate.on_registers(**qubit_regs)
    # Qubit order should be alternating (control, ancilla) pairs.
    c_and_a = sum(zip(qubit_regs["control"][1:], qubit_regs["ancilla"] + [0]), ())[:-1]
    qubit_order = qubit_regs["control"][0:1] + list(c_and_a) + qubit_regs["target"]
    # Test diagrams.
    cirq.testing.assert_has_diagram(
        cirq.Circuit(op),
        """
control0: ───@─────
             │
control1: ───(0)───
             │
ancilla0: ───Anc───
             │
control2: ───@─────
             │
ancilla1: ───Anc───
             │
control3: ───(0)───
             │
ancilla2: ───Anc───
             │
control4: ───@─────
             │
ancilla3: ───Anc───
             │
control5: ───(0)───
             │
target: ─────And───
""",
        qubit_order=qubit_order,
    )
    cirq.testing.assert_has_diagram(
        cirq.Circuit(op**-1),
        """
control0: ───@──────
             │
control1: ───(0)────
             │
ancilla0: ───Anc────
             │
control2: ───@──────
             │
ancilla1: ───Anc────
             │
control3: ───(0)────
             │
ancilla2: ───Anc────
             │
control4: ───@──────
             │
ancilla3: ───Anc────
             │
control5: ───(0)────
             │
target: ─────And†───
    """,
        qubit_order=qubit_order,
    )
    # Test diagram of decomposed 3-qubit and ladder.
    decomposed_circuit = cirq.Circuit(cirq.decompose_once(op)) + cirq.Circuit(
        cirq.decompose_once(op**-1)
    )
    cirq.testing.assert_has_diagram(
        decomposed_circuit,
        """
control0: ───@─────────────────────────────────────────────────────────@──────
             │                                                         │
control1: ───(0)───────────────────────────────────────────────────────(0)────
             │                                                         │
ancilla0: ───And───@────────────────────────────────────────────@──────And†───
                   │                                            │
control2: ─────────@────────────────────────────────────────────@─────────────
                   │                                            │
ancilla1: ─────────And───@───────────────────────────────@──────And†──────────
                         │                               │
control3: ───────────────(0)─────────────────────────────(0)──────────────────
                         │                               │
ancilla2: ───────────────And───@──────────────────@──────And†─────────────────
                               │                  │
control4: ─────────────────────@──────────────────@───────────────────────────
                               │                  │
ancilla3: ─────────────────────And───@─────@──────And†────────────────────────
                                     │     │
control5: ───────────────────────────(0)───(0)────────────────────────────────
                                     │     │
target: ─────────────────────────────And───And†───────────────────────────────
    """,
        qubit_order=qubit_order,
    )


@pytest.mark.parametrize(
    "cv, adjoint, str_output",
    [
        ((1, 1, 1), False, "And"),
        ((1, 0, 1), False, "And(1, 0, 1)"),
        ((1, 1, 1), True, "And†"),
        ((1, 0, 1), True, "And†(1, 0, 1)"),
    ],
)
def test_and_gate_str_and_repr(cv, adjoint, str_output):
    gate = cirq_qubitization.And(cv, adjoint=adjoint)
    assert str(gate) == str_output
    cirq.testing.assert_equivalent_repr(gate, setup_code="import cirq_qubitization\n")


@pytest.mark.parametrize("decomp", [False, True])
@pytest.mark.parametrize("cv", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_and_gate_adjoint(cv: Tuple[int, int], decomp: bool):
    c1, c2, t = cirq.LineQubit.range(3)
    all_cvs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    input_states = [inp + (1 if inp == cv else 0,) for inp in all_cvs]
    output_states = [inp + (0,) for inp in all_cvs]

    operation = cirq_qubitization.And(cv, adjoint=True).on(c1, c2, t)
    if decomp:
        circuit = cirq.Circuit(cirq.decompose_once(operation))
    else:
        circuit = cirq.Circuit(operation)
    for inp, out in zip(input_states, output_states):
        cq_testing.assert_circuit_inp_out_cirqsim(circuit, [c1, c2, t], inp, out)


def test_notebook():
    cq_testing.execute_notebook('and_gate')


@pytest.mark.parametrize(
    "C", [*itertools.chain(*[itertools.product(range(2), repeat=n) for n in range(2, 7 + 1)])]
)
@pytest.mark.parametrize("adjoint", [*range(2)])
def test_t_complexity(adjoint, C):
    cq_testing.assert_decompose_is_consistent_with_t_complexity(And(C, adjoint=adjoint))
