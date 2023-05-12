import cirq
import numpy as np
import pytest

import cirq_qubitization as cq
import cirq_qubitization.cirq_infra.testing as cq_testing

gateset_to_keep = cirq.Gateset(
    cq.And,
    cq.LessThanGate,
    cq.LessThanEqualGate,
    cq.MultiTargetCSwap,
    cq.MultiTargetCNOT,
    cq.MultiControlPauli,
    cirq.H,
    cirq.CCNOT,
    cirq.CNOT,
    cirq.FREDKIN,
    cirq.ControlledGate,
    cirq.AnyUnitaryGateFamily(1),
)


def keep(op: cirq.Operation):
    ret = op in gateset_to_keep
    if op.gate is not None and isinstance(op.gate, cirq.ops.raw_types._InverseCompositeGate):
        ret |= op.gate._original in gateset_to_keep
    return ret


def greedily_allocate_ancilla(circuit: cirq.AbstractCircuit) -> cirq.Circuit:
    greedy_mm = cq.cirq_infra.GreedyQubitManager(prefix="ancilla", maximize_reuse=True)
    circuit = cq.map_clean_and_borrowable_qubits(circuit, qm=greedy_mm)
    assert len(circuit.all_qubits()) < 30
    return circuit


def construct_gate_helper_and_qubit_order(gate):
    g = cq_testing.GateHelper(gate)
    with cq.cirq_infra.memory_management_context():
        circuit = cirq.Circuit(cirq.decompose(g.operation, keep=keep, on_stuck_raise=None))
    ordered_input = sum(g.quregs.values(), start=[])
    qubit_order = cirq.QubitOrder.explicit(ordered_input, fallback=cirq.QubitOrder.DEFAULT)
    return g, qubit_order, circuit


def get_3q_uniform_dirac_notation(signs):
    terms = [
        '0.35|000⟩',
        '0.35|001⟩',
        '0.35|010⟩',
        '0.35|011⟩',
        '0.35|100⟩',
        '0.35|101⟩',
        '0.35|110⟩',
        '0.35|111⟩',
    ]
    ret = terms[0] if signs[0] == '+' else f'-{terms[0]}'
    for c, term in zip(signs[1:], terms[1:]):
        ret = ret + f' {c} {term}'
    return ret


@pytest.mark.parametrize('num_ones', [*range(5, 9)])
@pytest.mark.parametrize('eps', [0.01])
def test_reflection_using_prepare(num_ones, eps):
    data = [1] * num_ones
    prepare_gate = cq.StatePreparationAliasSampling(data, probability_epsilon=eps)
    gate = cq.ReflectionUsingPrepare(prepare_gate)
    g, qubit_order, decomposed_circuit = construct_gate_helper_and_qubit_order(gate)
    decomposed_circuit = greedily_allocate_ancilla(decomposed_circuit)

    initial_state_prep = cirq.Circuit(cirq.H.on_each(*g.quregs['selection']))
    initial_state = cirq.dirac_notation(initial_state_prep.final_state_vector())
    assert initial_state == get_3q_uniform_dirac_notation('++++++++')
    result = cirq.Simulator(dtype=np.complex128).simulate(
        initial_state_prep + decomposed_circuit, qubit_order=qubit_order
    )
    selection = g.quregs['selection']
    prepared_state = result.final_state_vector.reshape(2 ** len(selection), -1).sum(axis=1)
    signs = '-' * num_ones + '+' * (9 - num_ones)
    assert cirq.dirac_notation(prepared_state) == get_3q_uniform_dirac_notation(signs)


def test_reflection_using_prepare_diagram():
    data = [1, 2, 3, 4, 5, 6]
    eps = 0.1
    prepare_gate = cq.StatePreparationAliasSampling(data, probability_epsilon=eps)
    # No control
    gate = cq.ReflectionUsingPrepare(prepare_gate, control_val=None)
    op = gate.on_registers(**gate.registers.get_named_qubits())
    circuit = greedily_allocate_ancilla(cirq.Circuit(cirq.decompose_once(op)))
    cirq.testing.assert_has_diagram(
        circuit,
        '''
               ┌──────────────────────────────┐          ┌──────────────────────────────┐
ancilla_0: ─────sigma_mu───────────────────────────────────sigma_mu─────────────────────────
                │                                          │
ancilla_1: ─────alt────────────────────────────────────────alt──────────────────────────────
                │                                          │
ancilla_2: ─────alt────────────────────────────────────────alt──────────────────────────────
                │                                          │
ancilla_3: ─────alt────────────────────────────────────────alt──────────────────────────────
                │                                          │
ancilla_4: ─────keep───────────────────────────────────────keep─────────────────────────────
                │                                          │
ancilla_5: ─────less_than_equal────────────────────────────less_than_equal──────────────────
                │                                          │
ancilla_6: ─────┼────────────────────────────X────Z───────X┼────────────────────────────────
                │                                 │        │
selection0: ────StatePreparationAliasSampling─────@(0)─────StatePreparationAliasSampling────
                │                                 │        │
selection1: ────selection─────────────────────────@(0)─────selection────────────────────────
                │                                 │        │
selection2: ────selection^-1──────────────────────@(0)─────selection────────────────────────
               └──────────────────────────────┘          └──────────────────────────────┘''',
    )

    # Control on `|1>` state
    gate = cq.ReflectionUsingPrepare(prepare_gate, control_val=1)
    op = gate.on_registers(**gate.registers.get_named_qubits())
    circuit = greedily_allocate_ancilla(cirq.Circuit(cirq.decompose_once(op)))
    cirq.testing.assert_has_diagram(
        circuit,
        '''
ancilla_0: ────sigma_mu───────────────────────────────sigma_mu────────────────────────
               │                                      │
ancilla_1: ────alt────────────────────────────────────alt─────────────────────────────
               │                                      │
ancilla_2: ────alt────────────────────────────────────alt─────────────────────────────
               │                                      │
ancilla_3: ────alt────────────────────────────────────alt─────────────────────────────
               │                                      │
ancilla_4: ────keep───────────────────────────────────keep────────────────────────────
               │                                      │
ancilla_5: ────less_than_equal────────────────────────less_than_equal─────────────────
               │                                      │
control: ──────┼───────────────────────────────Z──────┼───────────────────────────────
               │                               │      │
selection0: ───StatePreparationAliasSampling───@(0)───StatePreparationAliasSampling───
               │                               │      │
selection1: ───selection───────────────────────@(0)───selection───────────────────────
               │                               │      │
selection2: ───selection^-1────────────────────@(0)───selection───────────────────────
''',
    )

    # Control on `|0>` state
    gate = cq.ReflectionUsingPrepare(prepare_gate, control_val=0)
    op = gate.on_registers(**gate.registers.get_named_qubits())
    circuit = greedily_allocate_ancilla(cirq.Circuit(cirq.decompose_once(op)))
    cirq.testing.assert_has_diagram(
        circuit,
        '''
               ┌──────────────────────────────┐          ┌──────────────────────────────┐
ancilla_0: ─────sigma_mu───────────────────────────────────sigma_mu─────────────────────────
                │                                          │
ancilla_1: ─────alt────────────────────────────────────────alt──────────────────────────────
                │                                          │
ancilla_2: ─────alt────────────────────────────────────────alt──────────────────────────────
                │                                          │
ancilla_3: ─────alt────────────────────────────────────────alt──────────────────────────────
                │                                          │
ancilla_4: ─────keep───────────────────────────────────────keep─────────────────────────────
                │                                          │
ancilla_5: ─────less_than_equal────────────────────────────less_than_equal──────────────────
                │                                          │
control: ───────┼────────────────────────────X────Z───────X┼────────────────────────────────
                │                                 │        │
selection0: ────StatePreparationAliasSampling─────@(0)─────StatePreparationAliasSampling────
                │                                 │        │
selection1: ────selection─────────────────────────@(0)─────selection────────────────────────
                │                                 │        │
selection2: ────selection^-1──────────────────────@(0)─────selection────────────────────────
               └──────────────────────────────┘          └──────────────────────────────┘
''',
    )
