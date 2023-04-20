import random

import cirq
import numpy as np
import pytest

import cirq_qubitization
import cirq_qubitization.cirq_infra.testing as cq_testing
from cirq_qubitization.bloq_algos.swap_network import SwapWithZero
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder
from cirq_qubitization.t_complexity_protocol import t_complexity, TComplexity

random.seed(12345)


@pytest.mark.parametrize(
    "selection_bitsize, target_bitsize, n_target_registers",
    [[3, 5, 1], [2, 2, 3], [2, 3, 4], [3, 2, 5], [4, 1, 10]],
)
def test_swap_with_zero_bloq(selection_bitsize, target_bitsize, n_target_registers):
    # Construct the gate.
    gate = SwapWithZero(selection_bitsize, target_bitsize, n_target_registers)

    # # Allocate selection and target qubits.
    # all_qubits = cirq.LineQubit.range(cirq.num_qubits(gate))
    # selection = all_qubits[:selection_bitsize]
    # targets = {
    #     f'target_{i}': all_qubits[st : st + target_bitsize]
    #     for i, st in enumerate(range(selection_bitsize, len(all_qubits), target_bitsize))
    # }
    # # Create a circuit.
    # circuit = cirq.Circuit(gate.on_registers(selection=selection, **targets))

    # Load data[i] in i'th target register; where each register is of size target_bitsize
    data = [random.randint(0, 2**target_bitsize - 1) for _ in range(n_target_registers)]
    target_state = [int(x) for d in data for x in format(d, f"0{target_bitsize}b")]

    expected_state_vector = np.zeros(2**target_bitsize)
    # Iterate on every selection integer.
    for selection_integer in range(len(data)):
        # Load `selection_integer` in the selection register and construct initial state.
        selection_state = [int(x) for x in format(selection_integer, f"0{selection_bitsize}b")]

        bb = CompositeBloqBuilder()
        sel = bb.allocate(selection_bitsize, selection_integer)
        trgs = []
        for i in range(n_target_registers):
            (trg,) = bb.allocate(target_bitsize, data[i])
            trgs.append(trg)

        sel, trgs = bb.add(gate, selection=sel, target=trgs)
        circuit = bb.finalize(sel=sel, trgs=trgs)
        result = circuit.tensor_contract()
        # Get the sub_state_vector corresponding to qubit register `target[0]`.
        result_state_vector = cirq.sub_state_vector(
            result.final_state_vector,
            keep_indices=list(range(selection_bitsize, selection_bitsize + target_bitsize)),
        )
        # Expected state vector should correspond to data[selection_integer] due to the swap.
        expected_state_vector[data[selection_integer]] = 1
        # Assert that result and expected state vectors are equal; reset and continue.
        assert cirq.equal_up_to_global_phase(result_state_vector, expected_state_vector)
        expected_state_vector[data[selection_integer]] = 0


def test_swap_with_zero_gate_diagram():
    gate = cirq_qubitization.SwapWithZeroGate(3, 2, 4)
    q = cirq.LineQubit.range(cirq.num_qubits(gate))
    circuit = cirq.Circuit(gate.as_cirq_op(**gate.registers.split_qubits(q)))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ────@(r⇋0)───
       │
1: ────@(r⇋0)───
       │
2: ────@(r⇋0)───
       │
3: ────swap_0───
       │
4: ────swap_0───
       │
5: ────swap_1───
       │
6: ────swap_1───
       │
7: ────swap_2───
       │
8: ────swap_2───
       │
9: ────swap_3───
       │
10: ───swap_3───
""",
    )


def test_multi_target_cswap():
    qubits = cirq.LineQubit.range(5)
    c, q_x, q_y = qubits[0], qubits[1:3], qubits[3:]
    cswap = cirq_qubitization.MultiTargetCSwap(2).on_registers(
        control=c, target_x=q_x, target_y=q_y
    )
    cswap_approx = cirq_qubitization.MultiTargetCSwapApprox(2).on_registers(
        control=c, target_x=q_x, target_y=q_y
    )
    setup_code = "import cirq\nimport cirq_qubitization"
    cirq.testing.assert_implements_consistent_protocols(cswap, setup_code=setup_code)
    cirq.testing.assert_implements_consistent_protocols(cswap_approx, setup_code=setup_code)
    circuit = cirq.Circuit(cswap, cswap_approx)
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───@──────@(approx)───
      │      │
1: ───×(x)───×(x)────────
      │      │
2: ───×(x)───×(x)────────
      │      │
3: ───×(y)───×(y)────────
      │      │
4: ───×(y)───×(y)────────
    """,
    )
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ---@--------@(approx)---
      |        |
1: ---swap_x---swap_x------
      |        |
2: ---swap_x---swap_x------
      |        |
3: ---swap_y---swap_y------
      |        |
4: ---swap_y---swap_y------
    """,
        use_unicode_characters=False,
    )


def test_multi_target_cswap_make_on():
    qubits = cirq.LineQubit.range(5)
    c, q_x, q_y = qubits[0], qubits[1:3], qubits[3:]
    cswap1 = cirq_qubitization.MultiTargetCSwap(2).on_registers(
        control=c, target_x=q_x, target_y=q_y
    )
    cswap2 = cirq_qubitization.MultiTargetCSwap.make_on(control=c, target_x=q_x, target_y=q_y)
    assert cswap1 == cswap2


def test_notebook():
    cq_testing.execute_notebook('swap_network')


@pytest.mark.parametrize("n", [*range(1, 6)])
def test_t_complexity(n):
    g = cirq_qubitization.MultiTargetCSwap(n)
    cq_testing.assert_decompose_is_consistent_with_t_complexity(g)

    g = cirq_qubitization.MultiTargetCSwapApprox(n)
    cq_testing.assert_decompose_is_consistent_with_t_complexity(g)


@pytest.mark.parametrize(
    "selection_bitsize, target_bitsize, n_target_registers, want",
    [
        [3, 5, 1, TComplexity(t=0, clifford=0)],
        [2, 2, 3, TComplexity(t=16, clifford=86)],
        [2, 3, 4, TComplexity(t=36, clifford=195)],
        [3, 2, 5, TComplexity(t=32, clifford=172)],
        [4, 1, 10, TComplexity(t=36, clifford=189)],
    ],
)
def test_swap_with_zero_t_complexity(selection_bitsize, target_bitsize, n_target_registers, want):
    gate = cirq_qubitization.SwapWithZeroGate(selection_bitsize, target_bitsize, n_target_registers)
    assert want == t_complexity(gate)
