import cirq
import pytest

import cirq_qubitization
from cirq_qubitization.bit_tools import iter_bits
from cirq_qubitization.cirq_infra import testing as cq_testing


@pytest.mark.parametrize(
    "data", [[[1, 2, 3, 4, 5]], [[1, 2, 3], [4, 5, 10]], [[1], [2], [3], [4], [5], [6]]]
)
def test_qrom(data):
    qrom = cirq_qubitization.QROM(*data)
    g = cq_testing.GateHelper(qrom)

    for selection_integer in range(qrom.iteration_length):
        qubit_vals = {x: 0 for x in g.all_qubits}
        qubit_vals |= zip(
            g.quregs['selection'], iter_bits(selection_integer, g.r['selection'].bitsize)
        )

        initial_state = [qubit_vals[x] for x in g.all_qubits]
        for ti, d in enumerate(data):
            target = g.quregs[f"target{ti}"]
            qubit_vals |= zip(target, iter_bits(d[selection_integer], len(target)))
        final_state = [qubit_vals[x] for x in g.all_qubits]
        cq_testing.assert_circuit_inp_out_cirqsim(
            g.decomposed_circuit, g.all_qubits, initial_state, final_state
        )


def test_qrom_diagram():
    data = [[1, 2, 3], [4, 5, 6]]
    qrom = cirq_qubitization.QROM(*data)
    q = cirq.LineQubit.range(cirq.num_qubits(qrom))
    circuit = cirq.Circuit(qrom.on_registers(**qrom.registers.split_qubits(q)))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───In───────
      │
1: ───In───────
      │
2: ───QROM_0───
      │
3: ───QROM_0───
      │
4: ───QROM_1───
      │
5: ───QROM_1───
      │
6: ───QROM_1───""",
    )


def test_qrom_repr():
    qrom = cirq_qubitization.QROM([1, 2], [3, 5])
    cirq.testing.assert_equivalent_repr(qrom, setup_code="import cirq_qubitization\n")


def test_notebook():
    cq_testing.execute_notebook('qrom')
