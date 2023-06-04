import itertools

import cirq
import numpy as np
import pytest

import cirq_qubitization as cq
from cirq_qubitization.bit_tools import iter_bits
from cirq_qubitization.cirq_infra import testing as cq_testing
from cirq_qubitization.jupyter_tools import execute_notebook


@pytest.mark.parametrize(
    "data", [[[1, 2, 3, 4, 5]], [[1, 2, 3], [4, 5, 10]], [[1], [2], [3], [4], [5], [6]]]
)
@pytest.mark.parametrize("num_controls", [0, 1, 2])
def test_qrom_1d(data, num_controls):
    qrom = cq.QROM.build([np.array(d) for d in data], num_controls=num_controls)
    g = cq_testing.GateHelper(qrom)
    with cq.memory_management_context(cq.GreedyQubitManager('a', maximize_reuse=True)):
        _ = g.all_qubits

    with cq.memory_management_context(cq.GreedyQubitManager('a', maximize_reuse=True)):
        inverse = cirq.Circuit(cirq.decompose(g.operation**-1))

    assert inverse.all_qubits() == g.decomposed_circuit.all_qubits()

    for selection_integer in range(len(data[0])):
        for cval in range(2):
            qubit_vals = {x: 0 for x in g.all_qubits}
            qubit_vals |= zip(
                g.quregs['selection'], iter_bits(selection_integer, g.r['selection'].bitsize)
            )
            if num_controls:
                qubit_vals |= zip(g.quregs['control'], [cval] * num_controls)

            initial_state = [qubit_vals[x] for x in g.all_qubits]
            if cval or not num_controls:
                for ti, d in enumerate(data):
                    target = g.quregs[f"target{ti}"]
                    qubit_vals |= zip(target, iter_bits(d[selection_integer], len(target)))
            final_state = [qubit_vals[x] for x in g.all_qubits]

            cq_testing.assert_circuit_inp_out_cirqsim(
                g.decomposed_circuit, g.all_qubits, initial_state, final_state
            )
            cq_testing.assert_circuit_inp_out_cirqsim(
                g.decomposed_circuit + inverse, g.all_qubits, initial_state, initial_state
            )
            cq_testing.assert_circuit_inp_out_cirqsim(
                g.decomposed_circuit + inverse, g.all_qubits, final_state, final_state
            )


def test_qrom_diagram():
    data = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    qrom = cq.QROM.build(data)
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
    data = [np.array([1, 2]), np.array([3, 5])]
    selection_bitsizes = [(s - 1).bit_length() for s in data[0].shape]
    target_bitsizes = [int(np.max(d)).bit_length() for d in data]
    qrom = cq.QROM(data, selection_bitsizes, target_bitsizes)
    cirq.testing.assert_equivalent_repr(
        qrom, setup_code="import cirq_qubitization\nimport numpy as np"
    )


def test_notebook():
    execute_notebook('qrom')


@pytest.mark.parametrize(
    "data", [[[1, 2, 3, 4, 5]], [[1, 2, 3], [4, 5, 10]], [[1], [2], [3], [4], [5], [6]]]
)
def test_t_complexity(data):
    g = cq_testing.GateHelper(cq.QROM.build([np.array(d) for d in data]))
    n = len(data[0])
    assert cq.t_complexity(g.gate) == cq.t_complexity(g.operation)
    assert cq.t_complexity(g.gate).t == max(0, 4 * n - 8)


@pytest.mark.parametrize(
    "data",
    [[np.arange(6).reshape(2, 3), 4 * np.arange(6).reshape(2, 3)], [np.arange(8).reshape(2, 2, 2)]],
)
@pytest.mark.parametrize("num_controls", [0, 1, 2])
def test_qrom_multi_dim(data, num_controls):
    selection_bitsizes = [(s - 1).bit_length() for s in data[0].shape]
    target_bitsizes = [int(np.max(d)).bit_length() for d in data]
    qrom = cq.QROM(data, selection_bitsizes, target_bitsizes, num_controls=num_controls)
    g = cq_testing.GateHelper(qrom)
    with cq.memory_management_context(cq.GreedyQubitManager('a', maximize_reuse=True)):
        _ = g.all_qubits

    with cq.memory_management_context(cq.GreedyQubitManager('a', maximize_reuse=True)):
        inverse = cirq.Circuit(cirq.decompose(g.operation**-1))

    assert inverse.all_qubits() == g.decomposed_circuit.all_qubits()

    lens = tuple(reg.bitsize for reg in qrom.selection_registers)
    for idxs in itertools.product(*[range(dim) for dim in data[0].shape]):
        qubit_vals = {x: 0 for x in g.all_qubits}
        for cval in range(2):
            if num_controls:
                qubit_vals |= zip(g.quregs['control'], [cval] * num_controls)
            for isel in range(len(idxs)):
                qubit_vals.update(
                    zip(g.quregs[f'selection{isel}'], iter_bits(idxs[isel], lens[isel]))
                )
            initial_state = [qubit_vals[x] for x in g.all_qubits]
            if cval or not num_controls:
                for ti, d in enumerate(data):
                    target = g.quregs[f"target{ti}"]
                    qubit_vals |= zip(target, iter_bits(int(d[idxs]), len(target)))
            final_state = [qubit_vals[x] for x in g.all_qubits]
            qubit_vals = {x: 0 for x in g.all_qubits}
            cq_testing.assert_circuit_inp_out_cirqsim(
                g.decomposed_circuit, g.all_qubits, initial_state, final_state
            )
