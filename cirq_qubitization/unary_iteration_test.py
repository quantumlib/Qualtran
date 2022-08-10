import pytest
from pathlib import Path
from typing import Sequence

import cirq
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

import cirq_qubitization


class ApplyXToLthQubit(cirq_qubitization.UnaryIterationGate):
    def __init__(self, selection_bitsize: int, target_bitsize: int, control_bitsize: int = 1):
        self._selection_bitsize = selection_bitsize
        self._target_bitsize = target_bitsize
        self._control_bitsize = control_bitsize

    @property
    def control_bitsize(self) -> int:
        return self._control_bitsize

    @property
    def selection_bitsize(self) -> int:
        return self._selection_bitsize

    @property
    def target_bitsize(self) -> int:
        return self._target_bitsize

    @property
    def iteration_length(self) -> int:
        return self._target_bitsize

    def nth_operation(self, n: int, control: cirq.Qid, target: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        return cirq.CNOT(control, target[-(n + 1)])


@pytest.mark.parametrize(
    "selection_bitsize, target_bitsize, control_bitsize", [(3, 5, 1), (2, 4, 2)]
)
def test_unary_iteration(selection_bitsize, target_bitsize, control_bitsize):
    gate = ApplyXToLthQubit(selection_bitsize, target_bitsize, control_bitsize)
    qubit_regs = gate.registers.get_named_qubits()
    control, selection, ancilla, target = (
        qubit_regs["control"],
        qubit_regs["selection"],
        qubit_regs["ancilla"],
        qubit_regs["target"],
    )
    all_qubits = control + selection + ancilla + target

    circuit = cirq.Circuit(gate.on_registers(**qubit_regs))
    sim = cirq.Simulator()
    for n in range(len(target)):
        svals = [int(x) for x in format(n, f"0{len(selection)}b")]
        # turn on control bit to activate circuit:
        qubit_vals = {x: int(x in control) for x in all_qubits}
        # Initialize selection bits appropriately:

        qubit_vals.update({s: sval for s, sval in zip(selection, svals)})

        initial_state = [qubit_vals[x] for x in all_qubits]
        result = sim.simulate(circuit, initial_state=initial_state, qubit_order=all_qubits)
        # Build correct statevector with selection_integer bit flipped in the target register:
        initial_state[-(n + 1)] = 1
        expected_output = "".join(str(x) for x in initial_state)
        assert result.dirac_notation()[1:-1] == expected_output


def test_notebook():
    notebook_path = Path(__file__).parent / "unary_iteration.ipynb"
    with notebook_path.open() as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb)
