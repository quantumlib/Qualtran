import pytest
from pathlib import Path
from typing import Sequence, Tuple
from functools import cached_property
import itertools

import cirq
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from cirq_qubitization import UnaryIterationGate, Registers


class ApplyXToLthQubit(UnaryIterationGate):
    def __init__(self, selection_bitsize: int, target_bitsize: int, control_bitsize: int = 1):
        self._selection_bitsize = selection_bitsize
        self._target_bitsize = target_bitsize
        self._control_bitsize = control_bitsize

    @cached_property
    def control_registers(self) -> Registers:
        return Registers.build(control=self._control_bitsize)

    @cached_property
    def selection_registers(self) -> Registers:
        return Registers.build(selection=self._selection_bitsize)

    @cached_property
    def target_registers(self) -> Registers:
        return Registers.build(target=self._target_bitsize)

    @cached_property
    def iteration_lengths(self) -> Tuple[int, ...]:
        return (self._target_bitsize,)

    def nth_operation(
        self, control: cirq.Qid, selection: int, target: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        return cirq.CNOT(control, target[-(selection + 1)])


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


class ApplyXToIJKthQubit(UnaryIterationGate):
    def __init__(self, target_shape: Tuple[int, int, int]):
        self._target_shape = target_shape

    @cached_property
    def control_registers(self) -> Registers:
        return Registers([])

    @cached_property
    def selection_registers(self) -> Registers:
        return Registers.build(
            i=(self._target_shape[0] - 1).bit_length(),
            j=(self._target_shape[1] - 1).bit_length(),
            k=(self._target_shape[2] - 1).bit_length(),
        )

    @cached_property
    def target_registers(self) -> Registers:
        return Registers.build(
            t1=self._target_shape[0], t2=self._target_shape[1], t3=self._target_shape[2]
        )

    @cached_property
    def iteration_lengths(self) -> Tuple[int, ...]:
        return self._target_shape[0], self._target_shape[1], self._target_shape[2]

    def nth_operation(
        self,
        control: cirq.Qid,
        i: int,
        j: int,
        k: int,
        t1: Sequence[cirq.Qid],
        t2: Sequence[cirq.Qid],
        t3: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        yield [cirq.CNOT(control, t1[i]), cirq.CNOT(control, t2[j]), cirq.CNOT(control, t3[k])]


@pytest.mark.parametrize("target_shape", [(2, 3, 2), (2, 2, 2)])
def test_multi_dimensional_unary_iteration(target_shape):
    gate = ApplyXToIJKthQubit(target_shape)
    qubit_regs = gate.registers.get_named_qubits()
    all_qubits = gate.registers.merge_qubits(**qubit_regs)

    circuit = cirq.Circuit(gate.on_registers(**qubit_regs))
    sim = cirq.Simulator()
    max_i, max_j, max_k = target_shape
    i_len, j_len, k_len = tuple(reg.bitsize for reg in gate.selection_registers)
    for i, j, k in itertools.product(range(max_i), range(max_j), range(max_k)):
        qubit_vals = {x: 0 for x in all_qubits}
        # Initialize selection bits appropriately:
        qubit_vals.update({s: int(val) for s, val in zip(qubit_regs['i'], f'{i:0{i_len}b}')})
        qubit_vals.update({s: int(val) for s, val in zip(qubit_regs['j'], f'{j:0{j_len}b}')})
        qubit_vals.update({s: int(val) for s, val in zip(qubit_regs['k'], f'{k:0{k_len}b}')})
        # Construct initial state
        initial_state = [qubit_vals[x] for x in all_qubits]
        result = sim.simulate(circuit, initial_state=initial_state, qubit_order=all_qubits)
        # Build correct statevector with selection_integer bit flipped in the target register:
        for reg_name, idx in zip(['t1', 't2', 't3'], [i, j, k]):
            qubit_vals[qubit_regs[reg_name][idx]] = 1
        final_state = [qubit_vals[x] for x in all_qubits]
        expected_output = "".join(str(x) for x in final_state)
        assert result.dirac_notation()[1:-1] == expected_output


def test_notebook():
    notebook_path = Path(__file__).parent / "unary_iteration.ipynb"
    with notebook_path.open() as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb)
