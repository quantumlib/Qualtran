import itertools
from functools import cached_property
from typing import Sequence, Tuple

import cirq
import pytest

from cirq_qubitization import UnaryIterationGate, Registers
from cirq_qubitization import testing as cq_testing
from cirq_qubitization.bit_tools import iter_bits


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
    g = cq_testing.GateHelper(gate)
    for n in range(target_bitsize):

        # Initial qubit values
        qubit_vals = {q: 0 for q in g.all_qubits}
        # All controls 'on' to activate circuit
        qubit_vals |= {c: 1 for c in g.quregs['control']}
        # Set selection according to `n`
        qubit_vals |= zip(g.quregs['selection'], iter_bits(n, selection_bitsize))

        initial_state = [qubit_vals[x] for x in g.all_qubits]
        final_state = initial_state.copy()
        final_state[-(n + 1)] = 1
        cq_testing.assert_circuit_inp_out_cirqsim(
            g.circuit, g.all_qubits, initial_state, final_state
        )


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
    max_i, max_j, max_k = target_shape
    i_len, j_len, k_len = tuple(reg.bitsize for reg in gate.selection_registers)
    for i, j, k in itertools.product(range(max_i), range(max_j), range(max_k)):
        qubit_vals = {x: 0 for x in all_qubits}
        # Initialize selection bits appropriately:
        qubit_vals.update(zip(qubit_regs['i'], iter_bits(i, i_len)))
        qubit_vals.update(zip(qubit_regs['j'], iter_bits(j, j_len)))
        qubit_vals.update(zip(qubit_regs['k'], iter_bits(k, k_len)))
        # Construct initial state
        initial_state = [qubit_vals[x] for x in all_qubits]
        # Build correct statevector with selection_integer bit flipped in the target register:
        for reg_name, idx in zip(['t1', 't2', 't3'], [i, j, k]):
            qubit_vals[qubit_regs[reg_name][idx]] = 1
        final_state = [qubit_vals[x] for x in all_qubits]
        cq_testing.assert_circuit_inp_out_cirqsim(circuit, all_qubits, initial_state, final_state)


def test_notebook():
    cq_testing.execute_notebook('unary_iteration')
