from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cirq
import nbformat
import numpy as np
from nbconvert.preprocessors import ExecutePreprocessor

from cirq_qubitization.cirq_infra.decompose_protocol import decompose_once_into_operations
from cirq_qubitization.cirq_infra.gate_with_registers import GateWithRegisters, Registers
from cirq_qubitization.t_complexity_protocol import t_complexity


@dataclass(frozen=True)
class GateHelper:
    """A collection of related objects derivable from a `GateWithRegisters`.

    These are likely useful to have at one's fingertips while writing tests or
    demo notebooks.

    Attributes:
        gate: The gate from which all other objects are derived.
    """

    gate: GateWithRegisters

    @cached_property
    def r(self) -> Registers:
        """The Registers system for the gate."""
        return self.gate.registers

    @cached_property
    def quregs(self) -> Dict[str, List[cirq.Qid]]:
        """A dictionary of named qubits appropriate for the registers for the gate."""
        return self.r.get_named_qubits()

    @cached_property
    def all_qubits(self) -> List[cirq.Qid]:
        """All qubits in Register order."""
        merged_qubits = self.r.merge_qubits(**self.quregs)
        decomposed_qubits = self.decomposed_circuit.all_qubits()
        return merged_qubits + sorted(decomposed_qubits - frozenset(merged_qubits))

    @cached_property
    def operation(self) -> cirq.Operation:
        """The `gate` applied to example qubits."""
        return self.gate.on_registers(**self.quregs)

    @cached_property
    def circuit(self) -> cirq.Circuit:
        """The `gate` applied to example qubits wrapped in a `cirq.Circuit`."""
        return cirq.Circuit(self.operation)

    @cached_property
    def decomposed_circuit(self) -> cirq.Circuit:
        """The `gate` applied to example qubits, decomposed and wrapped in a `cirq.Circuit`."""
        return cirq.Circuit(cirq.decompose(self.operation))


def assert_circuit_inp_out_cirqsim(
    circuit: cirq.AbstractCircuit,
    qubit_order: Sequence[cirq.Qid],
    inputs: Sequence[int],
    outputs: Sequence[int],
    decimals: int = 2,
):
    """Use a Cirq simulator to test that `circuit` behaves correctly on an input.

    Args:
        circuit: The circuit representing the reversible classical operation.
        qubit_order: The qubit order to pass to the cirq simulator.
        inputs: The input state bit values.
        outputs: The (correct) output state bit values.
        decimals: The number of decimals of precision to use when comparing
            amplitudes. Reversible classical operations should produce amplitudes
            that are 0 or 1.
    """
    actual, should_be = find_circuit_inp_out_cirqsim(
        circuit, qubit_order, inputs, outputs, decimals
    )
    assert actual == should_be, (actual, should_be)


def find_circuit_inp_out_cirqsim(
    circuit: cirq.AbstractCircuit,
    qubit_order: Sequence[cirq.Qid],
    inputs: Sequence[int],
    outputs: Sequence[int],
    decimals: int = 2,
) -> Tuple[str, str]:
    """Use a Cirq simulator to test that `circuit` behaves correctly on an input.

    Args:
        circuit: The circuit representing the reversible classical operation.
        qubit_order: The qubit order to pass to the cirq simulator.
        inputs: The input state bit values.
        outputs: The (correct) output state bit values.
        decimals: The number of decimals of precision to use when comparing
            amplitudes. Reversible classical operations should produce amplitudes
            that are 0 or 1.

    Returns:
        actual: (bit) string representation of the simulated state.
        should_be: (bit) string representation of the expected state defined by outputs.
    """
    result = cirq.Simulator(dtype=np.complex128).simulate(
        circuit, initial_state=inputs, qubit_order=qubit_order
    )
    actual = result.dirac_notation(decimals=decimals)[1:-1]
    should_be = "".join(str(x) for x in outputs)
    return actual, should_be


def execute_notebook(name: str):
    """Execute a jupyter notebook in this directory.

    Args:
        name: The name of the notebook without extension.

    """
    import traceback

    # Assumes that the notebook is in the same path from where the function was called,
    # which may be different from `__file__`.
    notebook_path = Path(traceback.extract_stack()[-2].filename).parent / f"{name}.ipynb"
    with notebook_path.open() as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb)


def assert_decompose_is_consistent_with_t_complexity(val: Any):
    t_complexity_method = getattr(val, '_t_complexity_', None)
    expected = NotImplemented if t_complexity_method is None else t_complexity_method()
    if expected is NotImplemented or expected is None:
        return
    decomposition = decompose_once_into_operations(val)
    if decomposition is None:
        return
    from_decomposition = t_complexity(decomposition, fail_quietly=False)
    assert expected == from_decomposition, f'{expected} != {from_decomposition}'
