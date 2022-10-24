from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Sequence, Dict, List

import cirq
import numpy as np
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from cirq_qubitization.t_complexity_protocol import t_complexity
from cirq_qubitization.decompose_protocol import decompose

from cirq_qubitization.gate_with_registers import GateWithRegisters, Registers


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
        return self.r.merge_qubits(**self.quregs)

    @cached_property
    def operation(self) -> cirq.Operation:
        """The `gate` applied to example qubits."""
        return self.gate.on_registers(**self.quregs)

    @cached_property
    def circuit(self) -> cirq.Circuit:
        """The `gate` applied to example qubits wrapped in a `cirq.Circuit`."""
        return cirq.Circuit(self.operation)


def assert_circuit_inp_out_cirqsim(
    circuit: cirq.AbstractCircuit,
    qubits: Sequence[cirq.Qid],
    inputs: Sequence[int],
    outputs: Sequence[int],
    decimals: int = 2,
):
    """Use a Cirq simulator to test that `circuit` behaves correctly on an input.

    Args:
        circuit: The circuit representing the reversible classical operation.
        qubits: The qubits in a definite order.
        inputs: The input state bit values.
        outputs: The (correct) output state bit values.
        decimals: The number of decimals of precision to use when comparing
            amplitudes. Reversible classical operations should produce amplitudes
            that are 0 or 1.
    """
    result = cirq.Simulator(dtype=np.complex128).simulate(
        circuit, initial_state=inputs, qubit_order=qubits
    )
    actual = result.dirac_notation(decimals=decimals)[1:-1]
    should_be = "".join(str(x) for x in outputs)
    assert actual == should_be, (actual, should_be)


def execute_notebook(name: str):
    """Execute a jupyter notebook in this directory.

    Args:
        name: The name of the notebook without extension.

    """
    notebook_path = Path(__file__).parent / f"{name}.ipynb"
    with notebook_path.open() as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb)


def assert_decompose_is_consistent_with_t_complexity(val: Any):
    if not hasattr(val, '_t_complexity_'):
        return
    expected = val._t_complexity_()
    from_decomposition = t_complexity(decompose(val), fail_quietly=False)
    assert expected == from_decomposition, f'{expected} != {from_decomposition}'
