from dataclasses import dataclass
from functools import cached_property
from typing import Sequence, Dict, List

import cirq
import numpy as np

from cirq_qubitization.gate_with_registers import GateWithRegisters, Registers


@dataclass(frozen=True)
class GateSystem:
    """A collection of related objects derivable from a `GateWithRegisters`.

    This are likely useful to have at ones fingertips while writing tests or
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
    def quregs(self) -> Dict[str, Sequence[cirq.Qid]]:
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
