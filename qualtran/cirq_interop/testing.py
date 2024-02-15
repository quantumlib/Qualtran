#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Sequence, Tuple

import cirq
import numpy as np
from numpy.typing import NDArray

from qualtran import Signature
from qualtran._infra.gate_with_registers import GateWithRegisters, get_named_qubits, merge_qubits
from qualtran.cirq_interop.decompose_protocol import _decompose_once_considering_known_decomposition
from qualtran.cirq_interop.t_complexity_protocol import t_complexity


@dataclass(frozen=True)
class GateHelper:
    """A collection of related objects derivable from a `GateWithRegisters`.

    These are likely useful to have at one's fingertips while writing tests or
    demo notebooks.

    Attributes:
        gate: The gate from which all other objects are derived.
    """

    gate: GateWithRegisters
    context: cirq.DecompositionContext = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())

    @cached_property
    def r(self) -> Signature:
        """The Signature system for the gate."""
        return self.gate.signature

    @cached_property
    def quregs(self) -> Dict[str, NDArray[cirq.Qid]]:  # type: ignore[type-var]
        """A dictionary of named qubits appropriate for the signature for the gate."""
        return get_named_qubits(self.r)

    @cached_property
    def all_qubits(self) -> List[cirq.Qid]:
        """All qubits in Register order."""
        merged_qubits = merge_qubits(self.r, **self.quregs)
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
        return cirq.Circuit(cirq.decompose(self.operation, context=self.context))


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
    actual, should_be = get_circuit_inp_out_cirqsim(circuit, qubit_order, inputs, outputs, decimals)
    assert actual == should_be, f'\n{actual=}\n{should_be=}'


def get_circuit_inp_out_cirqsim(
    circuit: cirq.AbstractCircuit,
    qubit_order: Sequence[cirq.Qid],
    inputs: Sequence[int],
    outputs: Sequence[int],
    decimals: int = 2,
) -> Tuple[str, str]:
    """Use a Cirq simulator to get a outputs of a `circuit`.

    Args:
        circuit: The circuit representing the reversible classical operation.
        qubit_order: The qubit order to pass to the cirq simulator.
        inputs: The input state bit values.
        outputs: The (correct) output state bit values.
        decimals: The number of decimals of precision to use when comparing
            amplitudes. Reversible classical operations should produce amplitudes
            that are 0 or 1.

    Returns:
        actual: The simulated output state as a string bitstring.
        should_be: The outputs argument formatted as a string bitstring for ease of comparison.
    """
    result = cirq.Simulator(dtype=np.complex128).simulate(
        circuit, initial_state=inputs, qubit_order=qubit_order
    )
    actual = result.dirac_notation(decimals=decimals)[1:-1]
    should_be = "".join(str(x) for x in outputs)
    return actual, should_be


def assert_decompose_is_consistent_with_t_complexity(val: Any):
    for method in ['_t_complexity_', 't_complexity']:
        t_complexity_method = getattr(val, method, None)
        expected = NotImplemented if t_complexity_method is None else t_complexity_method()
        if not (expected is NotImplemented or expected is None):
            break
    if expected is NotImplemented or expected is None:
        raise AssertionError("No consistent t_complexity: no _t_complexity_.")
    decomposition = _decompose_once_considering_known_decomposition(val)
    if decomposition is None:
        raise AssertionError("No consistent t_complexity: no decomposition.")
    from_decomposition = t_complexity(decomposition, fail_quietly=False)
    assert expected == from_decomposition, f'{expected} != {from_decomposition}'
