import dataclasses
import itertools
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Sequence, Tuple, Optional, Iterable
from typing import List

import cirq
import nbformat
import numpy as np
import quimb.tensor as qtn
from nbconvert.preprocessors import ExecutePreprocessor

import cirq_qubitization.quimb_sim as cqq
from cirq_qubitization.gate_with_registers import (
    Register,
    GateWithRegisters,
    munge_classical_arrays,
)
from cirq_qubitization.gate_with_registers import Registers


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


def get_classical_inputs(
    variable_registers: Sequence[Register], fixed_registers: Optional[Dict[Register, int]] = None
) -> Dict[str, np.ndarray]:
    """Get classical all possible classical inputs for a given sequence of registers.

    Args:
        variable_registers: A sequence of registers over which we generate all possible
            bit assignments like `itertools.product`.
        fixed_registers: An optional mapping of additional registers to a fixed integer value,
            which will be broadcast to the correct shape.

    Returns:
        A mapping from register name to numpy array suitable for GateWithRegisters.apply_classical.
    """
    tot_bitsize = sum(reg.bitsize for reg in variable_registers)
    product_bits = np.array(list(itertools.product([0, 1], repeat=tot_bitsize)), dtype=np.uint8)
    n_states = len(product_bits)

    # Split `product_bits` into registers
    ret_inputs: Dict[str, np.ndarray] = {}
    base = 0
    for reg in variable_registers:
        ret_inputs[reg.name] = product_bits[:, base : base + reg.bitsize]
        base += reg.bitsize

    # Split fixed bits into registers
    for reg, v in fixed_registers.items():
        ret_inputs[reg.name] = v * np.ones((n_states, reg.bitsize), dtype=np.uint8)

    return ret_inputs


def _fmt_state(d) -> str:
    """Helper function for strings in TestingTensorSystem"""
    bs = {k: ''.join(f'{b}' for b in v) for k, v in d.items()}
    return ' '.join(f'{k}={v}' for k, v in bs.items())


@dataclasses.dataclass(frozen=True)
class TestingTensorSystem:
    """A collection of objects used for probing a initial+final state tensor network.

    Attributes:
        tn: The tensor network representing initializing kets, a unitary circuit, and
            finalizing bras.
        qubit_frontier: The names of the final indices (by qubit) which may be useful
            for further modification of the tensor network.
        fix: A mapping suitable for `tn.draw(fix=fix)` to lay out tensors
            in a diagram.
        test_input: A mapping from register name to one initial state
        test_output: A mapping from register name to one final state
    """

    tn: qtn.TensorNetwork
    qubit_frontier: Dict[cirq.Qid, int]
    fix: Dict[str, Tuple[float, float]]
    test_input: Dict[str, np.ndarray]
    test_output: Dict[str, np.ndarray]

    @property
    def input_str(self) -> str:
        return _fmt_state(self.test_input)

    @property
    def output_str(self) -> str:
        return _fmt_state(self.test_output)

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
def assert_circuit_inp_out_quimb(
    circuit: cirq.AbstractCircuit,
    qubits: Sequence[cirq.Qid],
    inputs: Sequence[int],
    outputs: Sequence[int],
    atol=1e-8,
):
    """Use Quimb to test that `circuit` behaves correctly on an input.

    Args:
        circuit: The circuit representing the reversible classical operation.
        qubits: The qubits in a definite order.
        inputs: The input state bit values.
        outputs: The (correct) output state bit values.
        atol: The absolute tolerance of the final amplitude found by contracting
            <out | circuit | inp>
    """
    tensors, _, _ = cqq.circuit_to_tensors(
        circuit, dict(zip(qubits, inputs)), dict(zip(qubits, outputs))
    )
    tn = qtn.TensorNetwork(tensors)
    assert np.isclose(tn.contract(), 1, atol=atol)


def yield_test_tensor_networks(
    gate: GateWithRegisters, test_inputs: Dict[str, np.ndarray], test_outputs: Dict[str, np.ndarray]
) -> Iterable[TestingTensorSystem]:
    """For each state in test_inputs/test_outputs, yield a tensor network.

    The tensor network can be used for visualizing or asserting a proper contraction.

    Args:
        gate: A GateWithRegisters
        test_inputs: ndarrays representing classical input conditions, perhaps from
            `get_classical_inputs`.
        test_outputs: ndarrays representing the correct classical output states, perhaps
            from applying `gate.apply_classical(test_inputs)`.
    """

    r = gate.registers
    quregs = r.get_named_qubits()
    operation = gate.on_registers(**quregs)
    circuit = cirq.Circuit(operation)

    test_inputs, n1 = munge_classical_arrays(r, test_inputs)
    test_outputs, n2 = munge_classical_arrays(r, test_outputs)
    if n1 != n2:
        raise ValueError(
            f"The number of test inputs {n1} does not match the number of test outputs {n2}"
        )

    for i in range(n1):
        test_input = {reg.name: test_inputs[reg.name][i, :] for reg in r}
        test_output = {reg.name: test_outputs[reg.name][i, :] for reg in r}

        initial_state: Dict[cirq.Qid, int] = {}
        final_state: Dict[cirq.Qid, int] = {}

        for reg in r:
            initial_state.update(zip(quregs[reg.name], test_input[reg.name]))
            final_state.update(zip(quregs[reg.name], test_output[reg.name]))

        tensors, qubit_frontier, fix = cqq.circuit_to_tensors(circuit, initial_state, final_state)
        tn = qtn.TensorNetwork(tensors)
        yield TestingTensorSystem(tn, qubit_frontier, fix, test_input, test_output)


def assert_gate_inputs_outputs(
    gate: GateWithRegisters,
    test_inputs: Dict[str, np.ndarray],
    test_outputs: Dict[str, np.ndarray],
    atol=1e-8,
):
    """Assert that the gate behaves correctly on each input/output pair using Quimb contraction.

    Args:
        gate: A GateWithRegisters
        test_inputs: ndarrays representing classical input conditions, perhaps from
            `get_classical_inputs`.
        test_outputs: ndarrays representing the correct classical output states, perhaps
            from applying `gate.apply_classical(test_inputs)`.
    """
    for tn in yield_test_tensor_networks(gate, test_inputs, test_outputs):
        amp = tn.tn.contract()
        assert np.isclose(amp, 1, atol=atol)
