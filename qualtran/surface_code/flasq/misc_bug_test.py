#  Copyright 2024 Google LLC
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

import cirq
import numpy as np
import pytest
from attrs import frozen

from qualtran import Bloq, BloqBuilder, QAny, Register, Side, Signature
from qualtran.bloqs.mcmt import And
from qualtran.surface_code.flasq.cirq_interop import convert_circuit_for_flasq_analysis
from qualtran.surface_code.flasq.examples.hwp import build_hwp_circuit
from qualtran.surface_code.flasq.naive_grid_qubit_manager import NaiveGridQubitManager


@frozen
class _TestThru(Bloq):
    n_bits: int

    @property
    def signature(self) -> "Signature":
        return Signature([Register("q", QAny(self.n_bits))])

    def __str__(self):
        return f"TestThru({self.n_bits})"


@frozen
class _TestTwoIn(Bloq):
    n_bits_a: int
    n_bits_b: int

    @property
    def signature(self) -> "Signature":
        return Signature(
            [
                Register("a", QAny(self.n_bits_a), side=Side.LEFT),
                Register("b", QAny(self.n_bits_b), side=Side.LEFT),
                Register("c", QAny(self.n_bits_a + self.n_bits_b), side=Side.RIGHT),
            ]
        )

    def __str__(self):
        return f"TestTwoIn({self.n_bits_a}, {self.n_bits_b})"


def test_hierarchical_bloq_flatten_error_is_caught():
    """Verifies that flattening a CompositeBloq containing another CompositeBloq works.

    This historically failed because the flatten logic tried to call `decompose_bloq()`
    on a CompositeBloq, which was not allowed.
    """

    bb_inner = BloqBuilder()
    a_in = bb_inner.add_register(Register("a", QAny(2), side=Side.LEFT))
    b_in = bb_inner.add_register(Register("b", QAny(3), side=Side.LEFT))
    bb_inner.add_register(Register("c", QAny(5), side=Side.RIGHT))
    assert a_in is not None
    assert b_in is not None
    a_thru = bb_inner.add(_TestThru(2), q=a_in)
    b_thru = bb_inner.add(_TestThru(3), q=b_in)
    c_out = bb_inner.add(_TestTwoIn(2, 3), a=a_thru, b=b_thru)
    inner_cbloq = bb_inner.finalize(c=c_out)

    bb_outer = BloqBuilder()
    a_start = bb_outer.add_register(Register("a", QAny(2), side=Side.LEFT))
    b_start = bb_outer.add_register(Register("b", QAny(3), side=Side.LEFT))
    bb_outer.add_register(Register("c", QAny(5), side=Side.RIGHT))
    assert a_start is not None
    assert b_start is not None
    c_final = bb_outer.add(inner_cbloq, a=a_start, b=b_start)
    outer_cbloq = bb_outer.finalize(c=c_final)

    # This now succeeds as the flatten logic has been improved.
    outer_cbloq.flatten()


@pytest.mark.xfail(
    raises=KeyError, reason="Cirq's QASM output fails on measurement keys from decomposed bloqs."
)
def test_cbloq_to_qasm_output_fails_from_as_cirq_op():
    """Reproduces KeyError from _to_qasm_output on a bloq derived via as_cirq_op.

    This test is currently failing but probably not because of code in qualtran-flasq.
    """
    ancilla_qubit_manager = NaiveGridQubitManager(max_cols=10, negative=True)
    hwp_bloq, hwp_circuit, hwp_data_qubits = build_hwp_circuit(
        n_qubits_data=5, angle=0.123, ancilla_qubit_manager=ancilla_qubit_manager
    )

    # This conversion is part of the flow that leads to the error.
    in_quregs = {"x": np.asarray(hwp_data_qubits)}
    _, circuit = convert_circuit_for_flasq_analysis(
        hwp_circuit,
        signature=hwp_bloq.signature,
        qubit_manager=ancilla_qubit_manager,
        in_quregs=in_quregs,
        out_quregs=in_quregs,
    )

    str(circuit._to_qasm_output())


def test_decomposed_circuit_to_qasm_fails_from_as_cirq_op():
    """Verifies that a circuit decomposed from as_cirq_op can be converted to QASM.

    This historically failed with a ValueError because Cirq's QASM output did not
    support classically controlled operations from measurements.
    """
    ancilla_qubit_manager = NaiveGridQubitManager(max_cols=10, negative=True)
    _, hwp_circuit, _ = build_hwp_circuit(
        n_qubits_data=5, angle=0.123, ancilla_qubit_manager=ancilla_qubit_manager
    )

    context = cirq.DecompositionContext(qubit_manager=ancilla_qubit_manager)
    decomposed_circuit = cirq.Circuit(cirq.decompose(hwp_circuit, context=context))

    decomposed_circuit.to_qasm()


@pytest.mark.xfail(
    raises=KeyError, reason="Cirq's QASM output fails on measurement keys from decomposed bloqs."
)
def test_and_then_uncompute_to_qasm():
    """Checks that measurement-based uncomputation of an AND gate fails QASM conversion.

    This test verifies that the `And(uncompute=True)` bloq, which uses a measurement-based
    decomposition in Cirq, cannot be converted to QASM. This is because Cirq's QASM
    serializer does not support circuits containing measurement gates.
    """
    bb = BloqBuilder()
    # The signature for And is ctrl: QAny(2), target: QBit().
    q0 = bb.add_register("q0", 1)
    q1 = bb.add_register("q1", 1)

    # Apply And gate
    qs, target = bb.add(And(), ctrl=[q0, q1])

    # Apply measurement-based uncomputation
    qs = bb.add(And(uncompute=True), ctrl=qs, target=target)

    circuit = bb.finalize(q0=qs[0], q1=qs[1]).to_cirq_circuit()

    _ = circuit.to_qasm()


def test_and_then_uncompute_to_qasm_after_decomposition():
    """Checks that measurement-based uncomputation of an AND gate can be converted to QASM.

    This test verifies that the `And(uncompute=True)` bloq can be converted to QASM
    after decomposition. This historically failed because Cirq's QASM serializer did
    not support circuits containing measurement gates in this context.
    """
    bb = BloqBuilder()
    # The signature for And is ctrl: QAny(2), target: QBit().
    q0 = bb.add_register("q0", 1)
    q1 = bb.add_register("q1", 1)

    # Apply And gate
    qs, target = bb.add(And(), ctrl=[q0, q1])

    # Apply measurement-based uncomputation
    qs = bb.add(And(uncompute=True), ctrl=qs, target=target)

    circuit = bb.finalize(q0=qs[0], q1=qs[1]).to_cirq_circuit()

    circuit = cirq.Circuit(cirq.decompose(circuit))

    _ = circuit.to_qasm()
