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

# tests/cirq_interop_test.py
import pytest

from qualtran import Signature
from qualtran._infra.composite_bloq import (
    CompositeBloq,
)
from qualtran.bloqs.basic_gates import (
    CNOT,
    Hadamard,
    Toffoli,
    ZPowGate,
)
from qualtran.cirq_interop import cirq_optree_to_cbloq
from qualtran.resource_counting._costing import get_cost_value
from qualtran.surface_code.flasq.cirq_interop import (
    _get_coords_from_op,
    cirq_op_to_bloq_with_span,
    convert_circuit_for_flasq_analysis,
)
from qualtran.surface_code.flasq.span_counting import (
    BloqWithSpanInfo,
    GateSpan,
    TotalSpanCost,
)
from qualtran.surface_code.flasq.volume_counting import (
    FLASQGateCounts,
    FLASQGateTotals,
)


def test_get_coords_from_op():
    assert _get_coords_from_op(cirq.H(cirq.LineQubit(5))) == [(5, 0)]
    assert _get_coords_from_op(
        cirq.CNOT(cirq.GridQubit(1, 2), cirq.GridQubit(3, 4))
    ) == [(1, 2), (3, 4)]
    with pytest.raises(ValueError):
        _get_coords_from_op(cirq.CNOT(cirq.LineQubit(1), cirq.GridQubit(1, 1)))
    with pytest.raises(TypeError):
        _get_coords_from_op(cirq.H(cirq.NamedQubit("a")))


def test_cirq_op_given_bloq_with_span():
    qubits = cirq.GridQubit.rect(5, 5)

    # Q(0,0) to Q(1,2) -> dist = |1-0|+|2-0| = 3
    op_cnot = cirq.CNOT.on(qubits[0], qubits[1 * 5 + 2])
    bloq_cnot = cirq_op_to_bloq_with_span(op_cnot)
    assert isinstance(bloq_cnot, BloqWithSpanInfo)
    assert bloq_cnot.wrapped_bloq == CNOT()
    # CNOT rule: connect_span = compute_span = distance
    assert bloq_cnot.connect_span == 3
    assert bloq_cnot.compute_span == 3

    # Q(0,0) to Q(4,2) -> dist = |4-0|+|2-0| = 6
    # Q(4,2) to Q(2,4) -> dist = |2-4|+|4-2| = 2+2 = 4
    # Q(2,4) to Q(0,0) -> dist = |0-2|+|0-4| = 2+4 = 6
    # Total = 6 + 4 + 6 = 16 Span = 16 / 2 = 8
    op_ccnot = cirq.CCNOT.on(qubits[0], qubits[4 * 5 + 2], qubits[2 * 5 + 4])
    bloq_ccnot = cirq_op_to_bloq_with_span(op_ccnot)

    assert isinstance(bloq_ccnot, BloqWithSpanInfo)
    assert bloq_ccnot.wrapped_bloq == Toffoli()
    # Default rule: connect_span=dist, compute_span=dist
    assert bloq_ccnot.connect_span == 8
    assert bloq_ccnot.compute_span == 8

    # Single qubit op should not be wrapped
    op_h = cirq.H.on(qubits[0])
    bloq_h = cirq_op_to_bloq_with_span(op_h)
    assert not isinstance(bloq_h, BloqWithSpanInfo)
    assert bloq_h == Hadamard()


def test_span_counting_for_op_tree():
    qubits = cirq.GridQubit.rect(5, 5)
    circuit = cirq.Circuit()

    circuit.append([cirq.H.on(qubit) for qubit in qubits])

    total_expected_connect_span = 0
    total_expected_compute_span = 0
    for i in range(5):
        # Q(i,0) to Q(i,2) -> dist = |i-i| + |2-0| = 2
        circuit.append(cirq.CNOT.on(qubits[i * 5 + 0], qubits[i * 5 + 2]))
        # CNOT rule: connect=dist, compute=dist
        total_expected_connect_span += 2
        total_expected_compute_span += 2

    optree = circuit.all_operations()

    cbloq = cirq_optree_to_cbloq(optree, op_conversion_method=cirq_op_to_bloq_with_span)

    cost_val = get_cost_value(cbloq, TotalSpanCost())

    # Should sum the spans from the BloqWithSpanInfo instances created during conversion
    assert cost_val == GateSpan(
        connect_span=total_expected_connect_span,
        compute_span=total_expected_compute_span,
        uncounted_bloqs={},
    )


# ---- New tests for convert_circuit_for_flasq_analysis ----


def test_convert_circuit_basic():
    """Tests basic conversion of H and CNOT."""
    q0, q1 = cirq.LineQubit.range(2)
    original_circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    cbloq, decomposed_circuit = convert_circuit_for_flasq_analysis(original_circuit)

    assert isinstance(cbloq, CompositeBloq)
    assert len(cbloq.bloq_instances) == 2

    callees = {inst.bloq for inst in cbloq.bloq_instances}
    assert Hadamard() in callees
    # CNOT should be wrapped due to span calculation
    assert (
        BloqWithSpanInfo(wrapped_bloq=CNOT(), connect_span=1, compute_span=1) in callees
    )
    flasq_cost = get_cost_value(cbloq, FLASQGateTotals())
    span_cost = get_cost_value(cbloq, TotalSpanCost())
    assert flasq_cost == FLASQGateCounts(hadamard=1, cnot=1)
    assert span_cost == GateSpan(
        connect_span=1, compute_span=1
    )  # Span from CNOT(q0, q1)
    # Check that the decomposed circuit is not empty and has the expected number of ops
    assert len(list(decomposed_circuit.all_operations())) == 2


def test_convert_circuit_zzpow_interception():
    """Tests that ZZPowGate is intercepted and decomposed."""
    q0, q1 = cirq.GridQubit.rect(1, 2)
    exponent = 0.2391
    original_circuit = cirq.Circuit(
        cirq.ZZPowGate(exponent=exponent, global_shift=-0.5).on(q0, q1)
    )
    cbloq, decomposed_circuit = convert_circuit_for_flasq_analysis(original_circuit)

    # Check cbloq properties
    assert isinstance(cbloq, CompositeBloq)
    # ZZPow should decompose into CNOT, ZPowGate, CNOT via the interceptor
    assert len(cbloq.bloq_instances) == 3

    callees = [inst.bloq for inst in cbloq.bloq_instances]

    expected_z_pow_bloq = ZPowGate(exponent=exponent)
    expected_cnot_bloq = BloqWithSpanInfo(
        wrapped_bloq=CNOT(), connect_span=1, compute_span=1
    )

    cnot_count = 0
    z_pow_count = 0
    found_z_pow_exponent = None

    for bloq in callees:
        if bloq == expected_cnot_bloq:
            cnot_count += 1
        elif isinstance(bloq, ZPowGate):
            z_pow_count += 1
            found_z_pow_exponent = bloq.exponent
        else:
            pytest.fail(f"Unexpected bloq type found in decomposition: {bloq}")

    assert cnot_count == 2, f"Expected 2 CNOTs, found {cnot_count}"
    assert z_pow_count == 1, f"Expected 1 ZPowGate, found {z_pow_count}"
    assert found_z_pow_exponent is not None, "ZPowGate gate not found"
    assert np.isclose(
        found_z_pow_exponent, expected_z_pow_bloq.exponent
    ), f"ZPowGate exponent mismatch: expected {expected_z_pow_bloq.exponent}, found {found_z_pow_exponent}"

    flasq_cost = get_cost_value(cbloq, FLASQGateTotals())
    span_cost = get_cost_value(cbloq, TotalSpanCost())
    # The FLASQ coster counts the decomposed ops: 2 CNOTs, 1 ZPowGate (which counts as 1 z_rotation)
    assert flasq_cost == FLASQGateCounts(cnot=2, z_rotation=1)
    assert span_cost == GateSpan(connect_span=2, compute_span=2)  # Span from two CNOTs

    # Check decomposed_circuit properties
    assert len(list(decomposed_circuit.all_operations())) == 3  # CNOT, ZPow, CNOT
    # Verify the operations in the decomposed circuit match the expected decomposition
    # This is implicitly tested by the cbloq structure check above, but an explicit check is good.
    expected_decomposed_ops = [
        cirq.CNOT(q0, q1),
        cirq.ZPowGate(exponent=exponent).on(
            q1
        ),  # Note: ZPowGate is on q1 in this specific decomposition
        cirq.CNOT(q0, q1),
    ]
    # Note: The exact qubit for ZPowGate might depend on the interceptor's implementation details.
    # For now, we check the types and count.
    assert (
        sum(1 for op in decomposed_circuit.all_operations() if op.gate == cirq.CNOT)
        == 2
    )
    assert (
        sum(
            1
            for op in decomposed_circuit.all_operations()
            if isinstance(op.gate, cirq.ZPowGate)
        )
        == 1
    )


def test_convert_circuit_cnot_keep():
    """Tests that CNOT is kept by the decomposer."""
    q0, q1 = cirq.LineQubit.range(2)
    original_circuit = cirq.Circuit(
        cirq.H(q1),
        cirq.CZ(
            q0, q1
        ),  # CZ is kept by default by cirq.decompose if no specific keep is given
        cirq.H(q1),
        cirq.CNOT(q0, q1),  # CNOT is explicitly kept by flasq_decompose_keep
    )
    # Decompose without our special keep to see what cirq.decompose would do
    # This is just for understanding, not part of the main test logic for convert_circuit
    decomposed_circuit_cirq_default = cirq.Circuit(cirq.decompose(original_circuit))

    cbloq, decomposed_circuit_flasq = convert_circuit_for_flasq_analysis(
        original_circuit
    )
    assert isinstance(cbloq, CompositeBloq)

    # Check that CNOT BloqWithSpanInfo is present among the callees
    found_cnot_wrapped = False
    expected_cnot_bloq = BloqWithSpanInfo(
        wrapped_bloq=CNOT(), connect_span=1, compute_span=1
    )
    for inst in cbloq.bloq_instances:
        if inst.bloq == expected_cnot_bloq:
            found_cnot_wrapped = True
            break
    assert found_cnot_wrapped, "CNOT should have been kept and wrapped"


    flasq_cost = get_cost_value(cbloq, FLASQGateTotals())
    span_cost = get_cost_value(cbloq, TotalSpanCost())
    # CZ is kept, CNOT is kept. CZ counts as 1 cz, CNOT counts as 1 cnot.
    assert flasq_cost == FLASQGateCounts(cnot=1, hadamard=2, cz=1)
    # Span comes from CZ (dist=1) and CNOT (dist=1). Both use the (D, D) rule now.
    # CZ: connect=1, compute=1. CNOT: connect=1, compute=1. Total: connect=2, compute=2
    assert span_cost == GateSpan(connect_span=2, compute_span=2)

    # Check the decomposed_circuit_flasq
    # It should contain H, CZ, H, CNOT (or their BloqWithSpanInfo equivalents for multi-qubit)
    # The `flasq_decompose_keep` ensures CNOT is not broken down further.
    # `cirq.decompose` with `flasq_intercepting_decomposer` and `flasq_decompose_keep`
    # should result in a circuit where CNOT and CZ are preserved.
    num_cnot_in_decomposed = sum(
        1 for op in decomposed_circuit_flasq.all_operations() if op.gate == cirq.CNOT
    )
    num_cz_in_decomposed = sum(
        1 for op in decomposed_circuit_flasq.all_operations() if op.gate == cirq.CZ
    )
    num_h_in_decomposed = sum(
        1 for op in decomposed_circuit_flasq.all_operations() if op.gate == cirq.H
    )

    assert num_cnot_in_decomposed == 1, "Expected 1 CNOT in FLASQ decomposed circuit"
    assert num_cz_in_decomposed == 1, "Expected 1 CZ in FLASQ decomposed circuit"
    assert num_h_in_decomposed == 2, "Expected 2 Hs in FLASQ decomposed circuit"
    assert len(list(decomposed_circuit_flasq.all_operations())) == 4


def test_convert_circuit_with_signature():
    """Tests conversion providing signature and quregs."""
    q0, q1 = cirq.LineQubit.range(2)
    original_circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))

    sig = Signature.build(q_reg=2)
    q_reg_cirq = np.array([q0, q1])
    in_quregs = {"q_reg": q_reg_cirq}
    out_quregs = {"q_reg": q_reg_cirq}
    cbloq, decomposed_circuit = convert_circuit_for_flasq_analysis(
        original_circuit, signature=sig, in_quregs=in_quregs, out_quregs=out_quregs
    )

    assert isinstance(cbloq, CompositeBloq)
    assert cbloq.signature == sig
    flasq_cost = get_cost_value(cbloq, FLASQGateTotals())
    span_cost = get_cost_value(cbloq, TotalSpanCost())
    assert flasq_cost == FLASQGateCounts(hadamard=1, cnot=1)
    assert span_cost == GateSpan(connect_span=1, compute_span=1)
    assert len(list(decomposed_circuit.all_operations())) == 2


def test_no_unknown_bloqs_for_fsim_circuit():
    """
    Tests that a circuit with PhasedXZGate and FSimGate, when converted for
    FLASQ analysis, results in no unknown bloqs for FLASQGateTotals.
    """
    example_circuit = cirq.Circuit(
        [
            cirq.Moment(
                cirq.FSimGate(theta=1.5707963267948966, phi=2.5622930213643267).on(
                    cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
                ),
                cirq.FSimGate(theta=1.5617711594196637, phi=-3.141592653589793).on(
                    cirq.GridQubit(0, 2), cirq.GridQubit(1, 0)
                ),
                cirq.FSimGate(theta=1.5615336550150525, phi=-3.141592653589793).on(
                    cirq.GridQubit(1, 1), cirq.GridQubit(1, 2)
                ),
                cirq.FSimGate(theta=0.9396453361404614, phi=-3.141592653589793).on(
                    cirq.GridQubit(2, 0), cirq.GridQubit(2, 1)
                ),
            ),
            cirq.Moment(
                cirq.FSimGate(theta=1.5707963267948966, phi=-3.133962353352754).on(
                    cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)
                ),
                cirq.FSimGate(theta=1.58117866219648, phi=-3.141592653589793).on(
                    cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)
                ),
                cirq.FSimGate(theta=1.7082774478642317, phi=-3.141592653589793).on(
                    cirq.GridQubit(1, 2), cirq.GridQubit(2, 0)
                ),
                cirq.FSimGate(theta=0.9396453361404614, phi=-3.141592653589793).on(
                    cirq.GridQubit(2, 1), cirq.GridQubit(2, 2)
                ),
            ),
        ]
    )

    cbloq, circuit = convert_circuit_for_flasq_analysis(example_circuit)
    flasq_counts = get_cost_value(cbloq, FLASQGateTotals())


    assert not flasq_counts.bloqs_with_unknown_cost


# =============================================================================
# Phase 1: Characterization tests for untested cirq_interop branches
# =============================================================================

from qualtran.bloqs.mcmt import And
from qualtran.surface_code.flasq.cirq_interop import (
    cirq_op_to_bloq_tolerate_classical_controls,
    flasq_decompose_keep,
)


class TolerateClassicalControlsTestSuite:
    """Characterization tests for cirq_op_to_bloq_tolerate_classical_controls (L51-57)."""

    def test_strips_classical_control(self):
        """A classically controlled H should yield a plain Hadamard bloq."""
        q = cirq.LineQubit(0)
        op = cirq.H(q).with_classical_controls("m")
        bloq = cirq_op_to_bloq_tolerate_classical_controls(op)
        assert bloq == Hadamard()

    def test_non_controlled_passthrough(self):
        """A non-controlled op should pass through unchanged."""
        q = cirq.LineQubit(0)
        op = cirq.H(q)
        bloq = cirq_op_to_bloq_tolerate_classical_controls(op)
        assert bloq == Hadamard()


class CirqOpToBloqWithSpanEdgeCasesTestSuite:
    """Characterization tests for edge cases in cirq_op_to_bloq_with_span."""

    def test_no_gate_raises_without_tolerate(self):
        """An op with gate=None should raise ValueError when not tolerating controls (L84-85)."""
        q = cirq.LineQubit(0)
        op = cirq.H(q).with_classical_controls("m")
        # ClassicallyControlledOperation has gate=None
        assert op.gate is None
        with pytest.raises(ValueError, match="has no gate"):
            cirq_op_to_bloq_with_span(op, tolerate_classical_controls=False)

    def test_tolerate_flag_delegates(self):
        """The tolerate_classical_controls flag should strip classical controls (L82)."""
        q = cirq.LineQubit(0)
        op = cirq.H(q).with_classical_controls("m")
        bloq = cirq_op_to_bloq_with_span(op, tolerate_classical_controls=True)
        # Single-qubit, so not wrapped in BloqWithSpanInfo
        assert bloq == Hadamard()

    def test_span_failure_warns_and_returns_base_bloq(self):
        """When span calculation fails (e.g. NamedQubits), should warn and return base bloq (L101-105)."""
        q0 = cirq.NamedQubit("a")
        q1 = cirq.NamedQubit("b")
        op = cirq.CNOT(q0, q1)
        with pytest.warns(UserWarning, match="Could not calculate span"):
            bloq = cirq_op_to_bloq_with_span(op)
        # Should return the base CNOT bloq, not wrapped
        assert bloq == CNOT()
        assert not isinstance(bloq, BloqWithSpanInfo)


class GetCoordsEdgeCasesTestSuite:
    """Characterization tests for _get_coords_from_op edge cases."""

    def test_empty_qubits_returns_empty(self):
        """An operation with no qubits should return empty list (L25-26)."""
        # Create a gate that acts on 0 qubits
        op = cirq.GlobalPhaseGate(1j).on()
        result = _get_coords_from_op(op)
        assert result == []


class FlasqDecomposeKeepTestSuite:
    """Characterization tests for flasq_decompose_keep."""

    def test_and_gate_kept(self):
        """And gate should be preserved by the keep function (L135-136)."""
        q0, q1, q2 = cirq.LineQubit.range(3)
        op = And().on(q0, q1, q2)
        assert flasq_decompose_keep(op) is True

    def test_cnot_kept(self):
        q0, q1 = cirq.LineQubit.range(2)
        assert flasq_decompose_keep(cirq.CNOT(q0, q1)) is True

    def test_arbitrary_gate_not_kept(self):
        q0, q1 = cirq.LineQubit.range(2)
        op = cirq.ZZPowGate(exponent=0.1).on(q0, q1)
        assert flasq_decompose_keep(op) is False
