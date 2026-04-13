# test_adder_example.py
import cirq


# Import functions/classes to be tested or used in tests
from qualtran.surface_code.flasq.examples.adder_example import (
    analyze_adder_costs,
    create_adder_circuit_and_decorations,
)
from qualtran.resource_counting import get_cost_value
from qualtran.surface_code.flasq.cirq_interop import convert_circuit_for_flasq_analysis
from qualtran.surface_code.flasq.span_counting import GateSpan, TotalSpanCost
from qualtran.surface_code.flasq.volume_counting import (
    FLASQGateTotals,
    FLASQGateCounts,
)



TEST_BITSIZE = 4


def test_analyze_adder_costs_runs():
    """Tests that analyze_adder_costs executes without exceptions."""
    analyze_adder_costs(TEST_BITSIZE)


def test_create_adder_circuit_runs_and_returns_circuit():
    """Tests create_adder_circuit runs and returns a Cirq circuit."""
    circuit, _, _, _ = create_adder_circuit_and_decorations(TEST_BITSIZE)
    assert isinstance(circuit, cirq.Circuit)
    assert len(list(circuit.all_operations())) > 0


def test_decomposed_adder_flasq_and_span_costs():
    """
    Tests applying FLASQ and Span costing to the decomposed adder circuit.
    Verifies that costs are calculated and no unknown/uncounted bloqs remain.
    """
    original_circuit, signature, in_quregs, out_quregs = (
        create_adder_circuit_and_decorations(TEST_BITSIZE)
    )
    print(original_circuit)
    cbloq, decomposed_circuit = convert_circuit_for_flasq_analysis(
        original_circuit,
        signature=signature,
        in_quregs=in_quregs,
        out_quregs=out_quregs,
    )
    assert cbloq is not None  # Ensure conversion succeeded
    assert decomposed_circuit is not None  # Ensure decomposed circuit is returned

    flasq_costs = get_cost_value(cbloq, FLASQGateTotals())
    print(f"FLASQ Costs: {flasq_costs}")
    assert isinstance(flasq_costs, FLASQGateCounts)
    assert not flasq_costs.bloqs_with_unknown_cost
    # Check that some expected gates were counted (Add decomposes to Toffoli/CNOT)
    assert flasq_costs.toffoli > 0 or flasq_costs.cnot > 0
    # 4. Calculate Span costs
    span_info = get_cost_value(cbloq, TotalSpanCost())
    print(f"Span info: {span_info}")
    assert isinstance(span_info, GateSpan)
    assert not span_info.uncounted_bloqs
    # Check that some span was counted (multi-qubit gates exist)
    # Resolve symbols in span_info before making boolean checks
    assert span_info.connect_span > 0
    # Check the decomposed circuit from the conversion
    assert len(list(decomposed_circuit.all_operations())) > 0


from qualtran import QUInt
from qualtran.bloqs.arithmetic import Add
from qualtran.bloqs.mcmt import And
from qualtran.cirq_interop import cirq_optree_to_cbloq
from qualtran._infra.gate_with_registers import get_named_qubits
import numpy as np
import cirq


def test_self_contained_adder_issue():
    adder_bloq = Add(a_dtype=QUInt(4), b_dtype=QUInt(4))

    a_qubits = np.asarray([cirq.LineQubit(i * 3 + 0) for i in range(4)])
    b_qubits = np.asarray([cirq.LineQubit(i * 3 + 1) for i in range(4)])

    adder_op, _ = adder_bloq.as_cirq_op(
        qubit_manager=cirq.SimpleQubitManager(), a=b_qubits, b=a_qubits
    )
    circuit = cirq.Circuit(adder_op)

    def is_and_or_short(op):

        if len(op.qubits) <= 2:
            return True

        if isinstance(op.gate, And):
            return True

        return False

    circuit = cirq.Circuit(cirq.decompose(circuit, keep=is_and_or_short))

    cbloq = cirq_optree_to_cbloq(
        circuit.all_operations(),
        signature=adder_bloq.signature,
        in_quregs={"a": a_qubits, "b": b_qubits},
        out_quregs={"a": a_qubits, "b": b_qubits},
    )
