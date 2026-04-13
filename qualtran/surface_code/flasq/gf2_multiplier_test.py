# TODO: Remove comments about coding process.
# This file is newly created based on our plan.

import cirq
import pytest

from qualtran.resource_counting import get_cost_value
from qualtran.surface_code.flasq.cirq_interop import convert_circuit_for_flasq_analysis
from qualtran.surface_code.flasq.examples.gf2_multiplier import (
    build_karatsuba_mult_circuit,
    build_quadratic_mult_circuit,
)
from qualtran.surface_code.flasq.span_counting import TotalSpanCost


def test_build_quadratic_mult_circuit_layout():
    """Tests the qubit layout for the quadratic multiplier.

    Verifies that for a small bitsize, the data and ancilla qubits are
    placed on the correct rows and columns as `cirq.GridQubit`s.
    """
    bitsize = 4
    res = build_quadratic_mult_circuit(bitsize=bitsize)
    qubits = res.circuit.all_qubits()

    expected_qubits = set()
    # Data qubits
    expected_qubits.update({cirq.GridQubit(0, i) for i in range(bitsize)})
    expected_qubits.update({cirq.GridQubit(1, i) for i in range(bitsize)})
    # Ancilla qubits
    num_anc = bitsize
    anc_cols = bitsize + num_anc // 2  # 4 + 3 = 7
    expected_qubits.update({cirq.GridQubit(0, i) for i in range(bitsize, anc_cols)})
    expected_qubits.update({cirq.GridQubit(1, i) for i in range(bitsize, anc_cols)})

    assert qubits == expected_qubits


def test_build_karatsuba_mult_circuit_layout():
    """Tests the qubit layout for the Karatsuba multiplier.

    Verifies that for a small bitsize, the data and ancilla qubits are
    placed on the correct rows and columns as `cirq.GridQubit`s.
    """
    bitsize = 4
    res = build_karatsuba_mult_circuit(bitsize=bitsize)
    qubits = res.circuit.all_qubits()

    expected_qubits = set()
    # Data qubits
    expected_qubits.update({cirq.GridQubit(0, i) for i in range(bitsize)})
    expected_qubits.update({cirq.GridQubit(1, i) for i in range(bitsize)})
    # Ancilla qubits
    num_anc = bitsize
    expected_qubits.update(
        {cirq.GridQubit(0, bitsize + i) for i in range(num_anc // 2)}
    )
    expected_qubits.update(
        {cirq.GridQubit(1, bitsize + i) for i in range(num_anc // 2)}
    )

    assert qubits == expected_qubits


def test_quadratic_mult_circuit_span_cost():
    """Golden test for the connect_span of the quadratic multiplier."""
    res = build_quadratic_mult_circuit(bitsize=10)
    cbloq, _ = convert_circuit_for_flasq_analysis(res.circuit)
    span_cost = get_cost_value(cbloq, TotalSpanCost())
    # Golden value derived from original notebook.
    assert span_cost.connect_span == 1125


def test_karatsuba_mult_circuit_span_cost():
    """Golden test for the connect_span of the Karatsuba multiplier."""
    res = build_karatsuba_mult_circuit(bitsize=10)
    cbloq, _ = convert_circuit_for_flasq_analysis(res.circuit)
    span_cost = get_cost_value(cbloq, TotalSpanCost())
    # Golden value derived from original notebook.
    assert span_cost.connect_span == 1204
