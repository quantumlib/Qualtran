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

# --- Consolidated Imports ---
import time
from typing import Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from qualtran.cirq_interop import CirqQuregT

import cirq
import numpy as np

import qualtran
import qualtran.bloqs
import qualtran.bloqs.mcmt
from qualtran import QUInt, Signature
from qualtran.bloqs.arithmetic.addition import Add
from qualtran.cirq_interop import cirq_optree_to_cbloq
from qualtran.resource_counting import get_cost_value
from qualtran.surface_code.flasq.cirq_interop import cirq_op_to_bloq_with_span
from qualtran.surface_code.flasq.span_counting import GateSpan, TotalSpanCost
from qualtran.surface_code.flasq.volume_counting import FLASQGateCounts, FLASQGateTotals


def analyze_adder_costs(bitsize: int):
    """
    Instantiates the Add bloq and calculates its FLASQ and Span costs directly.
    (Analyzes the abstract bloq before Cirq conversion/decomposition).

    Args:
        bitsize: The number of bits for the adder's registers.
    """
    print(f"\n--- Analyzing Abstract Add Bloq ({bitsize} bits) ---")

    try:
        register_type = QUInt(bitsize)
        adder_bloq = Add(a_dtype=register_type, b_dtype=register_type)
        print(f"Instantiated Bloq: {adder_bloq}")
    except Exception as e:
        print(f"An unexpected error occurred during bloq instantiation: {e}")
        return

    try:
        flasq_costs: FLASQGateCounts = get_cost_value(adder_bloq, FLASQGateTotals())
        print(f"\nFLASQ Gate Counts (Abstract):")
        print(flasq_costs)
    except Exception as e:
        print(f"\nError calculating FLASQ costs: {e}")

    try:
        span_costs: GateSpan = get_cost_value(adder_bloq, TotalSpanCost())
        print("\nTotal Span Cost (Abstract):")
        print(span_costs)
    except Exception as e:
        print(f"\nError calculating Span costs: {e}")

    print("-" * (30 + len(str(bitsize))))


def is_flasq_parseable(op: cirq.Operation) -> bool:
    """
    Checks if a Cirq operation is broken down for processing in the FLASQ cost model.

    An operation is succinct if it acts on <= 2 qubits or if it is a Toffoli gate.

    Args:
        op: The cirq.Operation to check.

    Returns:
        True if the operation is succinct, False otherwise.
    """
    if len(op.qubits) <= 2:
        return True

    if isinstance(op.gate, cirq.CCXPowGate):
        return True

    if isinstance(op.gate, qualtran.bloqs.mcmt.And):
        return True

    return False


def create_adder_circuit_and_decorations(
    bitsize: int,
) -> Tuple[cirq.Circuit, Signature, Dict[str, "CirqQuregT"], Dict[str, "CirqQuregT"]]:
    """
    Creates a Cirq circuit for the Add bloq, decomposes it, and remaps ancillas.

    Args:
        bitsize: The number of bits for the adder's registers.

    Returns:
        A decomposed cirq.Circuit with ancillas potentially remapped.
    """
    print(f"\n--- Creating and Decomposing Adder Circuit ({bitsize} bits) ---")

    register_type = QUInt(bitsize)
    adder_bloq = Add(a_dtype=register_type, b_dtype=register_type)
    print(f"Using Bloq: {adder_bloq}")
    # We'll use indices 0, 3, 6,... for 'a' and 1, 4, 7,... for 'b'
    # leaving 2, 5, 8,... potentially for ancillas.
    a_qubits = np.asarray([cirq.LineQubit(i * 3 + 0) for i in range(bitsize)])
    b_qubits = np.asarray([cirq.LineQubit(i * 3 + 1) for i in range(bitsize)])
    print(f"Target a_qubits: {a_qubits}")
    print(f"Target b_qubits: {b_qubits}")

    in_quregs = {"a": a_qubits, "b": b_qubits}
    out_quregs = {"a": a_qubits, "b": b_qubits}

    adder_op, _ = adder_bloq.as_cirq_op(
        qubit_manager=cirq.SimpleQubitManager(), a=b_qubits, b=a_qubits
    )
    assert adder_op is not None
    circuit = cirq.Circuit(adder_op)

    print("Initial circuit created (with potentially large BloqAsCirqGate or similar).")

    circuit = cirq.Circuit(cirq.decompose(circuit, keep=is_flasq_parseable))
    print("Circuit decomposed using 'is_succinct'.")

    # The decomposition might create qubits like `cirq.ops.CleanQubit(i, prefix='_decompose_protocol')`
    qubit_map: dict[cirq.Qid, cirq.Qid] = {}
    for i in range(bitsize):
        j = (bitsize - i - 1) * 3 - 1
        qubit_map[cirq.ops.CleanQubit(i, prefix="_decompose_protocol")] = cirq.LineQubit(j)

    circuit = circuit.transform_qubits(qubit_map=qubit_map)

    return circuit, adder_bloq.signature, in_quregs, out_quregs


# --- Example Usage ---
if __name__ == "__main__":
    num_bits = 5

    try:
        adder_circuit, _sig, _in_q, _out_q = create_adder_circuit_and_decorations(num_bits)
        print(f"\nFinal Decomposed Adder Circuit ({num_bits} bits):")
        print(
            f"(Circuit has {len(adder_circuit)} moments, {len(list(adder_circuit.all_operations()))} ops)"
        )
        print(adder_circuit)

        print("\nConverting decomposed circuit back to Bloq for costing...")
        start_time = time.time()
        cbloq = cirq_optree_to_cbloq(
            adder_circuit.all_operations(), op_conversion_method=cirq_op_to_bloq_with_span
        )
        convert_time = time.time() - start_time
        print(f"Conversion complete in {convert_time:.2f} seconds.")

        if cbloq is None:
            raise RuntimeError("Failed to convert decomposed circuit back to Bloq.")

        print("\nCalculating FLASQ counts and Span from decomposed circuit...")
        start_time = time.time()
        flasq_costs: FLASQGateCounts = get_cost_value(cbloq, FLASQGateTotals())
        span_info: GateSpan = get_cost_value(cbloq, TotalSpanCost())
        cost_calc_time = time.time() - start_time
        print(f"Cost calculation complete in {cost_calc_time:.2f} seconds.")

        print("\n--- Resource Counts (from Decomposed Circuit) ---")
        print(f"FLASQ Counts: {flasq_costs}")
        print(f"Span Info   : {span_info}")
        print("-" * (60 + len(str(num_bits))))

        if span_info.uncounted_bloqs:
            print("Warning: Uncounted bloqs found in Span calculation!")
        if flasq_costs.bloqs_with_unknown_cost:
            print("Warning: Bloqs with unknown cost found in FLASQ calculation!")

    except Exception as e:
        print(f"\nAn error occurred during the process: {e}")
