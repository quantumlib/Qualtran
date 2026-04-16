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

"""GF(2) multiplication circuit builder for FLASQ analysis examples."""

from dataclasses import dataclass
from typing import Dict, Tuple

import cirq

from qualtran import QGF, Signature
from qualtran.bloqs.gf_arithmetic import GF2Multiplication, GF2MulViaKaratsuba
from qualtran.cirq_interop import CirqQuregT


@dataclass
class CircuitAndData:
    """A container for a generated circuit and its associated metadata."""

    circuit: cirq.Circuit
    signature: Signature
    in_quregs: Dict[str, CirqQuregT]
    out_quregs: Dict[str, CirqQuregT]


def build_quadratic_mult_circuit(bitsize: int) -> CircuitAndData:
    """Builds, decomposes, and lays out a circuit for quadratic GF(2) multiplication.

    This function replicates the specific qubit layout from the original analysis
    notebook by manually creating GridQubits and remapping ancillas after
    decomposition. The 'x' register is on row 0, 'y' on row 1, and ancillas
    are interleaved between rows 0 and 1 in the columns after the data qubits.

    Args:
        bitsize: The number of bits for the x and y registers.

    Returns:
        A CircuitAndData object containing the final circuit and its metadata.
    """
    bloq = GF2Multiplication(qgf=QGF(2, bitsize))

    # Define the data qubit layout.
    x = [cirq.GridQubit(0, i) for i in range(bitsize)]
    y = [cirq.GridQubit(1, i) for i in range(bitsize)]
    in_quregs = {"x": x, "y": y}
    out_quregs = {"x": x, "y": y}

    # Decompose the bloq into a circuit with temporary `CleanQubit` ancillas.
    # We flatten the bloq to decompose it all the way to And and CNOT gates.
    frozen_circuit = bloq.decompose_bloq().flatten().to_cirq_circuit(cirq_quregs=in_quregs)
    circuit = frozen_circuit.unfreeze()

    # Define the target layout for the ancilla qubits.
    # The decomposition of GF2Multiplication creates bitsize**2 Toffolis, each of
    # which creates one CleanQubit when decomposed by cirq.
    num_anc = bitsize
    anc_per_row = num_anc // 2
    t = [cirq.GridQubit(0, bitsize + i) for i in range(anc_per_row)] + [
        cirq.GridQubit(1, bitsize + i) for i in range(num_anc - anc_per_row)
    ]

    # Map the temporary ancillas to the target GridQubits.
    anc_qubits = sorted(q for q in circuit.all_qubits() if isinstance(q, cirq.ops.CleanQubit))
    qubit_map = {anc_qubits[i]: t[i] for i in range(num_anc)}
    circuit = circuit.transform_qubits(qubit_map)

    return CircuitAndData(
        circuit=circuit, signature=bloq.signature, in_quregs=in_quregs, out_quregs=out_quregs
    )


def build_karatsuba_mult_circuit(bitsize: int) -> CircuitAndData:
    """Builds, decomposes, and lays out a circuit for Karatsuba-based GF(2) multiplication.

    This function replicates the specific qubit layout from the original analysis
    notebook by manually creating GridQubits and remapping ancillas after
    decomposition. The 'x' register is on row 0, 'y' on row 1, and ancillas
    are interleaved between rows 0 and 1 in the columns after the data qubits.

    Args:
        bitsize: The number of bits for the x and y registers.

    Returns:
        A CircuitAndData object containing the final circuit and its metadata.
    """
    bloq = GF2MulViaKaratsuba(dtype=QGF(2, bitsize))

    # Define the data qubit layout.
    x = [cirq.GridQubit(0, i) for i in range(bitsize)]
    y = [cirq.GridQubit(1, i) for i in range(bitsize)]
    in_quregs = {"x": x, "y": y}
    out_quregs = {"x": x, "y": y}

    # Decompose the bloq into a circuit with temporary `CleanQubit` ancillas.
    # The implementation of Karatsuba has an explicit ancilla register.
    decomposed_bloq = bloq.decompose_bloq()
    frozen_circuit = decomposed_bloq.flatten().to_cirq_circuit(cirq_quregs=in_quregs)
    circuit = frozen_circuit.unfreeze()

    # Define the target layout for the ancilla qubits.
    # The original notebook used `n` ancillas, matching the bloq's signature.
    num_anc = bitsize
    t = [cirq.GridQubit(0, bitsize + i) for i in range(num_anc // 2)] + [
        cirq.GridQubit(1, bitsize + i) for i in range(num_anc - (num_anc // 2))
    ]

    # Map the temporary ancillas to the target GridQubits.
    # The ancilla register in the decomposed bloq is named 'anc'.
    anc_qubits = sorted(q for q in circuit.all_qubits() if isinstance(q, cirq.ops.CleanQubit))
    assert len(anc_qubits) == num_anc
    qubit_map = {anc_qubits[i]: t[i] for i in range(num_anc)}
    circuit = circuit.transform_qubits(qubit_map)

    return CircuitAndData(
        circuit=circuit, signature=bloq.signature, in_quregs=in_quregs, out_quregs=out_quregs
    )
