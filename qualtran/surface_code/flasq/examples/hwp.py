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

"""Hamming Weight Phasing (HWP) circuit builder for FLASQ analysis examples."""

from typing import List, Optional, Tuple

import cirq
import numpy as np

from qualtran.bloqs.rotations import HammingWeightPhasing
from qualtran.surface_code.flasq.naive_grid_qubit_manager import NaiveGridQubitManager


def build_hwp_circuit(
    n_qubits_data: int,
    angle: float,
    *,
    data_qubit_manager: Optional[cirq.QubitManager] = None,
    ancilla_qubit_manager: Optional[cirq.QubitManager] = None,
) -> Tuple[HammingWeightPhasing, cirq.Circuit, List[cirq.GridQubit]]:
    """Builds a circuit for Hamming Weight Phasing.

    Args:
        n_qubits_data: The number of data qubits.
        angle: The rotation angle.
        data_qubit_manager: A qubit manager for allocating data qubits.
            Defaults to a fresh NaiveGridQubitManager(max_cols=10, negative=False).
        ancilla_qubit_manager: A qubit manager for allocating ancilla qubits.
            Defaults to a fresh NaiveGridQubitManager(max_cols=10, negative=True).

    Returns:
        A tuple containing:
            - The HammingWeightPhasing bloq instance.
            - The constructed cirq.Circuit.
            - A list of the data qubits.
    """
    if data_qubit_manager is None:
        data_qubit_manager = NaiveGridQubitManager(max_cols=10, negative=False)
    if ancilla_qubit_manager is None:
        ancilla_qubit_manager = NaiveGridQubitManager(max_cols=10, negative=True)

    # Allocate data qubits.
    data_qubits = data_qubit_manager.qalloc(n_qubits_data)

    # Instantiate the HammingWeightPhasing bloq.
    hamming_bloq = HammingWeightPhasing(bitsize=n_qubits_data, exponent=angle / np.pi)

    # Get the bloq as a Cirq operation. The qubit manager is stored with the
    # operation and will be used to allocate ancillas during decomposition.
    op, _ = hamming_bloq.as_cirq_op(qubit_manager=ancilla_qubit_manager, x=np.array(data_qubits))

    # Wrap the single, large operation into a circuit.
    circuit = cirq.Circuit(op)

    return hamming_bloq, circuit, data_qubits


def build_parallel_rz_circuit(
    n_qubits_data: int, angle: float, *, data_qubit_manager: Optional[cirq.QubitManager] = None
) -> Tuple[cirq.Circuit, List[cirq.GridQubit]]:
    """Builds a circuit for applying Rz gates to many qubits in parallel.

    Args:
        n_qubits_data: The number of data qubits.
        angle: The rotation angle in radians.
        data_qubit_manager: A qubit manager for allocating data qubits.
            Defaults to a fresh NaiveGridQubitManager(max_cols=10, negative=False).

    Returns:
        A tuple containing:
            - The constructed cirq.Circuit.
            - A list of the data qubits.
    """
    if data_qubit_manager is None:
        data_qubit_manager = NaiveGridQubitManager(max_cols=10, negative=False)
    data_qubits = data_qubit_manager.qalloc(n_qubits_data)
    parallel_rz_circuit = cirq.Circuit(cirq.rz(angle).on(q) for q in data_qubits)
    return parallel_rz_circuit, data_qubits
