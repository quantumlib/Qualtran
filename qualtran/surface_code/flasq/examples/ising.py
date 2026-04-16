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

"""Ising model Trotter circuit builder for FLASQ analysis examples."""

from functools import lru_cache
from typing import Generator, List, Tuple

import cirq


def ising_zz_layer(
    qubits: Tuple[cirq.GridQubit, ...],
    rows: int,
    cols: int,
    j_coupling: float,
    time_slice: float,
    periodic_boundary: bool = True,
) -> Generator[cirq.Moment, None, None]:
    """Generates the ZZ interaction layer for one Trotter step slice.

    The Ising Hamiltonian term is H_ZZ = - sum_{<i,j>} J * Z_i * Z_j.
    The evolution for a time `t` is exp(-i * H_ZZ * t) = exp(i * J * t * sum Z_i Z_j).
    For a single pair, the evolution is exp(i * J * t * Z_i * Z_j).
    We use the decomposition: exp(i * alpha * Z_i * Z_j) = CNOT(i,j) Rz(2*alpha)(j) CNOT(i,j).
    Here, alpha = J * time_slice. So the Rz angle is 2 * J * time_slice.

    Args:
        qubits: A tuple of GridQubits arranged in a grid.
        rows: Number of rows in the qubit grid.
        cols: Number of columns in the qubit grid.
        j_coupling: The interaction strength J.
        time_slice: The time duration for this specific layer
                    (e.g., dt/2 for the half ZZ layers in 2nd order Trotter).
        periodic_boundary: If True, interactions wrap around the edges of the
                           grid. Defaults to True.

    Returns:
        A generator of `cirq.Moment`s implementing the ZZ interactions. The
        number of moments depends on the lattice dimensions:
        - even x even: 12 moments
        - odd x even / even x odd: 15 moments
        - odd x odd: 18 moments
    """
    rz_angle = 2 * j_coupling * time_slice

    # Helper to get qubit from row/col, handling periodic boundaries
    def get_qubit(r, c):
        if periodic_boundary:
            r %= rows
            c %= cols
        elif not (0 <= r < rows and 0 <= c < cols):
            return None  # Out of bounds for open boundary conditions
        index = r * cols + c
        return qubits[index]

    def yield_interaction_set(interactions: List[Tuple[cirq.GridQubit, cirq.GridQubit]]):
        """Helper function to generate the 3 moments for a specific set of interactions."""
        if not interactions:
            return

        # We define the decomposition CNOT(q1, q2) Rz(q2) CNOT(q1, q2)
        start_cnot = [cirq.CNOT(q1, q2) for q1, q2 in interactions]
        rz = [cirq.Rz(rads=rz_angle).on(q2) for _, q2 in interactions]
        end_cnot = start_cnot  # CNOT is its own inverse

        yield cirq.Moment(start_cnot)
        yield cirq.Moment(rz)
        yield cirq.Moment(end_cnot)

    # --- Horizontal Interactions ---
    # For odd column counts, the wrap-around interaction conflicts with the first
    # interaction in the row. We must separate them into different moments.
    h_even_bulk, h_even_boundary = [], []
    h_odd_bulk = []
    for r in range(rows):
        # Horizontal Even Columns
        for c in range(0, cols, 2):
            q1, q2 = get_qubit(r, c), get_qubit(r, c + 1)
            if q1 is None or q2 is None or q1 == q2:
                continue
            # For odd cols, the last interaction (c=cols-1) wraps around and conflicts with c=0.
            # This only applies for periodic boundaries.
            if periodic_boundary and c == cols - 1 and cols % 2 != 0:
                h_even_boundary.append((q1, q2))
            else:
                h_even_bulk.append((q1, q2))
        # Horizontal Odd Columns
        for c in range(1, cols, 2):
            q1, q2 = get_qubit(r, c), get_qubit(r, c + 1)
            if q1 is None or q2 is None or q1 == q2:
                continue
            # This set never conflicts with itself on the boundary for odd cols.
            h_odd_bulk.append((q1, q2))

    yield from yield_interaction_set(h_even_bulk)
    yield from yield_interaction_set(h_odd_bulk)
    yield from yield_interaction_set(h_even_boundary)

    # --- Vertical Interactions ---
    # Similarly, for odd row counts, we separate the wrap-around interactions.
    v_even_bulk, v_even_boundary = [], []
    v_odd_bulk = []
    for c in range(cols):
        # Vertical Even Rows
        for r in range(0, rows, 2):
            q1, q2 = get_qubit(r, c), get_qubit(r + 1, c)
            if q1 is None or q2 is None or q1 == q2:
                continue
            if periodic_boundary and r == rows - 1 and rows % 2 != 0:
                v_even_boundary.append((q1, q2))
            else:
                v_even_bulk.append((q1, q2))
        # Vertical Odd Rows
        for r in range(1, rows, 2):
            q1, q2 = get_qubit(r, c), get_qubit(r + 1, c)
            if q1 is None or q2 is None or q1 == q2:
                continue
            v_odd_bulk.append((q1, q2))

    yield from yield_interaction_set(v_even_bulk)
    yield from yield_interaction_set(v_odd_bulk)
    yield from yield_interaction_set(v_even_boundary)


def ising_x_layer(
    qubits: Tuple[cirq.GridQubit, ...], h_field: float, time_slice: float
) -> Generator[cirq.Moment, None, None]:
    """Generates the X field layer for one Trotter step slice.

    The Ising Hamiltonian term is H_X = - sum_i h * X_i.
    The evolution for a time `t` is exp(-i * H_X * t) = exp(i * h * t * sum X_i).
    For a single qubit, the evolution is exp(i * h * t * X_i).
    We use the identity: cirq.Rx(theta) = exp(-i * theta/2 * X).
    To get exp(i * h * time_slice * X_i), we set:
    -theta/2 = h * time_slice  =>  theta = -2 * h * time_slice.

    Args:
        qubits: A tuple of GridQubits.
        h_field: The external field strength h.
        time_slice: The time duration for this specific layer (e.g., dt for the
                    full X layer in 2nd order Trotter).

    Returns:
        A generator of 1 `cirq.Moment` implementing the X field interactions.
    """
    # Calculate the required angle for cirq.Rx
    rx_angle = -2 * h_field * time_slice
    operations = [cirq.Rx(rads=rx_angle).on(q) for q in qubits]
    if operations:
        yield cirq.Moment(operations)


# Constant for 4th order Trotter-Suzuki decomposition
GAMMA_4TH_ORDER = (4 - 4 ** (1 / 3)) ** (-1)


@lru_cache(maxsize=None)
def build_ising_circuit(
    rows: int,
    cols: int,
    j_coupling: float,
    h_field: float,
    dt: float,
    n_steps: int,
    order: int = 2,
    periodic_boundary: bool = True,
) -> cirq.Circuit:
    """Builds a Cirq circuit for 2nd order Trotterized Ising evolution.

    Uses the symmetric second-order Trotter-Suzuki decomposition:
    exp(-iHt) ≈ [exp(-i H_X dt/2) exp(-i H_ZZ dt) exp(-i H_X dt/2)]^n_steps
    where H = H_ZZ + H_X.

    Args:
        rows: Number of rows in the lattice. Must be > 0.
        cols: Number of columns in the lattice. Must be > 0.
        j_coupling: Interaction strength J in H_ZZ = -J * sum ZZ.
        h_field: External field strength h in H_X = -h * sum X.
        dt: Time step size for a single Trotter step.
        n_steps: Number of Trotter steps. Must be >= 0.
        order: The order of the Trotter-Suzuki decomposition.
               Supports 2 and 4. Defaults to 2.
        periodic_boundary: If True, interactions wrap around the edges of the
                           grid. Defaults to True.

    Returns:
        A cirq.Circuit object simulating the Ising evolution.

    Raises:
        ValueError: If rows, cols <= 0, n_steps < 0, or order is not 2 or 4.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("Lattice dimensions must be positive.")
    if n_steps < 0:
        raise ValueError("Number of Trotter steps cannot be negative.")
    if order not in [2, 4]:
        raise ValueError(f"Trotter order must be 2 or 4, but got {order}.")
    if n_steps == 0:
        # Return an empty circuit on the qubits if no steps requested
        qubits_list = cirq.GridQubit.rect(rows, cols)
        return cirq.Circuit(cirq.I(q) for q in qubits_list)  # Or just cirq.Circuit()

    qubits = tuple(cirq.GridQubit.rect(rows, cols))
    circuit = cirq.Circuit()

    if order == 2:
        # Build the circuit step-by-step for 2nd order
        for i in range(n_steps):
            # --- X Layer (Start) ---
            if i == 0:
                # First half X layer
                circuit.append(ising_x_layer(qubits, h_field, dt / 2.0))

            # Full ZZ layer
            circuit.append(ising_zz_layer(qubits, rows, cols, j_coupling, dt, periodic_boundary))

            # --- X Layer (End/Merge) ---
            if i < n_steps - 1:
                # Two merged X layers
                circuit.append(ising_x_layer(qubits, h_field, dt))
            elif i == n_steps - 1:
                # Last half X layer
                circuit.append(ising_x_layer(qubits, h_field, dt / 2.0))

    elif order == 4:
        # U4(dt) = U2(g*dt) U2(g*dt) U2((1-4g)*dt) U2(g*dt) U2(g*dt)
        # where g = GAMMA_4TH_ORDER
        g = GAMMA_4TH_ORDER
        dt_vals = [g * dt, g * dt, (1 - 4 * g) * dt, g * dt, g * dt]

        for i in range(n_steps):
            for j, current_dt in enumerate(dt_vals):
                # First half X layer of the U2 step
                # This merges with the previous half X layer
                if i == 0 and j == 0:
                    x_time = current_dt / 2.0
                else:
                    # Merged X layer from U2(j-1) and U2(j)
                    prev_dt = dt_vals[j - 1] if j > 0 else dt_vals[-1]
                    x_time = (prev_dt / 2.0) + (current_dt / 2.0)

                circuit.append(ising_x_layer(qubits, h_field, x_time))

                # Full ZZ layer of the U2 step
                circuit.append(
                    ising_zz_layer(qubits, rows, cols, j_coupling, current_dt, periodic_boundary)
                )

        # Final half X layer from the last U2 step of the last U4 step
        circuit.append(ising_x_layer(qubits, h_field, dt_vals[-1] / 2.0))

    return circuit
