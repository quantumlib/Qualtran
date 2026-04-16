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

"""Grid qubit allocation strategy for constructing circuits on a 2D layout."""

from typing import Iterable, List, Set, Tuple

import cirq
from cirq import QubitManager


class NaiveGridQubitManager(QubitManager):
    """A QubitManager that allocates GridQubits with negative row indices.

    This manager allocates qubits sequentially using a zig-zag pattern
    starting from GridQubit(-1, 0) (if `negative=True`) or GridQubit(0, 0)
    (if `negative=False`) up to a maximum column index.
    It assumes the underlying system provides these qubits in the |0> state.
    It supports reusing freed qubits.


    Allocation Order (example max_cols=3):
        (-1, 0), (-1, 1), (-1, 2),
        (-2, 2), (-2, 1), (-2, 0),
        (-3, 0), (-3, 1), (-3, 2),
        ...

    `qborrow` is not implemented.
    Only supports allocating qubits (dim=2).
    """

    def __init__(self, max_cols: int, negative: bool = True):
        """Initializes the NaiveGridQubitManager.

        Args:
            max_cols: The maximum number of columns (exclusive). Qubits will be
                allocated in columns 0 to max_cols - 1. Must be >= 1.
            negative: If True (default), allocate qubits with negative row
                indices starting from -1. If False, allocate qubits with
                non-negative row indices starting from 0.
        """
        if not isinstance(max_cols, int) or max_cols < 1:
            raise ValueError("max_cols must be a positive integer.")
        self._max_cols = max_cols
        self._negative = negative
        self._allocated_qubits: Set[cirq.GridQubit] = set()
        self._free_qubits: List[cirq.GridQubit] = []
        self._num_generated: int = 0

    def _get_coords(self, index: int) -> Tuple[int, int]:
        """Calculates the grid coordinates for the nth generated qubit.

        Args:
            index: The 0-based index of the qubit in the generation sequence.

        Returns:
            A tuple (row, col).
        """
        if self._max_cols == 0:
            # Should not happen due to constructor check, but safeguard division by zero
            raise ValueError("max_cols cannot be zero.")

        full_rows = index // self._max_cols
        col_offset = index % self._max_cols

        if self._negative:
            row = -1 - full_rows
        else:
            row = full_rows

        # Determine column based on row index (even/odd for zig-zag)
        if full_rows % 2 == 0:
            # Even full_rows (0, 2, 4...) correspond to rows -1, -3, -5... (Left-to-Right)
            col = col_offset
        else:
            # Odd full_rows (1, 3, 5...) correspond to rows -2, -4, -6... (Right-to-Left)
            col = self._max_cols - 1 - col_offset

        return row, col

    def _get_next_new_qubit(self) -> cirq.GridQubit:
        """Generates the next qubit in the sequence."""
        row, col = self._get_coords(self._num_generated)
        qubit = cirq.GridQubit(row, col)
        self._num_generated += 1
        return qubit

    def qalloc(self, n: int, dim: int = 2) -> List["cirq.Qid"]:
        """Allocates `n` clean GridQubits.

        Prefers reusing previously freed qubits before generating new ones
        following the zig-zag pattern.

        Args:
            n: The number of qubits to allocate.
            dim: The dimension of the qubits. Must be 2.

        Returns:
            A list of `n` allocated `cirq.GridQubit`s.

        Raises:
            ValueError: If dim is not 2.
        """
        if dim != 2:
            raise ValueError("Only qubits (dim=2) are supported by NaiveGridQubitManager.")
        if n < 0:
            raise ValueError("Cannot allocate a negative number of qubits.")
        if n == 0:
            return []

        allocated: List[cirq.GridQubit] = []

        # Step 1: Reuse qubits from the free list
        num_reuse = min(n, len(self._free_qubits))
        for _ in range(num_reuse):
            # LIFO reuse (pop from end)
            qubit_to_reuse = self._free_qubits.pop()
            allocated.append(qubit_to_reuse)
            self._allocated_qubits.add(qubit_to_reuse)

        # Step 2: Generate new qubits if needed
        num_new = n - num_reuse
        for _ in range(num_new):
            new_qubit = self._get_next_new_qubit()
            # Sanity check: ensure we don't generate a qubit that's somehow already allocated
            # This shouldn't happen with correct logic but is a useful safeguard.
            if new_qubit in self._allocated_qubits:
                raise RuntimeError(
                    f"Generated qubit {new_qubit} which is already allocated. "
                    "Internal state inconsistency."
                )
            allocated.append(new_qubit)
            self._allocated_qubits.add(new_qubit)

        return allocated  # type: ignore # Ignore because we know they are GridQubits

    def qborrow(self, n: int, dim: int = 2) -> List["cirq.Qid"]:
        """Not implemented for NaiveGridQubitManager."""
        raise NotImplementedError("qborrow is not implemented for NaiveGridQubitManager.")

    def qfree(self, qubits: Iterable["cirq.Qid"]) -> None:
        """Frees the given qubits, making them available for future qalloc calls.

        Args:
            qubits: An iterable of `cirq.GridQubit`s previously allocated by
                this manager.

        Raises:
            ValueError: If any qubit in the iterable is not a `cirq.GridQubit`,
                or was not currently allocated by this manager.
        """
        for q in qubits:
            if not isinstance(q, cirq.GridQubit):
                raise ValueError(f"Can only manage cirq.GridQubit, but got {type(q)} ({q}).")

            if q not in self._allocated_qubits:
                # Check if it was perhaps already freed
                if q in self._free_qubits:
                    raise ValueError(f"Qubit {q} is already free.")
                # Otherwise, it was never allocated or is invalid
                # Check if it's invalid because it's in the wrong row-index space
                if (self._negative and q.row >= 0) or (not self._negative and q.row < 0):
                    raise ValueError(f"Qubit {q} is not managed by this manager (wrong row sign).")
                # Otherwise, it's just not currently allocated
                raise ValueError(f"Qubit {q} was not allocated by this manager or is invalid.")

            # Mark as free
            self._allocated_qubits.remove(q)
            self._free_qubits.append(q)

    def all_qubits(self) -> List[cirq.GridQubit]:
        """Returns a sorted list of all currently allocated qubits."""
        # Returning a sorted list makes the output deterministic for testing.
        return sorted(list(self._allocated_qubits))

    @property
    def num_generated_qubits(self) -> int:
        return self._num_generated

    @property
    def num_allocated(self) -> int:
        return len(self._allocated_qubits)

    @property
    def num_free(self) -> int:
        return len(self._free_qubits)
