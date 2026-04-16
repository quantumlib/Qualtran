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

from typing import List, Set  # CHANGED: Added for backward-compatible types

import cirq
import numpy as np

# test_naive_grid_qubit_manager.py
import pytest
from cirq import GridQubit, LineQubit

from qualtran.bloqs.rotations import HammingWeightPhasing
from qualtran.surface_code.flasq.cirq_interop import (
    flasq_decompose_keep,
    flasq_intercepting_decomposer,
)
from qualtran.surface_code.flasq.naive_grid_qubit_manager import NaiveGridQubitManager


# Helper function to check internal state
def _check_internal_state(
    manager: NaiveGridQubitManager,
    expected_allocated: Set[GridQubit],  # CHANGED: set -> Set
    expected_free: List[GridQubit],  # CHANGED: list -> List
    expected_generated: int,
):
    assert manager._allocated_qubits == expected_allocated
    # Order in free list might vary depending on pop/append, so check content
    assert set(manager._free_qubits) == set(expected_free)
    assert len(manager._free_qubits) == len(expected_free)
    assert manager._num_generated == expected_generated


@pytest.mark.parametrize("negative", [True, False])
def test_initialization(negative):
    """Verify initial state of the manager."""
    manager = NaiveGridQubitManager(max_cols=3, negative=negative)
    _check_internal_state(manager, set(), [], 0)
    # Check the flag is set correctly
    assert manager._negative == negative


def test_qalloc_first_qubit():
    """Verify allocation of the very first qubit."""
    manager = NaiveGridQubitManager(max_cols=3)
    q = manager.qalloc(1)
    expected_q = [GridQubit(-1, 0)]
    assert q == expected_q
    _check_internal_state(manager, set(expected_q), [], 1)


@pytest.mark.parametrize("negative", [True, False])
def test_qalloc_first_row(negative):
    """Verify allocation within the first row."""
    manager = NaiveGridQubitManager(max_cols=3, negative=negative)
    q = manager.qalloc(2)
    start_row = -1 if negative else 0
    expected_q = [GridQubit(start_row, 0), GridQubit(start_row, 1)]
    assert q == expected_q
    _check_internal_state(manager, set(expected_q), [], 2)


@pytest.mark.parametrize("negative", [True, False])
def test_qalloc_fill_first_row(negative):
    """Verify allocation filling the first row exactly."""
    manager = NaiveGridQubitManager(max_cols=3, negative=negative)
    q = manager.qalloc(3)
    start_row = -1 if negative else 0
    expected_q = [
        GridQubit(start_row, 0),
        GridQubit(start_row, 1),
        GridQubit(start_row, 2),
    ]
    assert q == expected_q
    _check_internal_state(manager, set(expected_q), [], 3)


@pytest.mark.parametrize("negative", [True, False])
def test_qalloc_wrap_to_second_row(negative):
    """Verify the zig-zag wrap to the second row (right-to-left)."""
    manager = NaiveGridQubitManager(max_cols=3, negative=negative)
    q = manager.qalloc(4)
    row1 = -1 if negative else 0
    row2 = -2 if negative else 1
    expected_q = [
        GridQubit(row1, 0),
        GridQubit(row1, 1),
        GridQubit(row1, 2),
        GridQubit(row2, 2),  # Start of second row, right-to-left
    ]
    assert q == expected_q
    _check_internal_state(manager, set(expected_q), [], 4)


@pytest.mark.parametrize("negative", [True, False])
def test_qalloc_multiple_rows_zig_zag(negative):
    """Verify the zig-zag pattern across multiple rows."""
    manager = NaiveGridQubitManager(max_cols=3, negative=negative)
    q = manager.qalloc(7)
    row1 = -1 if negative else 0
    row2 = -2 if negative else 1
    row3 = -3 if negative else 2
    expected_q = [
        GridQubit(row1, 0),
        GridQubit(row1, 1),
        GridQubit(row1, 2),  # Row 1 (L->R)
        GridQubit(row2, 2),
        GridQubit(row2, 1),
        GridQubit(row2, 0),  # Row 2 (R->L)
        GridQubit(row3, 0),  # Row 3 (L->R)
    ]
    assert q == expected_q
    _check_internal_state(manager, set(expected_q), [], 7)


def test_qalloc_invalid_dimension():
    """Verify qalloc raises ValueError for dim != 2."""
    manager = NaiveGridQubitManager(max_cols=3)
    with pytest.raises(ValueError, match="Only qubits \(dim=2\) are supported"):
        manager.qalloc(1, dim=3)


@pytest.mark.parametrize("negative", [True, False])
def test_qfree_single_qubit(negative):
    """Verify freeing a single qubit moves it to the free list."""
    manager = NaiveGridQubitManager(max_cols=3, negative=negative)
    q = manager.qalloc(3)
    assert len(q) == 3
    q_to_free = q[1]  # GridQubit(-1, 1) or (0, 1)
    manager.qfree([q_to_free])
    expected_allocated = {q[0], q[2]}
    expected_free = [q_to_free]
    _check_internal_state(manager, expected_allocated, expected_free, 3)


@pytest.mark.parametrize("negative", [True, False])
def test_qfree_all_qubits(negative):
    """Verify freeing all allocated qubits."""
    manager = NaiveGridQubitManager(max_cols=3, negative=negative)
    q = manager.qalloc(3)
    manager.qfree(q)
    expected_allocated = set()
    expected_free = q
    _check_internal_state(manager, expected_allocated, expected_free, 3)


def test_qfree_unmanaged_qubit():
    """Verify freeing a qubit not managed by this manager raises ValueError."""
    manager = NaiveGridQubitManager(max_cols=3, negative=True)
    manager.qalloc(1)
    unmanaged_qubit = GridQubit(0, 0)

    # CHANGED: Re-formatted the error string to be backward-compatible.
    escaped_qubit_str = str(unmanaged_qubit).replace("(", r"\(").replace(")", r"\)")
    expected_error_msg = (
        f"Qubit {escaped_qubit_str}"
        + r" is not managed by this manager \(wrong row sign\)."
    )
    with pytest.raises(ValueError, match=expected_error_msg):
        manager.qfree([unmanaged_qubit])

    manager_pos = NaiveGridQubitManager(max_cols=3, negative=False)
    manager_pos.qalloc(1)
    unmanaged_qubit_neg_row = GridQubit(-1, 0)

    # CHANGED: Re-formatted the error string.
    escaped_qubit_str_neg = (
        str(unmanaged_qubit_neg_row).replace("(", r"\(").replace(")", r"\)")
    )
    expected_error_msg_neg = (
        f"Qubit {escaped_qubit_str_neg}"
        + r" is not managed by this manager \(wrong row sign\)."
    )
    with pytest.raises(ValueError, match=expected_error_msg_neg):
        manager_pos.qfree([unmanaged_qubit_neg_row])

    unmanaged_qubit_not_alloc = GridQubit(-5, 0)

    # CHANGED: Re-formatted the error string.
    escaped_qubit_str_not_alloc = (
        str(unmanaged_qubit_not_alloc).replace("(", r"\(").replace(")", r"\)")
    )
    expected_error_msg_not_alloc = (
        f"Qubit {escaped_qubit_str_not_alloc}"
        + r" was not allocated by this manager or is invalid."
    )
    with pytest.raises(ValueError, match=expected_error_msg_not_alloc):
        manager.qfree([unmanaged_qubit_not_alloc])


def test_qfree_wrong_qubit_type():
    """Verify freeing a non-GridQubit raises ValueError."""
    manager = NaiveGridQubitManager(max_cols=3)
    manager.qalloc(1)
    wrong_type_qubit = LineQubit(0)
    with pytest.raises(ValueError, match="Can only manage cirq.GridQubit"):
        manager.qfree([wrong_type_qubit])


@pytest.mark.parametrize("negative", [True, False])
def test_qfree_already_freed_qubit(negative):
    """Verify freeing an already freed qubit raises ValueError."""
    manager = NaiveGridQubitManager(max_cols=3, negative=negative)
    q = manager.qalloc(1)
    q_to_free = q[0]
    manager.qfree([q_to_free])

    # CHANGED: Re-formatted the error string.
    escaped_qubit_str = str(q_to_free).replace("(", r"\(").replace(")", r"\)")
    expected_error_msg = f"Qubit {escaped_qubit_str} is already free."
    with pytest.raises(ValueError, match=expected_error_msg):
        manager.qfree([q_to_free])


@pytest.mark.parametrize("negative", [True, False])
def test_qfree_not_allocated_qubit(negative):
    """Verify freeing a valid grid qubit that hasn't been allocated raises ValueError."""
    manager = NaiveGridQubitManager(max_cols=3, negative=negative)
    q0 = manager.qalloc(1)[0]
    start_row = -1 if negative else 0
    q1 = GridQubit(start_row, 1)

    # CHANGED: Re-formatted the error string.
    escaped_qubit_str = str(q1).replace("(", r"\(").replace(")", r"\)")
    expected_error_msg = (
        f"Qubit {escaped_qubit_str}"
        + r" was not allocated by this manager or is invalid."
    )
    with pytest.raises(ValueError, match=expected_error_msg):
        manager.qfree([q1])

    _check_internal_state(manager, {q0}, [], 1)


@pytest.mark.parametrize("negative", [True, False])
def test_qalloc_reuse_single_qubit(negative):
    """Verify that a freed qubit is reused first."""
    manager = NaiveGridQubitManager(max_cols=3, negative=negative)
    q_initial = manager.qalloc(3)
    q0, q1, q2 = q_initial
    manager.qfree([q1])

    _check_internal_state(manager, {q0, q2}, [q1], 3)

    q_reused = manager.qalloc(1)
    assert q_reused == [q1]

    _check_internal_state(manager, {q0, q1, q2}, [], 3)


@pytest.mark.parametrize("negative", [True, False])
def test_qalloc_reuse_multiple_qubits_and_new(negative):
    """Verify reusing multiple freed qubits and allocating new ones."""
    manager = NaiveGridQubitManager(max_cols=3, negative=negative)
    q_initial = manager.qalloc(3)
    q0, q1, q2 = q_initial
    row2 = -2 if negative else 1
    q3_expected = GridQubit(row2, 2)

    manager.qfree([q0, q2])
    _check_internal_state(manager, {q1}, [q0, q2], 3)

    q_alloc = manager.qalloc(3)
    assert len(q_alloc) == 3
    assert set(q_alloc) == {q0, q2, q3_expected}
    assert q3_expected in q_alloc
    assert q0 in q_alloc
    assert q2 in q_alloc

    _check_internal_state(manager, {q0, q1, q2, q3_expected}, [], 4)


@pytest.mark.parametrize("negative", [True, False])
def test_qalloc_reuse_subset_of_free(negative):
    """Verify reusing only a subset of the available free qubits."""
    manager = NaiveGridQubitManager(max_cols=3, negative=negative)
    q_initial = manager.qalloc(3)
    q0, q1, q2 = q_initial
    manager.qfree(q_initial)

    _check_internal_state(manager, set(), [q0, q1, q2], 3)

    q_alloc = manager.qalloc(2)
    assert len(q_alloc) == 2
    reused_set = set(q_alloc)
    original_set = {q0, q1, q2}
    assert reused_set.issubset(original_set)

    remaining_free_list = list(original_set - reused_set)
    assert len(remaining_free_list) == 1

    _check_internal_state(manager, reused_set, remaining_free_list, 3)


def test_qborrow_raises_not_implemented():
    """Verify qborrow raises NotImplementedError."""
    manager = NaiveGridQubitManager(max_cols=3)
    with pytest.raises(NotImplementedError, match="qborrow is not implemented"):
        manager.qborrow(1)


def test_all_qubits():
    """Test the all_qubits() method."""
    manager = NaiveGridQubitManager(max_cols=3, negative=True)
    assert manager.all_qubits() == []

    # Allocate some qubits
    q_alloc1 = manager.qalloc(2)
    # all_qubits should return a sorted list
    assert manager.all_qubits() == sorted(q_alloc1)

    # Free one qubit
    q_to_free = q_alloc1[0]
    manager.qfree([q_to_free])
    remaining_qubits = [q for q in q_alloc1 if q != q_to_free]
    assert manager.all_qubits() == sorted(remaining_qubits)

    # Allocate more qubits (one will be reused)
    q_alloc2 = manager.qalloc(2)
    current_qubits = set(remaining_qubits) | set(q_alloc2)
    assert manager.all_qubits() == sorted(list(current_qubits))
    assert len(manager.all_qubits()) == 3  # 1 remaining + 2 new (1 reused)

    # Free all
    manager.qfree(list(current_qubits))
    assert manager.all_qubits() == []


@pytest.mark.parametrize("negative", [True, False])
def test_edge_case_max_cols_one(negative):
    """Test allocation with max_cols=1 simplifies correctly."""
    manager = NaiveGridQubitManager(max_cols=1, negative=negative)
    q = manager.qalloc(3)
    row1 = -1 if negative else 0
    row2 = -2 if negative else 1
    row3 = -3 if negative else 2
    expected_q = [
        GridQubit(row1, 0),
        GridQubit(row2, 0),
        GridQubit(row3, 0),
    ]
    assert q == expected_q
    _check_internal_state(manager, set(expected_q), [], 3)


def test_edge_case_qalloc_zero():
    """Test qalloc(0) returns empty list and doesn't change state."""
    manager = NaiveGridQubitManager(max_cols=3)
    q_initial = manager.qalloc(2)
    q = manager.qalloc(0)
    assert q == []
    _check_internal_state(manager, set(q_initial), [], 2)


def test_edge_case_qfree_empty():
    """Test qfree([]) doesn't change state and doesn't error."""
    manager = NaiveGridQubitManager(max_cols=3)
    q_initial = manager.qalloc(2)
    try:
        manager.qfree([])
    except Exception as e:
        pytest.fail(f"qfree([]) raised an exception: {e}")
    _check_internal_state(manager, set(q_initial), [], 2)


def test_hamming_weight_phasing_with_manager():
    """Test allocating qubits for HammingWeightPhasing and building a circuit."""
    initial_manager = NaiveGridQubitManager(max_cols=10, negative=False)
    bitsize = 5
    exponent = 0.02

    target_qubits = initial_manager.qalloc(bitsize)
    assert len(target_qubits) == bitsize
    assert all(isinstance(q, GridQubit) for q in target_qubits)

    row1 = 0
    assert target_qubits[0] == GridQubit(row1, 0)
    assert target_qubits[3] == GridQubit(row1, 3)

    target_quregs_arr = np.array(target_qubits).reshape((bitsize,))
    decomp_manager = NaiveGridQubitManager(max_cols=10)
    bloq = HammingWeightPhasing(bitsize=bitsize, exponent=exponent)
    op, out_quregs = bloq.as_cirq_op(qubit_manager=decomp_manager, x=target_quregs_arr)

    circuit = cirq.Circuit(op)
    decomposed_circuit = cirq.Circuit(
        cirq.decompose(
            circuit,
            intercepting_decomposer=flasq_intercepting_decomposer,
            keep=flasq_decompose_keep,
            on_stuck_raise=None,
            context=cirq.DecompositionContext(qubit_manager=decomp_manager),
        )
    )


    assert len(list(decomposed_circuit.all_operations())) > 1
    assert len(decomposed_circuit.all_qubits()) == 5 + (5 - 1) + 2

    initial_manager.qfree(target_qubits)
    _check_internal_state(
        decomp_manager,
        set(),
        [cirq.GridQubit(-1, i) for i in range(6)],  # CHANGED: set(...) -> list(...)
        bitsize - 1 + 2,
    )
