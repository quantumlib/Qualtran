from __future__ import annotations

import random
from typing import Iterable

import kickmix as km
import numpy as np
import pytest

from src.kickmix.py.circuit.subs.fuzz_test_util import assert_fuzz_testing_acts_like


def expected_toffolis(
    num_address_bits: int, address_values: Iterable[int], *, controlled: bool
) -> int:
    """Returns the Toffoli count for visiting in-range `address_values` in order.

    Positioning on the first value costs `num_address_bits` Toffolis when controlled,
    or `num_address_bits - 1` when uncontrolled. Each change of value from `a` to `b`
    then adds `(a ^ b).bit_length() - 1` Toffolis.
    """
    address_values = list(address_values)
    if num_address_bits == 0:
        return 0
    total = num_address_bits - 1 + (1 if controlled else 0)
    for a, b in zip(address_values, address_values[1:]):
        if a != b:
            total += (a ^ b).bit_length() - 1
    return total


def visited_address_values(cursor: km.UnaryIterationCursor) -> list[int]:
    """Runs one pass over `cursor` and returns the visited address values."""
    with cursor:
        return [address_value for address_value, _ in cursor]


def test_unary_iteration_diagram():
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(2, name="address")
    target = builder.create_quantum_register(1, name="target")
    with builder.unary_iteration(address) as it:
        for address_value, match_qubit in it:
            if address_value % 3 == 0:
                builder.cx(match_qubit, target[0])
    assert builder.finish_circuit().text_diagram().strip() == """
q0: -address[0]-------X---@-X----------Z**b0-X---@-X----------Z**b0--------------
                          |            |         |            |
q1: -address[1]-X---@-X---|------------|---------|------------|------------Z**b0-
                    |     |            |         |            |
q2: -target[0]------|-----|-X----------|---------|---X--------|------------------
                    |     | |          |         |   |        |
q3:                 | |0>-X-@-X-HMR=b0 |     |0>-X-X-@-HMR=b0 |
                    |     |   |        |         | |          |
q4:             |0>-X-----@---@--------@-----X---@-@----------@-----HMR=b0
    """.strip()


@pytest.mark.parametrize("n", range(1, 8))
def test_full_sweep_toffoli_count(n: int):
    # Sweeping all 2**n values costs 2**n - 1 Toffolis under a qubit control,
    # or 2**n - 2 Toffolis when uncontrolled.
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(n, name="address")
    with builder.unary_iteration(address) as it:
        for _ in it:
            pass
    assert (
        builder.finish_circuit().max_magic()
        == expected_toffolis(n, range(1 << n), controlled=False)
        == (1 << n) - 2
    )

    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(n, name="address")
    control = builder.create_quantum_register(1, name="control")
    with builder.unary_iteration(address, control=control[0]) as it:
        for _ in it:
            pass
    assert (
        builder.finish_circuit().max_magic()
        == expected_toffolis(n, range(1 << n), controlled=True)
        == (1 << n) - 1
    )


def test_partial_sweep_toffoli_count():
    # Sweeping [0, 73) on an 8-bit address with a qubit control costs 78 Toffolis.
    assert expected_toffolis(8, range(73), controlled=True) == 78
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(8, name="address")
    control = builder.create_quantum_register(1, name="control")
    with builder.unary_iteration(address, range(73), control=control[0]) as it:
        for _ in it:
            pass
    assert builder.finish_circuit().max_magic() == 78

    # Omitting the control saves one Toffoli.
    assert expected_toffolis(8, range(73), controlled=False) == 77
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(8, name="address")
    with builder.unary_iteration(address, range(73)) as it:
        for _ in it:
            pass
    assert builder.finish_circuit().max_magic() == 77

    # Stopping at 72 instead of 73 saves the 3 Toffolis needed to step from 71 to 72.
    assert expected_toffolis(8, range(72), controlled=True) == 75
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(8, name="address")
    control = builder.create_quantum_register(1, name="control")
    with builder.unary_iteration(address, range(72), control=control[0]) as it:
        for _ in it:
            pass
    assert builder.finish_circuit().max_magic() == 75


def test_a_false_control_emits_nothing():
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(3, name="address")
    target = builder.create_quantum_register(1, name="target")
    with builder.unary_iteration(address, control=False) as it:
        for _, match_qubit in it:
            builder.cx(match_qubit, target[0])
    with_iteration = builder.finish_circuit()

    builder = km.CircuitBuilder()
    builder.create_quantum_register(3, name="address")
    builder.create_quantum_register(1, name="target")
    assert with_iteration == builder.finish_circuit()


def test_unrepresentable_address_values_are_free():
    # Values outside 0..7 cannot match a 3-bit address, so moving to them emits no Toffolis
    # and keeps `match_qubit` at |0>.
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(3, name="address")
    target = builder.create_quantum_register(1, name="target")
    with builder.unary_iteration(address) as it:
        for address_value in [-5, -1, 8, 9, 999, 10**30]:
            it.move_to(address_value)
            assert it.cur_address_value == address_value
            builder.cx(it.match_qubit, target[0])

    assert builder.finish_circuit().max_magic() == 0


def test_moving_between_unrepresentable_and_representable_address_values():
    # Moving from an out-of-range value to an in-range value costs 2 Toffolis
    # for 3 uncontrolled address bits, and moving back out of range costs 0 Toffolis.
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(3, name="address")
    with builder.unary_iteration(address) as it:
        it.move_to(3)
        it.move_to(9)
        assert it.cur_address_value == 9
        it.move_to(-3)
        assert it.cur_address_value == -3
        it.move_to(10**30)
        assert it.cur_address_value == 10**30
        it.move_to(4)
        assert it.cur_address_value == 4
    assert builder.finish_circuit().max_magic() == 4


def test_an_empty_address_value_range_builds_nothing():
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(3, name="address")
    target = builder.create_quantum_register(1, name="target")
    with builder.unary_iteration(address, range(0)) as it:
        for _, match_qubit in it:
            builder.cx(match_qubit, target[0])
    with builder.unary_iteration(address, range(5, 5)) as it:
        for _, match_qubit in it:
            builder.cx(match_qubit, target[0])
    with_iteration = builder.finish_circuit()

    builder = km.CircuitBuilder()
    builder.create_quantum_register(3, name="address")
    builder.create_quantum_register(1, name="target")
    assert with_iteration == builder.finish_circuit()

    # The cursor starts parked at 8 and can still be moved with `move_to`.
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(3, name="address")
    with builder.unary_iteration(address, range(0)) as it:
        assert it.cur_address_value == 8
        it.move_to(3)
    assert builder.finish_circuit().max_magic() == 2


def test_empty_address_register_tracks_the_control():
    # An empty address register has a single value (0), where `match_qubit` copies `control`.
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(0, name="address")
    control = builder.create_quantum_register(1, name="control")
    out = builder.create_quantum_register(1, name="out")
    visited = []
    with builder.unary_iteration(address, control=control[0]) as it:
        for address_value, match_qubit in it:
            visited.append(address_value)
            builder.cx(match_qubit, out[0])
    assert visited == [0]
    circuit = builder.finish_circuit()
    assert circuit.max_magic() == 0

    assert_fuzz_testing_acts_like(
        circuit,
        """
            out ^= control
        """,
        shots=16,
    )


def test_lazy_cursor_emits_nothing_until_opened():
    builder = km.CircuitBuilder()
    builder.create_quantum_register(3, name="address")
    baseline_qubits = builder.num_allocated_qubits
    baseline_circuit = builder.finish_circuit()

    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(3, name="address")
    cursor = builder.unary_iteration(address)
    assert builder.num_allocated_qubits == baseline_qubits
    del cursor
    assert builder.finish_circuit() == baseline_circuit


def test_del_partially_evaluated_cursor_produces_no_additional_operations():
    # Destroying an unclosed cursor does not append any operations to the circuit.
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(3, name="address")
    cursor = builder.unary_iteration(address)
    cursor.__enter__()
    cursor.move_to(5)
    c1 = builder.finish_circuit()
    del cursor
    c2 = builder.finish_circuit()
    assert c1 == c2


def test_del_builder_with_partially_evaluated_cursor():
    # Destroying the builder before the cursor does not dereference a dangling pointer.
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(3, name="address")
    cursor = builder.unary_iteration(address)
    cursor.__enter__()
    cursor.move_to(5)
    del builder
    del cursor


def test_breaking_out_of_a_with_block_still_closes():
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(3, name="address")
    with builder.unary_iteration(address) as it:
        for _ in it:
            break
    assert builder.num_allocated_qubits == 3


def test_iterating_to_exhaustion_does_not_close():
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(2, name="address")
    with builder.unary_iteration(address) as it:
        assert [address_value for address_value, _ in it] == [0, 1, 2, 3]
        # Exhausting the loop keeps the pass open until the `with` block exits.
        assert builder.num_allocated_qubits == 4
        it.move_to(2)
        assert it.cur_address_value == 2
    assert builder.num_allocated_qubits == 2


def test_reusable_cursor_multi_pass():
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(2, name="address")
    target = builder.create_quantum_register(1, name="target")
    cursor = builder.unary_iteration(address)
    with cursor:
        for address_value, match_qubit in cursor:
            if address_value == 1:
                builder.cx(match_qubit, target[0])
    # Reusing the same cursor for uncomputation:
    with cursor:
        for address_value, match_qubit in cursor:
            if address_value == 1:
                builder.cx(match_qubit, target[0])
    assert builder.num_allocated_qubits == 3


def test_iteration_order_is_the_address_value_range_regardless_of_move_to():
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(2, name="address")
    it = builder.unary_iteration(address, range(1, 4))
    with it:
        it.move_to(3)
        assert it.cur_address_value == 3
    assert visited_address_values(it) == [1, 2, 3]
    builder.finish_circuit()


def test_access_when_not_open_raises():
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(2, name="address")
    it = builder.unary_iteration(address)
    with pytest.raises(Exception, match="not open"):
        _ = it.match_qubit
    with pytest.raises(Exception, match="not open"):
        it.move_to(1)
    with pytest.raises(Exception, match="not open"):
        _ = it.cur_address_value
    with pytest.raises(Exception, match="not open"):
        for _ in it:
            pass


def test_an_unpositioned_cursor_is_parked_past_the_last_address_value():
    # Opening a pass starts the cursor at `1 << len(address)` with `match_qubit` at |0>.
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(3, name="address")
    target = builder.create_quantum_register(1, name="target")
    baseline = builder.finish_circuit()

    with builder.unary_iteration(address) as it:
        assert it.cur_address_value == 8
        unpositioned = it.match_qubit
        builder.cx(unpositioned, target[0])
        it.move_to(9)
        assert it.cur_address_value == 9
        assert it.match_qubit == unpositioned
    assert builder.finish_circuit().max_magic() == 0


def test_a_classical_bit_control_is_lowered_to_a_pushed_condition():
    # Controlling by a classical bit pushes a classical condition for the pass,
    # requiring the same number of Toffolis as an uncontrolled pass.
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(3, name="address")
    target = builder.create_quantum_register(1, name="target")
    flag = builder.create_classical_register(1, name="flag")
    with builder.unary_iteration(address, control=flag[0]) as it:
        for address_value, match_qubit in it:
            if address_value == 5:
                builder.cx(match_qubit, target[0])
    circuit = builder.finish_circuit()
    assert circuit.max_magic() == expected_toffolis(3, range(8), controlled=False)
    assert builder.num_allocated_qubits == 4


def test_nested_open_raises():
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(2, name="address")
    it = builder.unary_iteration(address)
    with it:
        with pytest.raises(Exception, match="already open"):
            with it:
                pass


def test_nested_iteration_raises():
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(2, name="address")
    it = builder.unary_iteration(address)
    with it:
        with pytest.raises(Exception, match="already being iterated"):
            for _ in it:
                for _ in it:
                    pass


def test_an_exception_inside_the_with_block_still_closes():
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(3, name="address")
    with pytest.raises(ValueError, match="boom"):
        with builder.unary_iteration(address) as it:
            it.move_to(5)
            raise ValueError("boom")
    assert builder.num_allocated_qubits == 3
    builder.finish_circuit()


def test_workspace_counts_against_max_qubits():
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(3, name="address")
    baseline = builder.num_allocated_qubits

    builder.set_max_qubits(baseline + 2)
    it = builder.unary_iteration(address)
    with pytest.raises(Exception, match="Not enough free qubits"):
        with it:
            pass
    assert builder.num_allocated_qubits == baseline

    builder.set_max_qubits(baseline + 3)
    it = builder.unary_iteration(address)
    with it:
        assert builder.num_allocated_qubits == baseline + 3
        assert builder.num_free_qubits == 0
    assert builder.num_allocated_qubits == baseline
    builder.finish_circuit()


def test_numpy_integers_are_accepted():
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(3, name="address")
    assert visited_address_values(builder.unary_iteration(address, range(np.int64(3)))) == [0, 1, 2]
    assert visited_address_values(
        builder.unary_iteration(address, range(np.int64(2), np.int64(5)))
    ) == [2, 3, 4]
    with builder.unary_iteration(address) as it:
        it.move_to(np.uint8(6))
        assert it.cur_address_value == 6
        it.move_to(np.int32(2))
        assert it.cur_address_value == 2
        it.move_to(np.int64(-4))
        assert it.cur_address_value == -4
    builder.finish_circuit()


def test_bad_address_values_are_rejected():
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(3, name="address")
    with pytest.raises(Exception, match="must be None or a range"):
        builder.unary_iteration(address, 4)
    with pytest.raises(Exception, match="must be None or a range"):
        builder.unary_iteration(address, -1)
    with pytest.raises(Exception, match="must be None or a range"):
        builder.unary_iteration(address, [0, 1, 2])
    with pytest.raises(Exception, match="len\\(address\\) > 63"):
        builder.unary_iteration(builder.create_quantum_register(64, name="wide"))
    builder.finish_circuit()


def test_address_values_argument():
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(4, name="address")
    assert visited_address_values(builder.unary_iteration(address)) == list(range(16))
    assert visited_address_values(builder.unary_iteration(address, range(11))) == list(range(11))
    assert visited_address_values(builder.unary_iteration(address, range(3, 11))) == list(
        range(3, 11)
    )
    # Out-of-range values (negative or >= 16) are visited with match_qubit at |0>.
    assert visited_address_values(builder.unary_iteration(address, range(-2, 3))) == [
        -2,
        -1,
        0,
        1,
        2,
    ]
    assert visited_address_values(builder.unary_iteration(address, range(14, 19))) == [
        14,
        15,
        16,
        17,
        18,
    ]
    assert visited_address_values(builder.unary_iteration(address, range(20, 25))) == [
        20,
        21,
        22,
        23,
        24,
    ]
    with pytest.raises(Exception, match="step 1"):
        builder.unary_iteration(address, range(0, 16, 2))
    builder.finish_circuit()


def test_repr():
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(2, name="address")
    ctrl = builder.create_quantum_register(1, name="ctrl")
    it = builder.unary_iteration(address)
    assert repr(it) == "<km.UnaryIterationCursor address=km.array([km.q(0), km.q(1)])>"

    it_address_values = builder.unary_iteration(address, range(1, 3))
    assert repr(it_address_values) == (
        "<km.UnaryIterationCursor address=km.array([km.q(0), km.q(1)]), address_values=range(1, 3)>"
    )

    it_all = builder.unary_iteration(address, range(0, 4))
    assert repr(it_all) == "<km.UnaryIterationCursor address=km.array([km.q(0), km.q(1)])>"
    assert repr(builder.unary_iteration(address, range(2))) == (
        "<km.UnaryIterationCursor address=km.array([km.q(0), km.q(1)]), address_values=range(0, 2)>"
    )

    it_ctrl = builder.unary_iteration(address, control=ctrl[0])
    assert (
        repr(it_ctrl)
        == "<km.UnaryIterationCursor address=km.array([km.q(0), km.q(1)]), control=km.q(2)>"
    )


def test_direct_construction():
    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(2, name="address")
    it = km.UnaryIterationCursor(builder, address, address_values=range(4))
    assert visited_address_values(it) == [0, 1, 2, 3]


@pytest.mark.parametrize("n_address", range(1, 5))
def test_fuzz_sequential_sweep(n_address: int):
    table = [random.randrange(2) for _ in range(1 << n_address)]

    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(n_address, name="address")
    target = builder.create_quantum_register(1, name="target")
    with builder.unary_iteration(address) as it:
        for address_value, match_qubit in it:
            if table[address_value]:
                builder.cx(match_qubit, target[0])

    assert_fuzz_testing_acts_like(
        builder.finish_circuit(),
        """
            target ^= table[address]
        """,
        shots=64,
        context={'table': table},
    )


@pytest.mark.parametrize("n_address", range(1, 5))
def test_fuzz_controlled_sweep(n_address: int):
    table = [random.randrange(2) for _ in range(1 << n_address)]

    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(n_address, name="address")
    control = builder.create_quantum_register(1, name="control")
    target = builder.create_quantum_register(1, name="target")
    with builder.unary_iteration(address, control=control[0]) as it:
        for address_value, match_qubit in it:
            if table[address_value]:
                builder.cx(match_qubit, target[0])

    assert_fuzz_testing_acts_like(
        builder.finish_circuit(),
        """
            if control:
                target ^= table[address]
        """,
        shots=64,
        context={'table': table},
    )


@pytest.mark.parametrize("n_address_values", [1, 5, 11, 16])
def test_fuzz_partial_address_value_range(n_address_values: int):
    # Visiting only part of the address space is correct for *every* address, not just
    # the visited ones: unvisited values simply never match.
    n_address = 4
    table = [random.randrange(2) for _ in range(n_address_values)] + [0] * (16 - n_address_values)

    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(n_address, name="address")
    target = builder.create_quantum_register(1, name="target")
    with builder.unary_iteration(address, range(n_address_values)) as it:
        for address_value, match_qubit in it:
            if table[address_value]:
                builder.cx(match_qubit, target[0])

    assert_fuzz_testing_acts_like(
        builder.finish_circuit(),
        """
            target ^= table[address]
        """,
        shots=64,
        context={'table': table},
    )


def test_fuzz_random_access_order():
    # Visiting values in an arbitrary order (forwards and backwards) via `move_to`,
    # paying `(a ^ b).bit_length() - 1` Toffolis per jump.
    n_address = 4
    address_values = [random.randrange(16) for _ in range(24)]
    table = [random.randrange(2) for _ in range(16)]
    visits = [0] * 16

    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(n_address, name="address")
    target = builder.create_quantum_register(1, name="target")
    with builder.unary_iteration(address) as it:
        for address_value in address_values:
            it.move_to(address_value)
            assert it.cur_address_value == address_value
            if table[address_value]:
                builder.cx(it.match_qubit, target[0])
                visits[address_value] ^= 1
    circuit = builder.finish_circuit()
    assert circuit.max_magic() == expected_toffolis(n_address, address_values, controlled=False)

    assert_fuzz_testing_acts_like(
        circuit,
        """
            target ^= visits[address]
        """,
        shots=64,
        context={'visits': visits},
    )


def test_fuzz_nested_iterations():
    # An outer cursor over one register, with an inner cursor over another register
    # controlled by the outer cursor's match_qubit.
    outer_size = 3
    inner_size = 3
    table = [[random.randrange(2) for _ in range(inner_size)] for _ in range(outer_size)]

    builder = km.CircuitBuilder()
    outer = builder.create_quantum_register(2, name="outer")
    inner = builder.create_quantum_register(2, name="inner")
    target = builder.create_quantum_register(1, name="target")
    with builder.unary_iteration(outer, range(outer_size)) as outer_it:
        for i, match_qubit_outer in outer_it:
            with builder.unary_iteration(
                inner, range(inner_size), control=match_qubit_outer
            ) as inner_it:
                for j, match_qubit_inner in inner_it:
                    if table[i][j]:
                        builder.cx(match_qubit_inner, target[0])

    assert_fuzz_testing_acts_like(
        builder.finish_circuit(),
        """
            target ^= table[outer][inner]
        """,
        shots=64,
        context={'table': table},
        input_sampler=lambda: {
            'outer': random.randrange(outer_size),
            'inner': random.randrange(inner_size),
            'target': random.randrange(2),
        },
    )


# Comparison flags can be maintained alongside a cursor by toggling a clean qubit
# with `match_qubit` at each step: `[address < i + 1] ^ [address < i] == [address == i]`.


@pytest.mark.parametrize("pivot", [0, 1, 6, 11, 16])
def test_fuzz_less_than_flag_by_hand(pivot: int):
    n_address = 4

    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(n_address, name="address")
    control = builder.create_quantum_register(1, name="control")
    out = builder.create_quantum_register(1, name="out")
    lt = builder.alloc_qubits(1)
    with builder.unary_iteration(address, control=control[0]) as it:
        it.move_to(0)
        for i in range(pivot):
            builder.cx(it.match_qubit, lt)
            it.move_to(i + 1)
        builder.cx(lt, out[0])
        for i in reversed(range(pivot)):
            it.move_to(i)
            builder.cx(it.match_qubit, lt)
    builder.free(lt)

    assert_fuzz_testing_acts_like(
        builder.finish_circuit(),
        """
            if control:
                out ^= address < pivot
        """,
        shots=64,
        context={'pivot': pivot},
    )


@pytest.mark.parametrize("pivot", [0, 1, 6, 11, 16])
def test_fuzz_greater_equal_flag_by_hand(pivot: int):
    n_address = 4

    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(n_address, name="address")
    out = builder.create_quantum_register(1, name="out")
    ge = builder.alloc_qubits(1)
    with builder.unary_iteration(address) as it:
        it.move_to(0)
        builder.x(ge)
        for i in range(pivot):
            builder.cx(it.match_qubit, ge)
            it.move_to(i + 1)
        builder.cx(ge, out[0])
        for i in reversed(range(pivot)):
            it.move_to(i)
            builder.cx(it.match_qubit, ge)
        builder.x(ge)
    builder.free(ge)

    assert_fuzz_testing_acts_like(
        builder.finish_circuit(),
        """
            out ^= address >= pivot
        """,
        shots=64,
        context={'pivot': pivot},
    )


def test_fuzz_flags_with_independent_controls_share_one_cursor():
    # Two controlled comparisons sharing one cursor over `address`.
    n_address = 4
    pivot = 6

    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(n_address, name="address")
    taps = builder.create_quantum_register(2, name="taps")
    out = builder.create_quantum_register(2, name="out")
    flags = builder.alloc_qubits(2)
    lt, ge = flags[0], flags[1]
    with builder.unary_iteration(address) as it:
        it.move_to(0)
        builder.cx(taps[1], ge)
        for i in range(pivot):
            builder.ccx(it.match_qubit, taps[0], lt)
            builder.ccx(it.match_qubit, taps[1], ge)
            it.move_to(i + 1)
        builder.cx(lt, out[0])
        builder.cx(ge, out[1])
        for i in reversed(range(pivot)):
            it.move_to(i)
            builder.ccx(it.match_qubit, taps[0], lt)
            builder.ccx(it.match_qubit, taps[1], ge)
        builder.cx(taps[1], ge)
    builder.free(flags)

    assert_fuzz_testing_acts_like(
        builder.finish_circuit(),
        """
            out ^= ((taps & 1) and address < pivot) | (((taps >> 1) and address >= pivot) << 1)
        """,
        shots=64,
        context={'pivot': pivot},
    )


def test_fuzz_lockstep_cursors_like_the_dqi_eea_loop():
    # Advancing multiple cursors in lockstep across different registers and controls.
    m = 4

    builder = km.CircuitBuilder()
    nq = builder.create_quantum_register(3, name="nq")
    na = builder.create_quantum_register(3, name="na")
    c0 = builder.create_quantum_register(1, name="c0")
    c3 = builder.create_quantum_register(1, name="c3")
    out = builder.create_quantum_register(3 * m, name="out")

    na_lt = builder.alloc_qubits(1)
    with (
        builder.unary_iteration(nq, range(m), control=c0[0]) as nq_eq,
        builder.unary_iteration(nq, range(m + 1), control=c3[0]) as nq_eq1,
        builder.unary_iteration(na, range(m), control=c3[0]) as na_eq,
    ):
        for i in range(m):
            nq_eq.move_to(i)
            nq_eq1.move_to(i + 1)
            builder.cx(nq_eq.match_qubit, out[3 * i])
            builder.cx(nq_eq1.match_qubit, out[3 * i + 1])
            builder.cx(na_lt, out[3 * i + 2])
            na_eq.move_to(i)
            builder.cx(na_eq.match_qubit, na_lt)
        builder.cx(c3[0], na_lt)
    builder.free(na_lt)

    assert_fuzz_testing_acts_like(
        builder.finish_circuit(),
        """
            for i in range(m):
                out ^= (c0 and nq == i) << (3 * i)
                out ^= (c3 and nq == i + 1) << (3 * i + 1)
                out ^= (c3 and na < i) << (3 * i + 2)
        """,
        shots=64,
        context={'m': m},
        input_sampler=lambda: {
            'nq': random.randrange(m),
            'na': random.randrange(m),
            'c0': random.randrange(2),
            'c3': random.randrange(2),
            'out': random.randrange(1 << (3 * m)),
        },
    )
