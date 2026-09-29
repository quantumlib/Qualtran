from __future__ import annotations

import fractions
import random
from typing import Any

import numpy as np
import pytest

import kickmix as km


def test_sim_storage_int():
    sim = km.Simulator(batch_size=5)
    sim.write_across_shots(km.q(0), 1)
    sim.write_across_shots(km.q(1), 2)
    sim.write_across_shots(km.q(2), 3)
    sim.write_across_shots(km.q(3), 4)
    sim.write_across_shots(km.b(0), 5)
    sim.write_across_shots(km.b(1), 6)
    sim.write_across_shots(km.b(2), 7)
    sim.write_across_shots(km.b(3), 8)
    sim.write_across_shots(km.q(99), 9)
    sim.write_across_shots(km.b(99), 10)
    assert sim.read_across_shots(km.q(0), out=int) == 1
    assert sim.read_across_shots(km.q(1), out=int) == 2
    assert sim.read_across_shots(km.q(2), out=int) == 3
    assert sim.read_across_shots(km.q(3), out=int) == 4
    assert sim.read_across_shots(km.b(0), out=int) == 5
    assert sim.read_across_shots(km.b(1), out=int) == 6
    assert sim.read_across_shots(km.b(2), out=int) == 7
    assert sim.read_across_shots(km.b(3), out=int) == 8
    assert sim.read_across_shots(km.q(99), out=int) == 9
    assert sim.read_across_shots(km.b(99), out=int) == 10


def test_read_across_shots():
    sim = km.Simulator(batch_size=5)
    circuit = km.Circuit("""
        APPEND_TO_REGISTER q0 r0
        APPEND_TO_REGISTER q1 r0
        APPEND_TO_REGISTER q2 r0
        APPEND_TO_REGISTER q3 r0
        REGISTER r0 "test"
        X q3
    """)
    sim.use_same_registers_as(circuit)
    sim.do(circuit)
    sim.write_across_shots(km.q(0), 0b01000)
    sim.write_across_shots(km.q(1), 0b00100)
    sim.write_across_shots(km.q(2), 0b00011)

    assert sim.read_across_shots(km.q(0), out=int) == 0b01000
    assert sim.read_across_shots(km.q(1), out=int) == 0b00100
    assert sim.read_across_shots(km.q(2), out=int) == 0b00011
    assert sim.read_across_shots(km.q(3), out=int) == 0b11111
    np.testing.assert_array_equal(sim.read_across_shots("test", out=int), [8, 4, 3, 31])
    np.testing.assert_array_equal(sim.read_across_shots(km.r(0), out=int), [8, 4, 3, 31])
    np.testing.assert_array_equal(sim.read_across_shots([km.q(3), km.q(1)], out=int), [31, 4])

    np.testing.assert_array_equal(sim.read_across_shots(km.q(0)), [0, 0, 0, 1, 0])
    np.testing.assert_array_equal(
        sim.read_across_shots(km.r(0)),
        [[0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]],
    )
    np.testing.assert_array_equal(
        sim.read_across_shots([km.q(3), km.q(1)]), [[1, 1, 1, 1, 1], [0, 0, 1, 0, 0]]
    )
    np.testing.assert_array_equal(
        sim.read_across_shots("test"),
        [[0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]],
    )


def test_read_within_shot():
    sim = km.Simulator(batch_size=5)
    circuit = km.Circuit("""
        APPEND_TO_REGISTER q0 r0
        APPEND_TO_REGISTER q1 r0
        APPEND_TO_REGISTER q2 r0
        APPEND_TO_REGISTER q3 r0
        REGISTER r0 "test"
        X q3
    """)
    sim.use_same_registers_as(circuit)
    sim.do(circuit)

    sim.write_across_shots(km.q(0), 0b01000)
    sim.write_across_shots(km.q(1), 0b00100)
    sim.write_across_shots(km.q(2), 0b00011)

    assert sim.read_within_shot(km.q(0), 0, out=int) == 0
    assert sim.read_within_shot(km.q(1), 0, out=int) == 0
    assert sim.read_within_shot(km.q(2), 0, out=int) == 1
    assert sim.read_within_shot(km.q(3), 0, out=int) == 1
    assert sim.read_within_shot(km.q(0), 1, out=int) == 0
    assert sim.read_within_shot(km.q(1), 1, out=int) == 0
    assert sim.read_within_shot(km.q(2), 1, out=int) == 1
    assert sim.read_within_shot(km.q(3), 1, out=int) == 1
    assert sim.read_within_shot(km.q(0), 2, out=int) == 0
    assert sim.read_within_shot(km.q(1), 2, out=int) == 1
    assert sim.read_within_shot(km.q(2), 2, out=int) == 0
    assert sim.read_within_shot(km.q(3), 2, out=int) == 1
    assert sim.read_within_shot(km.q(0), 3, out=int) == 1
    assert sim.read_within_shot(km.q(1), 3, out=int) == 0
    assert sim.read_within_shot(km.q(2), 3, out=int) == 0
    assert sim.read_within_shot(km.q(3), 3, out=int) == 1
    assert sim.read_within_shot(km.q(0), 4, out=int) == 0
    assert sim.read_within_shot(km.q(1), 4, out=int) == 0
    assert sim.read_within_shot(km.q(2), 4, out=int) == 0
    assert sim.read_within_shot(km.q(3), 4, out=int) == 1
    with pytest.raises(IndexError, match="batch_size"):
        sim.read_within_shot(km.q(0), 99, out=int)
    with pytest.raises(IndexError, match="batch_size"):
        sim.read_within_shot(km.q(0), 99)

    assert sim.read_within_shot("test", 0, out=int) == 12
    assert sim.read_within_shot("test", 1, out=int) == 12
    assert sim.read_within_shot("test", 2, out=int) == 10
    assert sim.read_within_shot("test", 3, out=int) == 9
    assert sim.read_within_shot("test", 4, out=int) == 8
    assert sim.read_within_shot(km.r(0), 1, out=int) == 12
    assert sim.read_within_shot(km.r(0), 2, out=int) == 10
    assert sim.read_within_shot([km.q(3), km.q(1)], 1, out=int) == 1
    assert sim.read_within_shot([km.q(3), km.q(1)], 2, out=int) == 3
    assert sim.read_within_shot([km.q(3), km.q(1)], 3, out=int) == 1

    assert sim.read_within_shot(km.q(0), 0) == 0
    assert sim.read_within_shot(km.q(1), 0) == 0
    assert sim.read_within_shot(km.q(2), 0) == 1
    assert sim.read_within_shot(km.q(3), 0) == 1
    assert sim.read_within_shot(km.q(0), 1) == 0
    assert sim.read_within_shot(km.q(1), 1) == 0
    assert sim.read_within_shot(km.q(2), 1) == 1
    assert sim.read_within_shot(km.q(3), 1) == 1
    np.testing.assert_array_equal(sim.read_within_shot("test", 0), [0, 0, 1, 1])
    np.testing.assert_array_equal(sim.read_within_shot("test", 1), [0, 0, 1, 1])
    np.testing.assert_array_equal(sim.read_within_shot("test", 2), [0, 1, 0, 1])
    np.testing.assert_array_equal(sim.read_within_shot("test", 3), [1, 0, 0, 1])
    np.testing.assert_array_equal(sim.read_within_shot("test", 4), [0, 0, 0, 1])
    np.testing.assert_array_equal(sim.read_within_shot(km.r(0), 1), [0, 0, 1, 1])
    np.testing.assert_array_equal(sim.read_within_shot(km.r(0), 2), [0, 1, 0, 1])
    np.testing.assert_array_equal(sim.read_within_shot([km.q(3), km.q(1)], 1), [1, 0])
    np.testing.assert_array_equal(sim.read_within_shot([km.q(3), km.q(1)], 2), [1, 1])
    np.testing.assert_array_equal(sim.read_within_shot([km.q(3), km.q(1)], 3), [1, 0])


def test_read_write_within_shot_9():
    builder = km.CircuitBuilder()
    test = builder.create_quantum_register(length=9, name="test")
    sim = km.Simulator(batch_size=256)
    sim.use_same_registers_as(builder.finish_circuit())
    sim.write_within_shot("test", 8, (1 << 9) - 1)
    assert sim.read_within_shot("test", 8, out=int) == (1 << 9) - 1


@pytest.mark.parametrize("width", [257, 300, 512, 513, 1000, 2048])
def test_read_write_within_shot_wider_than_batch_size(width: int):
    # Regression test: writing an int into a register wider than the batch size used to overflow
    # an internal byte buffer that was only sized for batch_size bits.
    builder = km.CircuitBuilder()
    builder.create_quantum_register(length=width, name="test")
    sim = km.Simulator(batch_size=4)
    sim.use_same_registers_as(builder.finish_circuit())
    values = [(1 << width) - 1, 1 << (width - 1), 0x5555 << 200, 1]
    for shot, v in enumerate(values):
        sim.write_within_shot("test", shot, v)
    for shot, v in enumerate(values):
        assert sim.read_within_shot("test", shot, out=int) == v
    with pytest.raises(ValueError, match="not in range"):
        sim.write_within_shot("test", 0, 1 << width)


def test_write_within_shot():
    sim = km.Simulator(batch_size=8)
    circuit = km.Circuit("""
        APPEND_TO_REGISTER q0 r0
        APPEND_TO_REGISTER q1 r0
        APPEND_TO_REGISTER q2 r0
        APPEND_TO_REGISTER q3 r0
        REGISTER r0 "test"
    """)
    sim.use_same_registers_as(circuit)
    sim.do(circuit)
    assert sim.read_within_shot("test", 0, out=int) == 0
    assert sim.read_within_shot("test", 1, out=int) == 0
    assert sim.read_within_shot("test", 6, out=int) == 0
    sim.write_within_shot(km.q(0), 0, True)
    assert sim.read_within_shot("test", 0, out=int) == 1
    assert sim.read_within_shot("test", 1, out=int) == 0
    assert sim.read_within_shot("test", 6, out=int) == 0
    sim.write_within_shot("test", 6, 13)
    assert sim.read_within_shot("test", 0, out=int) == 1
    assert sim.read_within_shot("test", 1, out=int) == 0
    assert sim.read_within_shot("test", 6, out=int) == 13
    assert sim.read_within_shot("test", 7, out=int) == 0
    sim.write_within_shot("test", 7, np.array([True, True, False, True], dtype=np.bool_))
    assert sim.read_within_shot("test", 7, out=int) == 0b1011

    with pytest.raises(IndexError, match="batch_size"):
        sim.write_within_shot("test", 8, 13)
    with pytest.raises(ValueError, match="how to write"):
        sim.write_within_shot("test", 8, object())
    with pytest.raises(ValueError, match="how to write"):
        sim.write_within_shot("test", 8, None)
    with pytest.raises(ValueError, match="how to write"):
        sim.write_within_shot("test", 8, "01")


def test_write_across_shots():
    sim = km.Simulator(batch_size=8)
    circuit = km.Circuit("""
        APPEND_TO_REGISTER q0 r0
        APPEND_TO_REGISTER q1 r0
        APPEND_TO_REGISTER q2 r0
        APPEND_TO_REGISTER q3 r0
        REGISTER r0 "test"
    """)
    sim.use_same_registers_as(circuit)
    sim.do(circuit)

    sim.write_across_shots(km.q(0), 255)
    assert sim.read_across_shots(km.q(0), out=int) == 255
    sim.write_across_shots(km.q(0), np.array([1, 0, 1, 0, 1, 1, 0, 1], dtype=np.bool_))
    assert sim.read_across_shots(km.q(0), out=int) == 0b10110101

    with pytest.raises(ValueError, match="not in range"):
        sim.write_across_shots(km.q(0), 256)
    with pytest.raises(ValueError, match="how to write"):
        sim.write_across_shots(km.q(0), object())
    with pytest.raises(TypeError, match="Expected a q"):
        sim.write_across_shots(object(), 255)


def test_write_across_shots_multiple_indices():
    sim = km.Simulator(batch_size=8)
    circuit = km.Circuit("""
        APPEND_TO_REGISTER q0 r0
        APPEND_TO_REGISTER q1 r0
        APPEND_TO_REGISTER q2 r0
        APPEND_TO_REGISTER q3 r0
        REGISTER r0 "test"
    """)
    sim.use_same_registers_as(circuit)

    # One int per index, each bit packed across shots, matching read's layout.
    want = [0b10110101, 0, 255, 0b11110000]
    sim.write_across_shots("test", want)
    assert sim.read_across_shots("test", out=int) == want
    assert sim.read_across_shots(km.q(0), out=int) == 0b10110101
    assert sim.read_across_shots(km.q(3), out=int) == 0b11110000

    # An explicit array of ids works the same way, in the given order.
    sim.write_across_shots(km.array([km.q(3), km.q(0)]), [1, 2])
    assert sim.read_across_shots(km.q(3), out=int) == 1
    assert sim.read_across_shots(km.q(0), out=int) == 2

    # Bits and qubits can be mixed, and indices need not already exist.
    sim.write_across_shots(km.array([km.b(5), km.q(70)]), [3, 4])
    assert sim.read_across_shots(km.b(5), out=int) == 3
    assert sim.read_across_shots(km.q(70), out=int) == 4


def test_write_across_shots_multiple_indices_bool_array():
    sim = km.Simulator(batch_size=8)
    indices = km.array([km.q(0), km.q(1)])
    sim.write_across_shots(
        indices, np.array([[1, 0, 1, 0, 1, 1, 0, 1], [0, 0, 0, 0, 1, 1, 1, 1]], dtype=np.bool_)
    )
    assert sim.read_across_shots(km.q(0), out=int) == 0b10110101
    assert sim.read_across_shots(km.q(1), out=int) == 0b11110000


def test_write_across_shots_multiple_indices_errors():
    sim = km.Simulator(batch_size=8)
    indices = km.array([km.q(0), km.q(1)])

    with pytest.raises(ValueError, match="len\\(new_value\\)"):
        sim.write_across_shots(indices, [1, 2, 3])
    with pytest.raises(ValueError, match="not in range"):
        sim.write_across_shots(indices, [1, 256])
    with pytest.raises(ValueError, match="how to write"):
        sim.write_across_shots(indices, 255)
    with pytest.raises(ValueError, match="new_value.shape"):
        sim.write_across_shots(indices, np.zeros((2, 7), dtype=np.bool_))


def test_write_across_shots_round_trips_read_across_shots():
    rng = random.Random(42)
    sim = km.Simulator(batch_size=257)
    indices = km.array([km.q(k) for k in range(20)] + [km.b(k) for k in range(20)])
    want = [rng.randrange(1 << 257) for _ in range(40)]
    sim.write_across_shots(indices, want)
    assert sim.read_across_shots(indices, out=int) == want


@pytest.mark.parametrize('batch_size', [0, 1, 2, 3, 4, 5, 6, 7, 8, 255, 256, 257])
@pytest.mark.parametrize('index', [km.q(31), km.b(21)])
def test_sim_get_item(batch_size: int, index: Any):
    sim = km.Simulator(batch_size=batch_size)
    assert sim.read_across_shots(index, out=int) == 0
    assert sim.read_across_shots(km.q(55), out=int) == 0

    if isinstance(index, km.q):
        sim.do(km.Circuit(f"X {index}"))
    else:
        sim.do(km.Circuit(f"BIT_INVERT {index}"))
    assert sim.read_across_shots(index, out=int) == (1 << batch_size) - 1
    assert sim.read_across_shots(km.q(55), out=int) == 0

    assert sim.read_across_shots(True, out=int) == (1 << batch_size) - 1
    assert sim.read_across_shots(False, out=int) == 0


@pytest.mark.parametrize('batch_size', [0, 1, 2, 3, 4, 5, 6, 7, 8, 255, 256, 257])
@pytest.mark.parametrize('index', [km.q(31), km.b(21)])
def test_sim_set_item(batch_size: int, index: Any):
    sim = km.Simulator(batch_size=batch_size)
    assert sim.read_across_shots(index, out=int) == 0

    v = (1 << batch_size) - 1
    sim.write_across_shots(index, v)
    assert sim.read_across_shots(index, out=int) == v

    if batch_size > 0:
        v = 1
        sim.write_across_shots(index, v)
        assert sim.read_across_shots(index, out=int) == v

    v = random.randrange(1 << batch_size)
    sim.write_across_shots(index, v)
    assert sim.read_across_shots(index, out=int) == v
    assert sim.read_across_shots(km.q(55), out=int) == 0


@pytest.mark.parametrize('batch_size', [0, 1, 2, 3, 4, 5, 6, 7, 8, 255, 256, 257])
@pytest.mark.parametrize('index', [km.q(31), km.b(21)])
def test_sim_set_item_out_of_range(batch_size: int, index: Any):
    sim = km.Simulator(batch_size=batch_size)
    with pytest.raises(ValueError, match="not in range"):
        sim.write_across_shots(index, -1)
    with pytest.raises(ValueError, match="not in range"):
        sim.write_across_shots(index, 1 << (batch_size + 100))
    with pytest.raises(ValueError, match="not in range"):
        sim.write_across_shots(index, 1 << batch_size)


def test_read_within_shot_register_larger_than_batch_size():
    sim = km.Simulator(batch_size=1)
    assert sim.read_across_shots(km.q(0), out=int) == 0
    assert sim.read_across_shots(km.b(0), out=int) == 0
    c = km.Circuit('''
        APPEND_TO_REGISTER q0 r0
        APPEND_TO_REGISTER q1 r0
        APPEND_TO_REGISTER q2 r0
        APPEND_TO_REGISTER q3 r0
        REGISTER r0 "reg"
    ''')
    sim.use_same_registers_as(c)
    sim.write_within_shot("reg", 0, 0b1101)
    assert sim.read_within_shot("reg", 0, out=int) == 0b1101


def test_read_write_random_phases():
    sim = km.Simulator(batch_size=409)
    phases = [random.random() * 2 for _ in range(sim.batch_size)]
    for k in range(sim.batch_size):
        sim.write_shot_phase(k, phases[k])
    for k in range(sim.batch_size):
        assert sim.read_shot_phase(k) == fractions.Fraction(phases[k])


def test_read_write_phase():
    sim = km.Simulator(batch_size=409)
    eps = 2**-66
    for k in range(sim.batch_size):
        sim.write_shot_phase(k, eps * k)
    for k in range(sim.batch_size):
        assert sim.read_shot_phase(k) == fractions.Fraction(eps * k)

    # Affected by Z_POW
    sim.do(km.Circuit("""
        X q0
        Z_POW q0 0.25
    """))
    for k in range(sim.batch_size):
        assert sim.read_shot_phase(k) == fractions.Fraction(eps * k) + fractions.Fraction(0.25)
    sim.write_within_shot(km.q(0), 5, 0)

    # Affected by Z
    sim.do(km.Circuit("""
        Z q0
    """))
    for k in range(sim.batch_size):
        offset = fractions.Fraction(1) if k != 5 else fractions.Fraction(0)
        assert (
            sim.read_shot_phase(k)
            == fractions.Fraction(eps * k) + fractions.Fraction(0.25) + offset
        )


def test_read_phase_flipped_across_shots():
    sim = km.Simulator(batch_size=409)
    assert sim.read_phase_flipped_across_shots(out=int) == 0
    sim.do(km.Circuit("""
        X q0
        Z q0
    """))
    assert sim.read_phase_flipped_across_shots(out=int) == (1 << 409) - 1
    sim.do(km.Circuit("""
        Z q0
    """))
    assert sim.read_phase_flipped_across_shots(out=int) == 0
    sim.do(km.Circuit("""
        Z_POW q0 1.0
    """))
    assert sim.read_phase_flipped_across_shots(out=int) == (1 << 409) - 1
    sim.clear_for_shot()
    assert sim.read_phase_flipped_across_shots(out=int) == 0
    for k in range(21):
        sim.write_shot_phase(k, k * 0.101)
    assert sim.read_phase_flipped_across_shots(out=int) == 0b111111111100000
    np.testing.assert_array_equal(
        sim.read_phase_flipped_across_shots(), [False] * 5 + [True] * 10 + [False] * 394
    )
