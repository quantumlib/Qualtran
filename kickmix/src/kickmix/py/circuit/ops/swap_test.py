from __future__ import annotations

from typing import Any

import kickmix as km
import pytest

import op_testing_util as util


def test_builder_swap_simple():
    builder = km.CircuitBuilder()
    builder.swap(km.q(1), km.q(2))
    builder.swap(km.b(3), km.b(4))
    assert builder.finish_circuit() == km.Circuit("""
        SWAP q1 q2
        BIT_INVERT b3 if b4
        BIT_INVERT b4 if b3
        BIT_INVERT b3 if b4
    """)


@pytest.mark.parametrize("c1,c2", [
    (km.q(0), km.q(1)),
    (km.b(0), km.b(1)),
    (km.xb(0), km.xb(1)),
])
def test_builder_swap_produces_a_circuit_behaving_as_expected(c1: Any, c2: Any):
    builder = km.CircuitBuilder()
    builder.swap(c1, c2)

    sim = km.Simulator(batch_size=64)
    c1_val = util.store64(sim, c1, util.U64_BIT0)
    c2_val = util.store64(sim, c2, util.U64_BIT1)

    sim.do(builder.finish_circuit())
    assert util.read64(sim, c1) == c2_val
    assert util.read64(sim, c2) == c1_val
    assert sim.read_phase_flipped_across_shots(out=int) == 0


@pytest.mark.parametrize("c1,c2", [
    ([km.q(k) for k in range(256)], [km.q(k) for k in range(256, 512)]),
    ([km.b(k) for k in range(256)], [km.b(k) for k in range(256, 512)]),
    (km.q(0), km.q(1)),
    (km.b(0), km.b(1)),
    (km.q(0), [km.q(1), km.q(2)]),
    (km.q(0), km.array([km.q(1), km.q(2)])),
    ([km.q(1), km.q(2)], km.q(0)),
    (km.b(0), [km.b(1), km.b(2)]),
    ([km.b(0), km.q(0)], [km.b(1), km.q(1)]),
])
def test_builder_broadcast_swap_behaves_identically_to_repeated_appends(c1: Any, c2: Any):
    builder = km.CircuitBuilder()
    builder.swap(c1, c2)
    circuit = builder.finish_circuit()

    reference_builder = km.CircuitBuilder()
    c1, c2 = util.broadcast_logic(c1, c2)
    for k in range(len(c1)):
        reference_builder.swap(c1[k], c2[k])
    reference_circuit = reference_builder.finish_circuit()

    assert circuit == reference_circuit


@pytest.mark.parametrize("control", [km.q(10), km.b(0), True, False])
def test_builder_cswap(control: Any):
    builder = km.CircuitBuilder()
    r1 = builder.create_quantum_register(4, name="r1")
    r2 = builder.create_quantum_register(4, name="r2")
    builder.cswap(control, r1, r2)
    circuit = builder.finish_circuit()

    sim = km.Simulator(batch_size=2)
    sim.use_same_registers_as(circuit)
    if not isinstance(control, bool):
        sim.write_within_shot(control, 0, 0)
        sim.write_within_shot(control, 1, 1)
    for shot in range(2):
        sim.write_within_shot("r1", shot, 0b1010)
        sim.write_within_shot("r2", shot, 0b0101)
    sim.do(circuit)

    for shot in range(2):
        active = control if isinstance(control, bool) else bool(shot)
        assert sim.read_within_shot("r1", shot, out=int) == (0b0101 if active else 0b1010)
        assert sim.read_within_shot("r2", shot, out=int) == (0b1010 if active else 0b0101)

