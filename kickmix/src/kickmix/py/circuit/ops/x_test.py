from __future__ import annotations

from typing import Any

import op_testing_util as util
import pytest

import kickmix as km


def test_builder_append_x_simple():
    builder = km.CircuitBuilder()
    builder.x(km.q(1))
    builder.x(km.xb(3))
    builder.x("|->")
    assert builder.finish_circuit() == km.Circuit("""
        X q1
        NEG if b3
        NEG
    """)


@pytest.mark.parametrize("t", util.x_cases_q(1))
def test_builder_x_produces_a_circuit_behaving_as_expected(t: Any):
    builder = km.CircuitBuilder()
    builder.x(t)

    sim = km.Simulator(batch_size=64)
    t_val = util.store64(sim, t, util.U64_BIT1)

    sim.do(builder.finish_circuit())
    assert util.read64(sim, t) == t_val ^ util.U64_TRUE
    assert sim.read_phase_flipped_across_shots(out=int) == 0


@pytest.mark.parametrize("t", util.x_cases_bc(1))
def test_builder_x_produces_a_circuit_behaving_as_expected_xbit(t: Any):
    builder = km.CircuitBuilder()
    builder.x(t)

    sim = km.Simulator(batch_size=64)
    t_val = util.store64(sim, t, util.U64_BIT1)

    sim.do(builder.finish_circuit())
    assert util.read64(sim, t) == t_val
    assert sim.read_phase_flipped_across_shots(out=int) == t_val


@pytest.mark.parametrize("t", util.x_list_cases(0, 20))
def test_builder_broadcast_x_behaves_identically_to_repeated_appends(t: Any):
    builder = km.CircuitBuilder()
    builder.x(t)
    circuit = builder.finish_circuit()

    reference_builder = km.CircuitBuilder()
    (t,) = util.broadcast_logic(t)
    for k in range(len(t)):
        reference_builder.x(t[k])
    reference_circuit = reference_builder.finish_circuit()

    assert circuit == reference_circuit
