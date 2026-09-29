from __future__ import annotations

from typing import Any

import op_testing_util as util
import pytest

import kickmix as km


def test_builder_cx_simple():
    builder = km.CircuitBuilder()
    builder.cx(km.q(1), km.q(2))
    builder.cx(True, km.q(3))
    assert builder.finish_circuit() == km.Circuit("""
        CX q1 q2
        X q3
    """)


@pytest.mark.parametrize("c", util.z_cases(0))
@pytest.mark.parametrize("t", util.x_cases_q(1))
def test_builder_cx_produces_a_circuit_behaving_as_expected(c: Any, t: Any):
    builder = km.CircuitBuilder()
    builder.cx(c, t)

    sim = km.Simulator(batch_size=64)
    c_val = util.store64(sim, c, util.U64_BIT0)
    t_val = util.store64(sim, t, util.U64_BIT1)

    sim.do(builder.finish_circuit())
    assert util.read64(sim, c) == c_val
    assert util.read64(sim, t) == t_val ^ c_val
    assert sim.read_phase_flipped_across_shots(out=int) == 0


@pytest.mark.parametrize("c", util.z_cases(0))
@pytest.mark.parametrize("t", util.x_cases_bc(1))
def test_builder_cx_produces_a_circuit_behaving_as_expected_x_target(c: Any, t: Any):
    builder = km.CircuitBuilder()
    builder.cx(c, t)

    sim = km.Simulator(batch_size=64)
    c_val = util.store64(sim, c, util.U64_BIT0)
    t_val = util.store64(sim, t, util.U64_BIT1)

    sim.do(builder.finish_circuit())
    assert util.read64(sim, c) == c_val
    assert util.read64(sim, t) == t_val
    assert sim.read_phase_flipped_across_shots(out=int) == c_val & t_val


@pytest.mark.parametrize("c", util.z_list_cases(0, 20))
@pytest.mark.parametrize("t", util.x_list_cases(20, 20))
def test_builder_broadcast_cx_behaves_identically_to_repeated_appends(c: Any, t: Any):
    builder = km.CircuitBuilder()
    builder.cx(c, t)
    circuit = builder.finish_circuit()

    reference_builder = km.CircuitBuilder()
    c, t = util.broadcast_logic(c, t)
    for k in range(len(c)):
        reference_builder.cx(c[k], t[k])
    reference_circuit = reference_builder.finish_circuit()

    assert circuit == reference_circuit
