from __future__ import annotations

from typing import Any

import op_testing_util as util
import pytest

import kickmix as km


def test_builder_cz_simple():
    builder = km.CircuitBuilder()
    builder.cz(km.q(1), km.q(2))
    builder.cz(True, km.q(3))
    assert builder.finish_circuit() == km.Circuit("""
        CZ q1 q2
        Z q3
    """)


@pytest.mark.parametrize("c1", util.z_cases(0))
@pytest.mark.parametrize("c2", util.z_cases(1))
def test_builder_cz_produces_a_circuit_behaving_as_expected(c1: Any, c2: Any):
    builder = km.CircuitBuilder()
    builder.cz(c1, c2)

    sim = km.Simulator(batch_size=64)
    c1_val = util.store64(sim, c1, util.U64_BIT0)
    c2_val = util.store64(sim, c2, util.U64_BIT1)

    sim.do(builder.finish_circuit())
    assert util.read64(sim, c1) == c1_val
    assert util.read64(sim, c2) == c2_val
    assert sim.read_phase_flipped_across_shots(out=int) == c1_val & c2_val


@pytest.mark.parametrize("c1", util.z_list_cases(0, 128))
@pytest.mark.parametrize("c2", util.z_list_cases(256, 128))
def test_builder_broadcast_cz_behaves_identically_to_repeated_appends(c1: Any, c2: Any):
    builder = km.CircuitBuilder()
    builder.cz(c1, c2)
    circuit = builder.finish_circuit()

    reference_builder = km.CircuitBuilder()
    c1, c2 = util.broadcast_logic(c1, c2)
    for k in range(len(c1)):
        reference_builder.cz(c1[k], c2[k])
    reference_circuit = reference_builder.finish_circuit()

    assert circuit == reference_circuit
