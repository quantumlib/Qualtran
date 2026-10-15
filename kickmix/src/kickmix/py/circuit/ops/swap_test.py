#  Copyright 2026 Google LLC
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

from __future__ import annotations

import io
from typing import Any

import op_testing_util as util
import pytest

import kickmix as km


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


@pytest.mark.parametrize("c1,c2", [(km.q(0), km.q(1)), (km.b(0), km.b(1)), (km.xb(0), km.xb(1))])
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


@pytest.mark.parametrize(
    "c1,c2",
    [
        ([km.q(k) for k in range(256)], [km.q(k) for k in range(256, 512)]),
        ([km.b(k) for k in range(256)], [km.b(k) for k in range(256, 512)]),
        (km.q(0), km.q(1)),
        (km.b(0), km.b(1)),
        (km.q(0), [km.q(1), km.q(2)]),
        (km.q(0), km.array([km.q(1), km.q(2)])),
        ([km.q(1), km.q(2)], km.q(0)),
        (km.b(0), [km.b(1), km.b(2)]),
        ([km.b(0), km.q(0)], [km.b(1), km.q(1)]),
    ],
)
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


def test_builder_cswap_simple():
    builder = km.CircuitBuilder()
    qs = [km.q(k) for k in range(10)]

    builder.cswap(qs[0], qs[1], qs[2])
    builder.cswap(True, qs[1], qs[2])
    builder.cswap(False, qs[1], qs[2])
    builder.cswap(km.b(0), qs[1], qs[2])
    assert builder.finish_circuit() == km.Circuit("""
        CSWAP q0 q1 q2
        SWAP q1 q2
        SWAP q1 q2 if b0
    """)


@pytest.mark.parametrize("c", util.z_cases(0))
@pytest.mark.parametrize("t1", util.x_cases_q(1))
@pytest.mark.parametrize("t2", util.x_cases_q(2))
def test_builder_cswap_produces_a_circuit_behaving_as_expected(c: Any, t1: Any, t2: Any):
    builder = km.CircuitBuilder()
    builder.cswap(c, t1, t2)

    sim = km.Simulator(batch_size=64)
    c_val = util.store64(sim, c, util.U64_BIT0)
    t1_val = util.store64(sim, t1, util.U64_BIT1)
    t2_val = util.store64(sim, t2, util.U64_BIT2)

    sim.do(builder.finish_circuit())
    d = (t1_val ^ t2_val) & c_val
    assert util.read64(sim, c) == c_val
    assert util.read64(sim, t1) == t1_val ^ d
    assert util.read64(sim, t2) == t2_val ^ d
    assert sim.read_phase_flipped_across_shots(out=int) == 0


@pytest.mark.parametrize("c", util.z_list_cases(0, 32))
@pytest.mark.parametrize("t1", util.x_list_cases(32, 32, include_x_bits=False))
@pytest.mark.parametrize("t2", util.x_list_cases(64, 32, include_x_bits=False))
def test_builder_cswap_behaves_identically_to_repeated_appends(c: Any, t1: Any, t2: Any):
    builder = km.CircuitBuilder()
    builder.cswap(c, t1, t2)
    circuit = builder.finish_circuit()

    reference_builder = km.CircuitBuilder()
    c, t1, t2 = util.broadcast_logic(c, t1, t2)
    for k in range(len(c)):
        reference_builder.cswap(c[k], t1[k], t2[k])
    reference_circuit = reference_builder.finish_circuit()

    assert circuit == reference_circuit


@pytest.mark.parametrize("control", [km.q(10), km.b(0), True, False])
def test_builder_cswap(control: Any):
    builder = km.CircuitBuilder()
    r1 = builder.create_quantum_register(4, name="r1")
    r2 = builder.create_quantum_register(4, name="r2")
    builder.cswap(control, r1, r2)
    circuit = builder.finish_circuit()

    if control == km.q(10):
        assert circuit == km.Circuit("""
            REGISTER r0 "r1"
            APPEND_TO_REGISTER q0 r0
            APPEND_TO_REGISTER q1 r0
            APPEND_TO_REGISTER q2 r0
            APPEND_TO_REGISTER q3 r0
            REGISTER r1 "r2"
            APPEND_TO_REGISTER q4 r1
            APPEND_TO_REGISTER q5 r1
            APPEND_TO_REGISTER q6 r1
            APPEND_TO_REGISTER q7 r1
            CSWAP q10 q0 q4
            CSWAP q10 q1 q5
            CSWAP q10 q2 q6
            CSWAP q10 q3 q7
        """)

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


def test_cswap_kmx_and_sim():
    c = km.Circuit("""
        CSWAP q0 q1 q2
        CSWAP q3 q4 q5 if b0
    """)
    assert str(c).strip() == "CSWAP q0 q1 q2\nCSWAP q3 q4 q5 if b0"
    assert c.max_magic() == 2
    assert c.reaction_depth() == 1
    bio = io.BytesIO()
    c.to_file(bio, format="kmb")
    bio.seek(0)
    assert km.Circuit.from_file(bio, format="kmb") == c
    sim = km.Simulator(batch_size=2)
    sim.write_within_shot(km.q(0), 0, 0)
    sim.write_within_shot(km.q(0), 1, 1)
    sim.write_within_shot(km.q(1), 0, 1)
    sim.write_within_shot(km.q(1), 1, 1)
    sim.write_within_shot(km.q(2), 0, 0)
    sim.write_within_shot(km.q(2), 1, 0)
    sim.write_within_shot(km.b(0), 0, 0)
    sim.write_within_shot(km.b(0), 1, 1)
    sim.write_within_shot(km.q(3), 0, 1)
    sim.write_within_shot(km.q(3), 1, 1)
    sim.write_within_shot(km.q(4), 0, 1)
    sim.write_within_shot(km.q(4), 1, 1)
    sim.write_within_shot(km.q(5), 0, 0)
    sim.write_within_shot(km.q(5), 1, 0)
    sim.do(c)
    assert sim.read_within_shot(km.q(1), 0, out=int) == 1
    assert sim.read_within_shot(km.q(2), 0, out=int) == 0
    assert sim.read_within_shot(km.q(1), 1, out=int) == 0
    assert sim.read_within_shot(km.q(2), 1, out=int) == 1
    assert sim.read_within_shot(km.q(4), 0, out=int) == 1
    assert sim.read_within_shot(km.q(5), 0, out=int) == 0
    assert sim.read_within_shot(km.q(4), 1, out=int) == 0
    assert sim.read_within_shot(km.q(5), 1, out=int) == 1
