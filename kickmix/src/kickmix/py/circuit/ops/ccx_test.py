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

from typing import Any

import op_testing_util as util
import pytest

import kickmix as km


def test_builder_ccx_simple():
    builder = km.CircuitBuilder()
    qs = [km.q(k) for k in range(10)]

    builder.ccx(qs[0], qs[1], qs[2])
    builder.ccx(qs[0], True, qs[2])
    assert builder.finish_circuit() == km.Circuit("""
        CCX q0 q1 q2
        CX q0 q2
    """)


@pytest.mark.parametrize("c1", util.z_cases(0))
@pytest.mark.parametrize("c2", util.z_cases(1))
@pytest.mark.parametrize("t", util.x_cases_q(2))
def test_builder_ccx_produces_a_circuit_behaving_as_expected(c1: Any, c2: Any, t: Any):
    builder = km.CircuitBuilder()
    builder.ccx(c1, c2, t)

    sim = km.Simulator(batch_size=64)
    c1_val = util.store64(sim, c1, util.U64_BIT0)
    c2_val = util.store64(sim, c2, util.U64_BIT1)
    t_val = util.store64(sim, t, util.U64_BIT2)

    sim.do(builder.finish_circuit())
    assert util.read64(sim, c1) == c1_val
    assert util.read64(sim, c2) == c2_val
    assert util.read64(sim, t) == t_val ^ (c1_val & c2_val)
    assert sim.read_phase_flipped_across_shots(out=int) == 0


@pytest.mark.parametrize("c2", util.z_list_cases(0, 32))
@pytest.mark.parametrize("c1", util.z_list_cases(32, 32))
@pytest.mark.parametrize("t", util.x_list_cases(64, 32, include_x_bits=False))
def test_builder_ccx_behaves_identically_to_repeated_appends(c2: Any, c1: Any, t: Any):
    builder = km.CircuitBuilder()
    builder.ccx(c2, c1, t)
    circuit = builder.finish_circuit()

    reference_builder = km.CircuitBuilder()
    c2, c1, t = util.broadcast_logic(c2, c1, t)
    for k in range(len(c1)):
        reference_builder.ccx(c2[k], c1[k], t[k])
    reference_circuit = reference_builder.finish_circuit()

    assert circuit == reference_circuit


def test_builder_del_and_clears_the_target_without_toffolis():
    builder = km.CircuitBuilder()
    c1 = builder.create_quantum_register(1, name="c1")
    c2 = builder.create_quantum_register(1, name="c2")
    t = builder.create_quantum_register(2, name="t")
    builder.ccx(c1[0], c2[0], t)
    builder.del_and(c1[0], c2[0], t)
    circuit = builder.finish_circuit()

    # Only the two computing CCXs are magic; the uncomputation is free.
    assert circuit.max_magic() == 2

    sim = km.Simulator(batch_size=4)
    sim.use_same_registers_as(circuit)
    for slot, (a, b) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        sim.write_within_shot("c1", slot, a)
        sim.write_within_shot("c2", slot, b)
    sim.do(circuit)

    assert sim.read_phase_flipped_across_shots(out=int) == 0
    for slot in range(4):
        assert sim.read_within_shot("t", slot, out=int) == 0


@pytest.mark.parametrize("c2", util.z_list_cases(0, 32))
@pytest.mark.parametrize("c1", util.z_list_cases(32, 32))
@pytest.mark.parametrize("t", util.x_list_cases(64, 32, include_x_bits=False))
def test_builder_del_and_behaves_identically_to_repeated_appends(c2: Any, c1: Any, t: Any):
    builder = km.CircuitBuilder()
    builder.del_and(c2, c1, t)
    circuit = builder.finish_circuit()

    reference_builder = km.CircuitBuilder()
    c2, c1, t = util.broadcast_logic(c2, c1, t)
    for k in range(len(c1)):
        reference_builder.del_and(c2[k], c1[k], t[k])
    reference_circuit = reference_builder.finish_circuit()

    assert circuit == reference_circuit


def test_builder_init_and_resets_and_computes():
    builder = km.CircuitBuilder()
    c1 = builder.create_quantum_register(1, name="c1")
    c2 = builder.create_quantum_register(1, name="c2")
    t = builder.create_quantum_register(1, name="t")
    # Set t to 1 initially to verify that init_and resets it before computing
    builder.x(t[0])
    builder.init_and(c1[0], c2[0], t[0])
    circuit = builder.finish_circuit()

    sim = km.Simulator(batch_size=4)
    sim.use_same_registers_as(circuit)
    for slot, (a, b) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        sim.write_within_shot("c1", slot, a)
        sim.write_within_shot("c2", slot, b)
    sim.do(circuit)

    for slot, (a, b) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        assert sim.read_within_shot("t", slot, out=int) == (a & b)


@pytest.mark.parametrize("c2", util.z_list_cases(0, 32))
@pytest.mark.parametrize("c1", util.z_list_cases(32, 32))
@pytest.mark.parametrize("t", util.x_list_cases(64, 32, include_x_bits=False))
def test_builder_init_and_behaves_identically_to_repeated_appends(c2: Any, c1: Any, t: Any):
    builder = km.CircuitBuilder()
    builder.init_and(c2, c1, t)
    circuit = builder.finish_circuit()

    reference_builder = km.CircuitBuilder()
    c2, c1, t = util.broadcast_logic(c2, c1, t)
    for k in range(len(c1)):
        reference_builder.init_and(c2[k], c1[k], t[k])
    reference_circuit = reference_builder.finish_circuit()

    assert circuit == reference_circuit
