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


def test_builder_ccz_simple():
    builder = km.CircuitBuilder()
    builder.ccz(km.q(0), km.q(1), km.q(2))
    builder.ccz(True, km.q(3), km.b(4))
    builder.ccz(False, km.q(5), km.b(6))
    assert builder.finish_circuit() == km.Circuit("""
        CCZ q0 q1 q2
        Z q3 if b4
    """)


@pytest.mark.parametrize("c1", util.z_cases(0))
@pytest.mark.parametrize("c2", util.z_cases(1))
@pytest.mark.parametrize("c3", util.z_cases(2))
def test_builder_ccz_produces_a_circuit_behaving_as_expected(c1: Any, c2: Any, c3: Any):
    builder = km.CircuitBuilder()
    builder.ccz(c1, c2, c3)

    sim = km.Simulator(batch_size=64)
    c1_val = util.store64(sim, c1, util.U64_BIT0)
    c2_val = util.store64(sim, c2, util.U64_BIT1)
    c3_val = util.store64(sim, c3, util.U64_BIT2)

    sim.do(builder.finish_circuit())
    assert util.read64(sim, c1) == c1_val
    assert util.read64(sim, c2) == c2_val
    assert util.read64(sim, c3) == c3_val
    assert sim.read_phase_flipped_across_shots(out=int) == c1_val & c2_val & c3_val


@pytest.mark.parametrize("c1", util.z_list_cases(0, 32))
@pytest.mark.parametrize("c2", util.z_list_cases(32, 32))
@pytest.mark.parametrize("c3", util.z_list_cases(64, 32))
def test_builder_ccz_behaves_identically_to_repeated_appends(c1: Any, c2: Any, c3: Any):
    builder = km.CircuitBuilder()
    builder.ccz(c1, c2, c3)
    circuit = builder.finish_circuit()

    reference_builder = km.CircuitBuilder()
    c1, c2, c3 = util.broadcast_logic(c1, c2, c3)
    for k in range(len(c1)):
        reference_builder.ccz(c1[k], c2[k], c3[k])
    reference_circuit = reference_builder.finish_circuit()

    assert circuit == reference_circuit


def test_builder_ccz_errors():
    builder = km.CircuitBuilder()
    with pytest.raises((TypeError, ValueError)):
        builder.ccz(km.xb(0), km.q(1), km.q(2))
    with pytest.raises((TypeError, ValueError)):
        builder.ccz(km.q(0), km.xbool(True), km.q(2))
    with pytest.raises((TypeError, ValueError)):
        builder.ccz(km.q(0), km.q(1), km.xb(2))
    with pytest.raises((TypeError, ValueError)):
        builder.ccz(
            km.array([km.q(0), km.q(1)]),
            km.array([km.q(2), km.q(3)]),
            km.array([km.q(4), km.xb(5)]),
        )
    assert builder.finish_circuit() == km.Circuit()
