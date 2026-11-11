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


def test_builder_append_z_simple():
    builder = km.CircuitBuilder()
    builder.z(km.q(1))
    builder.z(km.b(3))
    builder.z(True)
    builder.z(False)
    assert builder.finish_circuit() == km.Circuit("""
        Z q1
        NEG if b3
        NEG
    """)


@pytest.mark.parametrize("t", util.z_cases(1))
def test_builder_z_produces_a_circuit_behaving_as_expected(t: Any):
    builder = km.CircuitBuilder()
    builder.z(t)

    sim = km.Simulator(batch_size=64)
    t_val = util.store64(sim, t, util.U64_BIT1)

    sim.do(builder.finish_circuit())
    assert util.read64(sim, t) == t_val
    assert sim.read_phase_flipped_across_shots(out=int) == t_val


@pytest.mark.parametrize("t", util.z_list_cases(0, 20))
def test_builder_broadcast_z_behaves_identically_to_repeated_appends(t: Any):
    builder = km.CircuitBuilder()
    builder.z(t)
    circuit = builder.finish_circuit()

    reference_builder = km.CircuitBuilder()
    (t,) = util.broadcast_logic(t)
    for k in range(len(t)):
        reference_builder.z(t[k])
    reference_circuit = reference_builder.finish_circuit()

    assert circuit == reference_circuit


def test_builder_z_errors():
    builder = km.CircuitBuilder()
    with pytest.raises((TypeError, ValueError)):
        builder.z(km.xb(0))
    with pytest.raises((TypeError, ValueError)):
        builder.z(km.xbool(True))
    with pytest.raises((TypeError, ValueError)):
        builder.z(km.array([km.q(0), km.xb(1)]))
    assert builder.finish_circuit() == km.Circuit()
