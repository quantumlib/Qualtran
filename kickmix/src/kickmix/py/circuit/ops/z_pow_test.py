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

import fractions

import op_testing_util as util

import kickmix as km


def test_builder_z_pow_simple():
    builder = km.CircuitBuilder()
    builder.z_pow(km.q(0), 0.25)
    builder.z_pow(km.q(1), fractions.Fraction(1, 10))
    builder.z_pow(km.q(2), 0.1)
    assert builder.finish_circuit() == km.Circuit("""
        Z_POW q0 0.25
        Z_POW q1 0.1
        Z_POW q2 0.1000000000000000055511151231257827021181583404541015625
    """)


def test_ccz_decomposed_into_7_z_pow():
    c = km.Circuit("""
        Z_POW q0 0.25
        Z_POW q1 0.25
        Z_POW q2 0.25
        CX q1 q2
        Z_POW q2 -0.25
        CX q0 q2
        Z_POW q2 0.25
        CX q1 q2
        Z_POW q2 -0.25
        CX q0 q2
        CX q0 q1
        Z_POW q1 -0.25
        CX q0 q1
    """)

    sim = km.Simulator(batch_size=64)
    q0_val = util.store64(sim, km.q(0), util.U64_BIT0)
    q1_val = util.store64(sim, km.q(1), util.U64_BIT1)
    q2_val = util.store64(sim, km.q(2), util.U64_BIT2)

    sim.do(c)
    assert util.read64(sim, km.q(0)) == q0_val
    assert util.read64(sim, km.q(1)) == q1_val
    assert util.read64(sim, km.q(2)) == q2_val
    assert sim.read_phase_flipped_across_shots(out=int) == q0_val & q1_val & q2_val
    for k in range(sim.batch_size):
        assert sim.read_shot_phase(k) % 1 == 0
