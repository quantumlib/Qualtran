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

import pytest

import kickmix as km


def test_circuit():
    with pytest.raises(ValueError, match="unknown operation name"):
        km.Circuit("test")
    c = km.Circuit("X q0")
    assert len(c) == 1


def test_circuit_max_t_and_max_rotations():
    empty = km.Circuit("")
    assert empty.max_magic() == 0
    assert empty.max_t() == 0
    assert empty.max_rotations() == 0

    c = km.Circuit("""
        CCX q0 q1 q2
        CCZ q0 q1 q2 if b0
        Z q0
        Z_POW q0 0.0
        Z_POW q0 0.5
        Z_POW q0 1.0
        Z_POW q0 1.5
        Z_POW q0 -0.5 if b0
        Z_POW q0 0.25
        Z_POW q0 0.75
        Z_POW q0 -0.25 if b0
        Z_POW q0 0.125
        Z_POW q0 0.1 if b1
    """)
    assert c.max_magic() == 2
    assert c.max_t() == 3
    assert c.max_rotations() == 2


def test_circuit_max_op_counts():
    empty = km.Circuit("")
    assert all(v == 0 for v in empty.max_op_counts().values())

    c = km.Circuit("""
        X q0
        X q0 if b0
        CX q0 q1
        CX q0 q1 if b0
        PUSH_CONDITION if b0
        CCX q0 q1 q2
        CCZ q0 q1 q2 if b1
        Z_POW q0 0.25
        Z_POW q0 0.125 if b1
        POP_CONDITION
        DEBUG_PRINT
        DEBUG_PRINT q0
        DEBUG_PRINT b0 if b1
    """)
    counts = c.max_op_counts()
    assert counts["X"] == 2
    assert counts["CX"] == 2
    assert counts["PUSH_CONDITION"] == 1
    assert counts["POP_CONDITION"] == 1
    assert counts["CCX"] == 1
    assert counts["CCZ"] == 1
    assert counts["Z_POW"] == 2
    assert counts["DEBUG_PRINT"] == 3
    assert counts["CZ"] == 0
    assert counts["HMR"] == 0

    sim = km.Simulator(batch_size=4, count_operations=True, ignore_debug_prints=True)
    assert set(counts.keys()) == set(sim.op_counts().keys())

    def classify_z_pow(angle):
        if angle.denominator <= 2:
            return "Z_POW_CLIFFORD"
        if angle.denominator == 4:
            return "T"
        return f"Z_POW_2^-{angle.denominator.bit_length() - 1}"

    grouped = c.max_op_counts(z_pow_key=classify_z_pow)
    assert "Z_POW" not in grouped
    assert grouped["T"] == 1
    assert grouped["Z_POW_2^-3"] == 1
    assert grouped["X"] == 2

    with pytest.raises(TypeError, match="z_pow_key must return a str"):
        c.max_op_counts(z_pow_key=lambda _: 123)  # type: ignore[arg-type, return-value]
