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

import xml.etree.ElementTree as ET

import pytest

import kickmix as km


def test_builder_create_quantum_register():
    builder = km.CircuitBuilder()
    reg = builder.create_quantum_register(5, name="test")
    assert reg == km.array([km.q(k) for k in range(5)])


def test_builder_create_classical_register():
    builder = km.CircuitBuilder()
    reg = builder.create_classical_register(5, name="test")
    assert reg == km.array([km.b(k) for k in range(5)])


def test_builder_broadcast_swap():
    builder = km.CircuitBuilder()
    a = km.array([km.q(k) for k in range(5)])
    b = km.array([km.q(k) for k in range(5, 10)])
    builder.swap(a, b)

    assert builder.finish_circuit() == km.Circuit("""
        SWAP q0 q5
        SWAP q1 q6
        SWAP q2 q7
        SWAP q3 q8
        SWAP q4 q9
    """)


def test_z_pow():
    builder = km.CircuitBuilder()
    builder.z_pow(km.q(0), 0.1)
    builder.z_pow(km.q(1), "0.1")
    builder.z_pow([km.q(2), km.q(3)], 2.125)
    builder.z_pow(km.array([km.q(4)]), -0.125)

    with pytest.raises(ValueError, match="qubit ids"):
        builder.z_pow(km.xbool(True), 0.5)
    with pytest.raises(ValueError, match="Fraction"):
        builder.z_pow(km.q(0), "test")
    with pytest.raises(TypeError, match="string or"):
        builder.z_pow(km.q(0), object())

    assert builder.finish_circuit() == km.Circuit("""
        Z_POW q0 0.1000000000000000055511151231257827021181583404541015625
        Z_POW q1 0.1000000000000000000000000000000000000011754943508222875079687365372222456778186655567720875215087517062784172594547271728515625
        Z_POW q2 0.125
        Z_POW q3 0.125
        Z_POW q4 1.875
    """)


def test_builder_mark():
    builder = km.CircuitBuilder()
    q = builder.create_quantum_register(4, name="q")

    reused = builder.mark("reused")
    assert repr(reused) == "<km._CircuitBuilderMark name='reused'>"
    for _ in range(2):
        with reused:
            builder.ccx(q[0], q[1], q[2])

    @builder.mark
    def bound_fn(x: int, *, y: int = 1) -> int:
        """Bound docstring."""
        with builder.mark(name="inner <0 & 1>"):
            builder.ccx(q[1], q[2], q[3])
        return x + y

    class Gadget:
        def __init__(self, cb: km.CircuitBuilder):
            self.cb = cb

        @km.CircuitBuilder.mark("gadget_step")
        def step(self) -> None:
            self.cb.ccx(q[0], q[1], q[2])

    @km.CircuitBuilder.mark
    def unbound_fn(cb: km.CircuitBuilder) -> None:
        cb.ccx(q[0], q[1], q[2])

    assert bound_fn.__name__ == "bound_fn"
    assert bound_fn.__doc__ == "Bound docstring."
    assert bound_fn(3, y=4) == 7
    Gadget(builder).step()
    unbound_fn(builder)

    svg = builder.flame_chart_svg()
    ET.fromstring(svg)
    assert "entire circuit" not in svg
    for expected in [
        ">reused (x2)</text>",
        ">bound_fn</text>",
        ">inner &lt;0 &amp; 1&gt;</text>",
        ">gadget_step</text>",
        ">unbound_fn</text>",
        ">4 qubits, 5 toffolis</text>",
    ]:
        assert expected in svg


def test_builder_mark_errors():
    builder = km.CircuitBuilder()
    q = builder.create_quantum_register(3, name="q")

    with pytest.raises(RuntimeError, match="boom"):
        with builder.mark("unwound"):
            builder.ccx(q[0], q[1], q[2])
            raise RuntimeError("boom")
    assert ">unwound</text>" in builder.flame_chart_svg()

    with pytest.raises(ValueError, match="mark name must be provided"):
        with builder.mark():
            pass
    with pytest.raises(ValueError, match="unbound CircuitBuilder.mark"):
        with km.CircuitBuilder.mark("unbound"):
            pass
    with pytest.raises(ValueError, match="Could not find a CircuitBuilder"):
        km.CircuitBuilder.mark(lambda x: x)(5)

    m1, m2 = builder.mark("m1"), builder.mark("m2")
    m1.__enter__()
    m2.__enter__()
    with pytest.raises(ValueError, match="LIFO"):
        m1.__exit__(None, None, None)
    m2.__exit__(None, None, None)
    m1.__exit__(None, None, None)

    for bad_call in [
        lambda: builder.mark(123),
        lambda: builder.mark("a", "b"),
        lambda: builder.mark("a", name="b"),
        lambda: builder.mark(invalid_kw="a"),
    ]:
        with pytest.raises(TypeError):
            bad_call()
