from __future__ import annotations

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
