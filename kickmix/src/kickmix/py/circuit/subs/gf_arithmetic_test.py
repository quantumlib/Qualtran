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

import random

import pytest
from src.kickmix.py.circuit.subs.fuzz_test_util import assert_fuzz_testing_acts_like

import kickmix as km


def test_gf2_field_basics():
    f8 = km.GF2Field(8)
    assert f8.degree == 8
    assert f8.modulus == 0x11D
    assert km.GF2Field.is_irreducible(f8.modulus)
    assert km.GF2Field.is_irreducible(0x11B)
    assert not km.GF2Field.is_irreducible(0x101)
    assert f8 == km.GF2Field(8, 0x11D)
    assert f8 == km.GF2Field(8, 0x1D)
    assert f8 != km.GF2Field(8, 0x11B)
    assert f8 != km.GF2Field(4)
    assert f8 != "not a field"
    assert hash(f8) == hash(km.GF2Field(8, 0x11D))
    assert repr(f8) == "km.GF2Field(8)"
    assert repr(km.GF2Field(8, 0x11B)) == "km.GF2Field(8, modulus=0x11B)"
    assert str(f8) == "GF(2^8, modulus=0x11D)"

    aes = km.GF2Field(8, 0x11B)
    assert aes.is_element(0)
    assert aes.is_element(255)
    assert not aes.is_element(256)
    assert not aes.is_element(-1)

    for a in [1, 2, 0x53, 0xCA, 255]:
        inv = aes.invert(a)
        assert aes.mul(a, inv) == 1
        assert aes.div(1, a) == inv
        assert aes.pow(a, -1) == inv
        assert aes.pow(a, 0) == 1
        assert aes.pow(a, 2) == aes.square(a)
        assert aes.pow(a, 4) == aes.frobenius(a, 2)
        assert aes.pow(a, 255) == 1
    assert aes.invert(0) == 0
    assert aes.div(0x53, 0) == 0
    assert aes.add(0x53, 0xCA) == (0x53 ^ 0xCA)
    assert aes.mod(0x1FF) == (0xFF ^ 0x1B)


def test_gf2_field_large_degree():
    f128 = km.GF2Field(128)
    assert f128.degree == 128
    a = (1 << 127) | (1 << 65) | 0x12345
    b = (1 << 120) | (1 << 70) | 0xABCDE
    prod = f128.mul(a, b)
    assert f128.div(prod, b) == a
    assert f128.mul(a, f128.invert(a)) == 1
    assert f128.pow(a, (1 << 128) - 1) == 1
    assert f128.pow(a, -2) == f128.invert(f128.square(a))


def test_gf2_field_string_parsing():
    aes_1 = km.GF2Field(modulus="x^8 + x^4 + x^3 + x + 1")
    aes_2 = km.GF2Field(8, "x^8 + x^4 + x^3 + x + 1")
    aes_3 = km.GF2Field(8, "x^4 + x^3 + x + 1")
    assert aes_1 == km.GF2Field(8, 0x11B)
    assert aes_2 == aes_1
    assert aes_3 == aes_1

    assert km.GF2Field(modulus="x**4 + x + 1") == km.GF2Field(4, 0x13)
    assert km.GF2Field(4, "0x19") == km.GF2Field(4, 0x19)
    assert km.GF2Field(4, "0b10011") == km.GF2Field(4, 0x13)

    f100 = km.GF2Field(modulus="x^100 + x^8 + x^7 + x^2 + 1")
    assert f100.degree == 100
    assert f100.modulus == (1 << 100) | (1 << 8) | (1 << 7) | (1 << 2) | 1
    assert f100 == km.GF2Field(100, "x^100 + x^8 + x^7 + x^2 + 1")

    with pytest.raises(
        ValueError,
        match=r"modulus .* \(x\^100 \+ x\^10 \+ x\^2 \+ 1\) of degree 100 is not irreducible over GF\(2\)",
    ):
        km.GF2Field(100, "x^100 + x^10 + x^2 + 1")
    with pytest.raises(
        ValueError,
        match=r"modulus .* \(x\^100 \+ x\^10 \+ x\^2 \+ 1\) of degree 100 is not irreducible over GF\(2\)",
    ):
        km.GF2Field(modulus="x^100 + x^10 + x^2 + 1")
    with pytest.raises(
        ValueError,
        match=r"degree must be an int or None; to construct from a polynomial string, pass it as modulus=",
    ):
        km.GF2Field("x^4 + x + 1")


def test_gf2_field_validation():
    with pytest.raises(
        ValueError, match=r"degree must be between 1 and 512.*For example, use GF2Field\(8\)"
    ):
        km.GF2Field(0)
    with pytest.raises(
        ValueError, match=r"degree must be between 1 and 512.*For example, use GF2Field\(8\)"
    ):
        km.GF2Field(513)
    with pytest.raises(
        ValueError,
        match=(
            r"modulus 0x11 \(x\^4 \+ 1\) of degree 4 is not irreducible over GF\(2\): "
            r"divisible by 0x3 \(x \+ 1\)\. For example, use GF2Field\(4\) for the default modulus, "
            r"or GF2Field\(4, 0x13\) for 0x13 \(x\^4 \+ x \+ 1\)\."
        ),
    ):
        km.GF2Field(4, 0b10001)
    with pytest.raises(
        ValueError,
        match=(
            r"modulus 0x1B \(x\^4 \+ x\^3 \+ x \+ 1\) \(from input 0xB with implied x\^4 bit\) "
            r"of degree 4 is not irreducible over GF\(2\).*use GF2Field\(3, 0xB\) if you intended GF\(2\^3\)"
        ),
    ):
        km.GF2Field(4, 0xB)
    with pytest.raises(
        ValueError,
        match=(
            r"modulus 0x23 \(x\^5 \+ x \+ 1\) has degree 5, expected degree 4 for GF\(2\^4\)\. "
            r"For example, use GF2Field\(4\) for the default modulus, or GF2Field\(4, 0x13\)"
        ),
    ):
        km.GF2Field(4, 0b100011)
    with pytest.raises(
        ValueError,
        match=(
            r"modulus 0x25 \(x\^5 \+ x\^2 \+ 1\) has degree 5, expected degree 4 for GF\(2\^4\).*"
            r"use GF2Field\(5, 0x25\) if you intended GF\(2\^5\)"
        ),
    ):
        km.GF2Field(4, 0x25)
    with pytest.raises(ValueError, match="non-negative"):
        km.GF2Field(4, -1)
    # Huge ints used to escape as an opaque RuntimeError from the C++ size_t cast.
    with pytest.raises(
        ValueError, match=r"degree must be between 1 and 512, got 1180591620717411303424"
    ):
        km.GF2Field(2**70)
    with pytest.raises(ValueError, match="requires a degree, a modulus, or a galois"):
        km.GF2Field()


def _clmul(a: int, b: int) -> int:
    """Carry-less (GF(2)[x]) product of two polynomials encoded as ints."""
    result = 0
    while b:
        if b & 1:
            result ^= a
        a <<= 1
        b >>= 1
    return result


def test_gf2_field_reducible_modulus_reports_irreducible_factor():
    # (x^3 + x + 1)(x^3 + x^2 + 1): used to be reported as "divisible by" itself.
    assert _clmul(0xB, 0xD) == 0x7F
    with pytest.raises(ValueError, match=r"divisible by 0x(B|D) "):
        km.GF2Field(6, 0x7F)
    # Product of all three irreducible quartics.
    with pytest.raises(ValueError, match=r"divisible by 0x(13|19|1F) "):
        km.GF2Field(12, _clmul(_clmul(0x13, 0x19), 0x1F))


def test_gf2_field_string_rejects_repeated_terms():
    # Repeated terms cancel over GF(2); this used to silently create GF(2^1).
    with pytest.raises(ValueError, match="appears more than once"):
        km.GF2Field(modulus="x^4 + x^4 + x + 1")


def test_gf2_field_from_galois():
    assert km.GF2Field(12).modulus == 0x1009

    class MockPoly:
        def __init__(self, char: int, deg: int, val: int):
            self.field = type("F", (), {"characteristic": char})()
            self.degree = deg
            self.coeffs = [1]
            self._val = val

        def __int__(self) -> int:
            return self._val

    class MockGaloisField:
        characteristic = 2
        degree = 8
        irreducible_poly = MockPoly(2, 8, 0x11B)

    aes = km.GF2Field(8, 0x11B)
    assert km.GF2Field(MockGaloisField) == aes
    assert km.GF2Field.from_galois(MockGaloisField) == aes
    assert km.GF2Field(modulus=MockGaloisField.irreducible_poly) == aes
    assert km.GF2Field(8, MockGaloisField.irreducible_poly) == aes

    builder = km.CircuitBuilder()
    a = builder.create_quantum_register(8, name="a")
    b = builder.create_quantum_register(8, name="b")
    c = builder.create_quantum_register(8, name="c")
    builder.ixor_gf2_mul(target=c, lhs=a, rhs=b, field=MockGaloisField)

    class MockGF3Field:
        characteristic = 3
        degree = 2
        irreducible_poly = MockPoly(3, 2, 17)

    with pytest.raises(ValueError, match="characteristic 2"):
        km.GF2Field(MockGF3Field)
    with pytest.raises(ValueError, match="characteristic 2"):
        km.GF2Field.from_galois(MockGF3Field)
    with pytest.raises(ValueError, match="over GF\\(2\\)"):
        km.GF2Field(modulus=MockGF3Field.irreducible_poly)
    with pytest.raises(ValueError, match="modulus must be None"):
        km.GF2Field(MockGaloisField, 0x11B)
    with pytest.raises(ValueError, match="galois_field must be"):
        km.GF2Field.from_galois(8)

    try:
        import galois
    except ImportError:
        return

    gf16 = galois.GF(2**4, irreducible_poly="x^4 + x^3 + 1")
    assert km.GF2Field(gf16) == km.GF2Field(4, 0x19)
    assert km.GF2Field.from_galois(gf16) == km.GF2Field(4, 0x19)
    assert km.GF2Field(modulus=gf16.irreducible_poly) == km.GF2Field(4, 0x19)
    with pytest.raises(ValueError, match="characteristic 2"):
        km.GF2Field(galois.GF(3**2))


def test_gf2_field_primitive_element():
    assert km.GF2Field(1).primitive_element == 1
    assert km.GF2Field(1).is_primitive_element(1)
    assert not km.GF2Field(1).is_primitive_element(0)
    assert not km.GF2Field(1).is_primitive_element(2)

    for m in range(2, 12):
        f = km.GF2Field(m)
        assert f.primitive_element == 2
        assert f.is_primitive_element(2)

    # For m = 12, the default trinomial x^12 + x^3 + 1 (0x1009) has primitive_element = 3 (x + 1).
    f12 = km.GF2Field(12)
    assert f12.modulus == 0x1009
    assert f12.primitive_element == 3
    assert not f12.is_primitive_element(2)
    assert f12.is_primitive_element(3)
    assert len({f12.pow(f12.primitive_element, k) for k in range((1 << 12) - 1)}) == (1 << 12) - 1

    # Also works for m = 14 and m = 16 where x = 2 is not primitive.
    for m in (14, 16):
        f = km.GF2Field(m)
        assert f.is_primitive_element(f.primitive_element)

    # AES polynomial 0x11B: x (2) has order 51, while x + 1 (3) has order 255.
    aes = km.GF2Field(8, 0x11B)
    assert aes.primitive_element == 3
    assert not aes.is_primitive_element(2)
    assert aes.is_primitive_element(3)
    assert len({aes.pow(aes.primitive_element, k) for k in range(255)}) == 255

    # Custom primitive element: 0x9 (x^3 + 1) has order 15 in GF(2^4, modulus=0x13).
    f4_custom = km.GF2Field(4, 0x13, primitive_element=0x9)
    assert f4_custom.primitive_element == 0x9
    assert f4_custom != km.GF2Field(4)
    assert repr(f4_custom) == "km.GF2Field(4, primitive_element=0x9)"

    with pytest.raises(ValueError, match="not a primitive element"):
        km.GF2Field(8, 0x11B, primitive_element=2)
    with pytest.raises(ValueError, match="not a non-zero element"):
        km.GF2Field(8, 0x11B, primitive_element=0)


@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_gf2_iadd_fuzz(m: int):
    builder = km.CircuitBuilder()
    target = builder.create_quantum_register(m, name="target")
    offset = builder.create_quantum_register(m, name="offset")
    builder.gf2_iadd(target=target, offset=offset)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            target ^= offset
        """,
        shots=64,
    )


def test_gf2_iadd_controlled_fuzz():
    m = 5
    builder = km.CircuitBuilder()
    ctrl = builder.create_quantum_register(1, name="ctrl")
    target = builder.create_quantum_register(m, name="target")
    offset = builder.create_quantum_register(m, name="offset")
    builder.gf2_iadd(target=target, offset=offset, control=ctrl[0])
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            if ctrl:
                target ^= offset
        """,
        shots=64,
    )


@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_gf2_iadd_classical_int_fuzz(m: int):
    constant = random.randrange(1 << m)
    builder = km.CircuitBuilder()
    target = builder.create_quantum_register(m, name="target")
    builder.gf2_iadd(target=target, offset=constant)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            target ^= constant
        """,
        shots=64,
        context={"constant": constant},
    )


@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_gf2_iadd_classical_int_controlled_fuzz(m: int):
    constant = random.randrange(1 << m)
    builder = km.CircuitBuilder()
    ctrl = builder.create_quantum_register(1, name="ctrl")
    target = builder.create_quantum_register(m, name="target")
    builder.gf2_iadd(target=target, offset=constant, control=ctrl[0])
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            if ctrl:
                target ^= constant
        """,
        shots=64,
        context={"constant": constant},
    )


@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_gf2_iadd_classical_register_fuzz(m: int):
    builder = km.CircuitBuilder()
    target = builder.create_quantum_register(m, name="target")
    offset = builder.create_classical_register(m, name="offset")
    builder.gf2_iadd(target=target, offset=offset)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            target ^= offset
        """,
        shots=64,
    )


@pytest.mark.parametrize("m", [1, 2, 3, 5, 8])
def test_gf2_imul_and_idiv_fuzz(m: int):
    field = km.GF2Field(m)
    constant = random.randrange(1, 1 << m)

    builder = km.CircuitBuilder()
    target = builder.create_quantum_register(m, name="target")
    builder.gf2_imul(constant, target=target, field=field)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            target = field.mul(target, constant)
        """,
        shots=64,
        context={"field": field, "constant": constant},
    )

    builder2 = km.CircuitBuilder()
    target2 = builder2.create_quantum_register(m, name="target")
    builder2.gf2_idiv(constant, target=target2, field=field)
    circuit2 = builder2.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit2,
        """
            target = field.div(target, constant)
        """,
        shots=64,
        context={"field": field, "constant": constant},
    )


@pytest.mark.parametrize("m", [1, 2, 3, 4, 6, 8])
def test_gf2_mul_and_del_gf2_mul_fuzz(m: int):
    field = km.GF2Field(m)

    builder = km.CircuitBuilder()
    lhs = builder.create_quantum_register(m, name="lhs")
    rhs = builder.create_quantum_register(m, name="rhs")
    target = builder.create_quantum_register(m, name="target")
    builder.ixor_gf2_mul(target=target, lhs=lhs, rhs=rhs, field=field)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            target ^= field.mul(lhs, rhs)
        """,
        shots=64,
        context={"field": field},
    )

    # Test init_gf2_mul resets target to 0 before multiplying
    builder_init = km.CircuitBuilder()
    lhs_i = builder_init.create_quantum_register(m, name="lhs")
    rhs_i = builder_init.create_quantum_register(m, name="rhs")
    target_i = builder_init.create_quantum_register(m, name="target")
    builder_init.init_gf2_mul(target=target_i, lhs=lhs_i, rhs=rhs_i, field=field)
    circuit_init = builder_init.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit_init,
        """
            target = field.mul(lhs, rhs)
        """,
        shots=64,
        context={"field": field},
        input_sampler=lambda: {
            "lhs": random.randrange(1 << m),
            "rhs": random.randrange(1 << m),
            "target": 0,
        },
    )

    builder_del = km.CircuitBuilder()
    lhs_d = builder_del.create_quantum_register(m, name="lhs")
    rhs_d = builder_del.create_quantum_register(m, name="rhs")
    target_d = builder_del.create_quantum_register(m, name="target")
    builder_del.del_gf2_mul(target=target_d, lhs=lhs_d, rhs=rhs_d, field=field)
    circuit_del = builder_del.finish_circuit()
    assert circuit_del.max_magic() == 0

    assert_fuzz_testing_acts_like(
        circuit_del,
        """
            target = 0
        """,
        shots=64,
        input_sampler=lambda: {
            "lhs": (a := random.randrange(1 << m)),
            "rhs": (b := random.randrange(1 << m)),
            "target": field.mul(a, b),
        },
    )


def test_gf2_mul_controlled_fuzz():
    m = 4
    field = km.GF2Field(m, 0b11001)

    builder = km.CircuitBuilder()
    ctrl = builder.create_quantum_register(1, name="ctrl")
    lhs = builder.create_quantum_register(m, name="lhs")
    rhs = builder.create_quantum_register(m, name="rhs")
    target = builder.create_quantum_register(m, name="target")
    builder.ixor_gf2_mul(target=target, lhs=lhs, rhs=rhs, field=field, control=ctrl[0])
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            if ctrl:
                target ^= field.mul(lhs, rhs)
        """,
        shots=64,
        context={"field": field},
    )


def test_init_gf2_mul_resets_target():
    field = km.GF2Field(4)
    builder = km.CircuitBuilder()
    lhs = builder.create_quantum_register(4, name="lhs")
    rhs = builder.create_quantum_register(4, name="rhs")
    target = builder.create_quantum_register(4, name="target")
    # Set target to non-zero before init_gf2_mul
    builder.x(target)
    builder.init_gf2_mul(target=target, lhs=lhs, rhs=rhs, field=field)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            target = field.mul(lhs, rhs)
        """,
        shots=64,
        context={"field": field},
        ignore_phase=True,
    )


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 7])
def test_init_and_del_gf2_inverse_fuzz(m: int):
    field = km.GF2Field(m)

    builder = km.CircuitBuilder()
    inp = builder.create_quantum_register(m, name="inp")
    target = builder.create_quantum_register(m, name="target")
    builder.init_gf2_inverse(target=target, input=inp, field=field)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            target = field.invert(inp)
        """,
        shots=64,
        context={"field": field},
        input_sampler=lambda: {"inp": random.randrange(1 << m), "target": 0},
    )

    builder_del = km.CircuitBuilder()
    inp_d = builder_del.create_quantum_register(m, name="inp")
    target_d = builder_del.create_quantum_register(m, name="target")
    builder_del.del_gf2_inverse(target=target_d, input=inp_d, field=field)
    circuit_del = builder_del.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit_del,
        """
            target = 0
        """,
        shots=64,
        input_sampler=lambda: {"inp": (a := random.randrange(1 << m)), "target": field.invert(a)},
    )


@pytest.mark.parametrize("m", [1, 2, 3, 5, 6])
def test_init_and_del_gf2_inverse_with_scaffold_fuzz(m: int):
    field = km.GF2Field(m)

    builder = km.CircuitBuilder()
    inp = builder.create_quantum_register(m, name="inp")
    out = builder.create_quantum_register(m, name="out")
    inv, scaffold = builder.init_gf2_inverse_with_scaffold(inp, field=field)
    builder.gf2_iadd(inv, target=out)
    builder.del_gf2_inverse_with_scaffold(inp, target=inv, scaffold=scaffold, field=field)
    builder.free(scaffold)
    builder.free(inv)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            out ^= field.invert(inp)
        """,
        shots=64,
        context={"field": field},
    )

    # Verify del_gf2_inverse_with_scaffold uses 0 Toffolis.
    del_only = km.CircuitBuilder()
    q_inp = del_only.create_quantum_register(m, name="inp")
    q_inv = del_only.create_quantum_register(m, name="inv")
    anc = del_only.alloc_qubits(len(scaffold))
    del_only.del_gf2_inverse_with_scaffold(q_inp, target=q_inv, scaffold=anc, field=field)
    assert del_only.finish_circuit().max_magic() == 0


@pytest.mark.parametrize("m", [1, 2, 3, 4, 6])
def test_init_and_del_gf2_div_with_scaffold_fuzz(m: int):
    field = km.GF2Field(m)

    builder = km.CircuitBuilder()
    lhs = builder.create_quantum_register(m, name="lhs")
    rhs = builder.create_quantum_register(m, name="rhs")
    out = builder.create_quantum_register(m, name="out")
    target, scaffold = builder.init_gf2_div_with_scaffold(lhs, rhs, field=field)
    builder.gf2_iadd(target, target=out)
    builder.del_gf2_div_with_scaffold(lhs, rhs, target=target, scaffold=scaffold, field=field)
    builder.free(scaffold)
    builder.free(target)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            out ^= field.div(lhs, rhs)
        """,
        shots=64,
        context={"field": field},
    )

    # Verify del_gf2_div_with_scaffold uses 0 Toffolis.
    del_only = km.CircuitBuilder()
    d_lhs = del_only.create_quantum_register(m, name="lhs")
    d_rhs = del_only.create_quantum_register(m, name="rhs")
    d_target = del_only.create_quantum_register(m, name="target")
    d_anc = del_only.alloc_qubits(len(scaffold))
    del_only.del_gf2_div_with_scaffold(d_lhs, d_rhs, target=d_target, scaffold=d_anc, field=field)
    assert del_only.finish_circuit().max_magic() == 0


def test_field_does_not_have_inverse_chain_size():
    field = km.GF2Field(4)
    assert not hasattr(field, "inverse_chain_size")


@pytest.mark.parametrize("m", [1, 2, 3, 4, 6])
def test_init_gf2_div_with_scaffold_keeps_inverse(m: int):
    field = km.GF2Field(m)

    builder = km.CircuitBuilder()
    lhs = builder.create_quantum_register(m, name="lhs")
    rhs = builder.create_quantum_register(m, name="rhs")
    target, scaffold = builder.init_gf2_div_with_scaffold(lhs, rhs, field=field)
    circuit = builder.finish_circuit()

    sim = km.Simulator(batch_size=16)
    sim.use_same_registers_as(circuit)
    inputs = []
    for slot in range(16):
        a = random.randrange(1 << m)
        b = random.randrange(1 << m)
        inputs.append((a, b))
        sim.write_within_shot("lhs", slot, a)
        sim.write_within_shot("rhs", slot, b)
    sim.do(circuit)
    for slot, (a, b) in enumerate(inputs):
        actual_target = sim.read_within_shot(target, slot, out=int)
        assert actual_target == field.div(a, b)
        actual_inv = sim.read_within_shot(scaffold[:m], slot, out=int)
        assert actual_inv == field.invert(b)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 6])
def test_gf2_div_and_del_gf2_div_fuzz(m: int):
    field = km.GF2Field(m)

    builder = km.CircuitBuilder()
    lhs = builder.create_quantum_register(m, name="lhs")
    rhs = builder.create_quantum_register(m, name="rhs")
    target = builder.create_quantum_register(m, name="target")
    builder.ixor_gf2_div(target=target, lhs=lhs, rhs=rhs, field=field)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            target ^= field.div(lhs, rhs)
        """,
        shots=64,
        context={"field": field},
    )

    # Test init_gf2_div resets target to 0 before dividing
    builder_init = km.CircuitBuilder()
    lhs_i = builder_init.create_quantum_register(m, name="lhs")
    rhs_i = builder_init.create_quantum_register(m, name="rhs")
    target_i = builder_init.create_quantum_register(m, name="target")
    builder_init.init_gf2_div(target=target_i, lhs=lhs_i, rhs=rhs_i, field=field)
    circuit_init = builder_init.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit_init,
        """
            target = field.div(lhs, rhs)
        """,
        shots=64,
        context={"field": field},
        input_sampler=lambda: {
            "lhs": random.randrange(1 << m),
            "rhs": random.randrange(1 << m),
            "target": 0,
        },
    )

    builder_del = km.CircuitBuilder()
    lhs_d = builder_del.create_quantum_register(m, name="lhs")
    rhs_d = builder_del.create_quantum_register(m, name="rhs")
    target_d = builder_del.create_quantum_register(m, name="target")
    builder_del.del_gf2_div(target=target_d, lhs=lhs_d, rhs=rhs_d, field=field)
    circuit_del = builder_del.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit_del,
        """
            target = 0
        """,
        shots=64,
        input_sampler=lambda: {
            "lhs": (a := random.randrange(1 << m)),
            "rhs": (b := random.randrange(1 << m)),
            "target": field.div(a, b),
        },
    )


def test_gf2_div_self_divisor():
    # gen_gf_div allows lhs and rhs to be the same register.
    field = km.GF2Field(4)
    builder = km.CircuitBuilder()
    a = builder.create_quantum_register(4, name="a")
    target = builder.create_quantum_register(4, name="target")
    builder.ixor_gf2_div(target=target, lhs=a, rhs=a, field=field)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            target ^= (1 if a != 0 else 0)
        """,
        shots=64,
    )


def test_init_gf2_div_resets_target():
    field = km.GF2Field(4)
    builder = km.CircuitBuilder()
    lhs = builder.create_quantum_register(4, name="lhs")
    rhs = builder.create_quantum_register(4, name="rhs")
    target = builder.create_quantum_register(4, name="target")
    # Set target to non-zero before init_gf2_div
    builder.x(target)
    builder.init_gf2_div(target=target, lhs=lhs, rhs=rhs, field=field)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            target = field.div(lhs, rhs)
        """,
        shots=64,
        context={"field": field},
        ignore_phase=True,
    )


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_gf2_phase_by_product_exhaustive(m: int):
    field = km.GF2Field(m)

    builder = km.CircuitBuilder()
    lhs = builder.create_quantum_register(m, name="lhs")
    rhs = builder.create_quantum_register(m, name="rhs")
    mask = builder.create_classical_register(m, name="mask")
    builder.gf2_phase_by_product(mask=mask, lhs=lhs, rhs=rhs, field=field)
    circuit = builder.finish_circuit()
    assert circuit.max_magic() == 0

    size = 1 << m
    cases = [(a, b, c) for a in range(size) for b in range(size) for c in range(size)]
    sim = km.Simulator(batch_size=64)
    sim.use_same_registers_as(circuit)
    for start in range(0, len(cases), 64):
        batch = cases[start : start + 64]
        sim.clear_for_shot()
        for slot, (a, b, c) in enumerate(batch):
            sim.write_within_shot("lhs", slot, a)
            sim.write_within_shot("rhs", slot, b)
            sim.write_within_shot("mask", slot, c)
        sim.do(circuit)
        flipped = sim.read_phase_flipped_across_shots(out=int)
        for slot, (a, b, c) in enumerate(batch):
            expected = bin(field.mul(a, b) & c).count("1") % 2
            assert bool((flipped >> slot) & 1) == bool(expected), f"{a} * {b} & {c}"


def test_gf2_error_handling():
    builder = km.CircuitBuilder()
    a = builder.create_quantum_register(4, name="a")
    b = builder.create_quantum_register(4, name="b")
    c = builder.create_quantum_register(5, name="c")

    with pytest.raises(ValueError, match="size"):
        builder.gf2_iadd(target=a, offset=c)
    with pytest.raises(ValueError, match="range"):
        builder.gf2_iadd(target=a, offset=-1)
    with pytest.raises(ValueError, match="range"):
        builder.gf2_iadd(target=a, offset=16)
    with pytest.raises(ValueError, match="multiplication by zero"):
        builder.gf2_imul(0, target=a)
    with pytest.raises(ValueError, match="division by zero"):
        builder.gf2_idiv(0, target=a)
    # 16 is not an element of GF(2^4); it used to be silently reduced to 16 mod 0x13 = 3.
    with pytest.raises(ValueError, match=r"factor 16 is not an element of GF\(2\^4\)"):
        builder.gf2_imul(16, target=a)
    with pytest.raises(ValueError, match=r"divisor 19 is not an element of GF\(2\^4\)"):
        builder.gf2_idiv(0x13, target=a)
    with pytest.raises(NotImplementedError, match="classical constant"):
        builder.gf2_imul(b, target=a)
    with pytest.raises(NotImplementedError, match="classical constant"):
        builder.gf2_idiv(b, target=a)
    with pytest.raises(NotImplementedError, match="classical constant"):
        builder.gf2_imul(True, target=a)
    with pytest.raises(NotImplementedError, match="classical constant"):
        builder.gf2_idiv(False, target=a)
    with pytest.raises(ValueError, match="field.degree"):
        builder.ixor_gf2_mul(target=a, lhs=a, rhs=b, field=km.GF2Field(5))
    with pytest.raises(ValueError, match=r"field must be None, a km\.GF2Field, or a galois\.GF"):
        builder.ixor_gf2_mul(target=a, lhs=a, rhs=b, field=0x13)
    with pytest.raises(ValueError, match=r"field must be None, a km\.GF2Field, or a galois\.GF"):
        builder.ixor_gf2_mul(target=a, lhs=a, rhs=b, field="x^4 + x + 1")
    with pytest.raises(ValueError, match="clean qubits"):
        builder.set_max_qubits(builder.num_allocated_qubits)
        builder.ixor_gf2_div(target=a, lhs=b, rhs=b)
    with pytest.raises(ValueError, match="collision"):
        builder.gf2_iadd(target=a, offset=a)
        builder.finish_circuit()


@pytest.mark.parametrize(
    "call",
    [
        lambda b, a, c, short: b.init_gf2_mul(a, c, target=short),
        lambda b, a, c, short: b.init_gf2_mul(a, short, target=c),
        lambda b, a, c, short: b.init_gf2_div(a, c, target=short),
        lambda b, a, c, short: b.init_gf2_div(a, short, target=c),
        lambda b, a, c, short: b.init_gf2_div_with_scaffold(a, c, target=short),
        lambda b, a, c, short: b.init_gf2_inverse(a, target=short),
        lambda b, a, c, short: b.init_gf2_inverse_with_scaffold(a, target=short),
    ],
)
def test_rejected_init_calls_leave_builder_untouched(call):
    # These calls used to reset the target (and allocate workspace) before the
    # size check fired, so catching the error left a corrupted circuit behind.
    builder = km.CircuitBuilder()
    a = builder.create_quantum_register(4, name="a")
    c = builder.create_quantum_register(4, name="c")
    short = builder.create_quantum_register(3, name="short")
    allocated = builder.num_allocated_qubits
    with pytest.raises(ValueError, match="size|qubits"):
        call(builder, a, c, short)
    assert builder.num_allocated_qubits == allocated
    assert len(builder.finish_circuit()) == 0
