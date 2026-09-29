from __future__ import annotations

import random

import kickmix as km
import pytest

from src.kickmix.py.circuit.subs.fuzz_test_util import assert_fuzz_testing_acts_like


@pytest.mark.parametrize("n", [0, 1, 2, 8, 64, 256])
def test_fuzz_flip_if_lt(n: int):
    builder = km.CircuitBuilder()
    lhs = builder.create_quantum_register(n, name="lhs")
    rhs = builder.create_classical_register(n, name="rhs")
    target = builder.create_quantum_register(1, name="target")[0]
    control = builder.create_quantum_register(1, name="control")[0]
    or_equal = builder.create_quantum_register(1, name="or_equal")[0]
    builder.flip_if_less_than(
        lhs=lhs,
        rhs=rhs,
        target=target,
        control=control,
        or_equal=or_equal,
    )
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            if control:
                target ^= lhs < rhs + or_equal
        """,
        shots=64,
        input_sampler=lambda: {
            'lhs': (e := random.randrange(1 << n)),
            'target': random.randrange(2),
            'or_equal': random.randrange(2),
            'control': random.randrange(2),
            # Prefer to test near misses:
            'rhs': (e + random.randrange(-10, 10)) % (1 << n) if random.randrange(2) else random.randrange(1 << n),
        })


@pytest.mark.parametrize("n", [0, 1, 2, 8, 64, 256])
def test_fuzz_flip_if_lt_phase_target(n: int):
    builder = km.CircuitBuilder()
    lhs = builder.create_quantum_register(n, name="lhs")
    rhs = builder.create_classical_register(n, name="rhs")
    control = builder.create_quantum_register(1, name="control")[0]
    or_equal = builder.create_quantum_register(1, name="or_equal")[0]
    builder.flip_if_less_than(
        lhs=lhs,
        rhs=rhs,
        target="|->",
        control=control,
        or_equal=or_equal,
    )
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            if control:
                Z(lhs < rhs + or_equal)
        """,
        shots=64,
        input_sampler=lambda: {
            'lhs': (e := random.randrange(1 << n)),
            'or_equal': random.randrange(2),
            'control': random.randrange(2),
            # Prefer to test near misses:
            'rhs': (e + random.randrange(-10, 10)) % (1 << n) if random.randrange(2) else random.randrange(1 << n),
        })


@pytest.mark.parametrize("n", [0, 1, 2, 8, 64, 256])
def test_fuzz_flip_if_gt(n: int):
    builder = km.CircuitBuilder()
    lhs = builder.create_quantum_register(n, name="lhs")
    rhs = builder.create_classical_register(n, name="rhs")
    target = builder.create_quantum_register(1, name="target")[0]
    control = builder.create_quantum_register(1, name="control")[0]
    or_equal = builder.create_quantum_register(1, name="or_equal")[0]
    builder.flip_if_greater_than(
        lhs=lhs,
        rhs=rhs,
        target=target,
        control=control,
        or_equal=or_equal,
    )
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            if control:
                target ^= lhs > rhs - or_equal
        """,
        shots=64,
        input_sampler=lambda: {
            'lhs': (e := random.randrange(1 << n)),
            'target': random.randrange(2),
            'or_equal': random.randrange(2),
            'control': random.randrange(2),
            # Prefer to test near misses:
            'rhs': (e + random.randrange(-10, 10)) % (1 << n) if random.randrange(2) else random.randrange(1 << n),
        })


@pytest.mark.parametrize("n", [0, 1, 2, 8, 64, 256])
def test_fuzz_flip_if_eq(n: int):
    builder = km.CircuitBuilder()
    lhs = builder.create_quantum_register(n, name="lhs")
    rhs = builder.create_classical_register(n, name="rhs")
    target = builder.create_quantum_register(1, name="target")[0]
    control = builder.create_quantum_register(1, name="control")[0]
    builder.free(builder.alloc_qubits(n))
    builder.flip_if_equal(
        lhs=lhs,
        rhs=rhs,
        target=target,
        control=control,
    )
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            if control:
                target ^= lhs == rhs
        """,
        shots=64,
        input_sampler=lambda: {
            'lhs': (e := random.randrange(1 << n)),
            'target': random.randrange(2),
            'control': random.randrange(2),
            # Prefer to test near misses:
            'rhs': (e + random.randrange(-10, 10)) % (1 << n) if random.randrange(2) else random.randrange(1 << n),
        })


@pytest.mark.parametrize("n", [0, 1, 2, 8, 64, 70, 256])
def test_fuzz_flip_if_cmp_with_int_rhs(n: int):
    rhs_val = random.randrange(1 << n)
    for op_name, expr in [
        ("flip_if_less_than", "lhs < rhs + or_equal"),
        ("flip_if_greater_than", "lhs > rhs - or_equal"),
        ("flip_if_equal", "lhs == rhs"),
    ]:
        builder = km.CircuitBuilder()
        lhs = builder.create_quantum_register(n, name="lhs")
        target = builder.create_quantum_register(1, name="target")[0]
        control = builder.create_quantum_register(1, name="control")[0]
        builder.free(builder.alloc_qubits(n))
        kwargs = dict(lhs=lhs, rhs=rhs_val, target=target, control=control)
        if op_name != "flip_if_equal":
            kwargs["or_equal"] = builder.create_quantum_register(1, name="or_equal")[0]
            kwargs["btol"] = float("inf")
        getattr(builder, op_name)(**kwargs)
        circuit = builder.finish_circuit()

        assert_fuzz_testing_acts_like(
            circuit,
            f"""
                if control:
                    target ^= {expr}
            """,
            shots=32,
            context={"rhs": rhs_val},
            input_sampler=lambda: {
                "lhs": (rhs_val + random.randrange(-5, 6)) % (1 << n)
                if random.randrange(2)
                else random.randrange(1 << n),
                "target": random.randrange(2),
                "control": random.randrange(2),
                **({"or_equal": random.randrange(2)} if op_name != "flip_if_equal" else {}),
            },
        )


def test_flip_if_cmp_rejects_out_of_range_int_rhs():
    builder = km.CircuitBuilder()
    lhs = builder.create_quantum_register(4, name="lhs")
    target = builder.create_quantum_register(1, name="target")[0]
    with pytest.raises(ValueError, match="range"):
        builder.flip_if_less_than(lhs=lhs, rhs=-1, target=target)
    with pytest.raises(ValueError, match="range"):
        builder.flip_if_equal(lhs=lhs, rhs=16, target=target)


def test_flip_if_cmp_positional_args():
    builder = km.CircuitBuilder()
    lhs = builder.create_quantum_register(4, name="lhs")
    target = builder.create_quantum_register(1, name="target")[0]
    # Positional lhs, rhs with keyword target should work:
    builder.flip_if_equal(lhs, 5, target=target)
    builder.flip_if_less_than(lhs, 3, target=target)
    builder.flip_if_greater_than(lhs, 2, target=target)
    # Target cannot be passed positionally:
    with pytest.raises(TypeError):
        builder.flip_if_equal(lhs, 5, target)
    with pytest.raises(TypeError):
        builder.flip_if_less_than(lhs, 3, target)
    with pytest.raises(TypeError):
        builder.flip_if_greater_than(lhs, 2, target)


