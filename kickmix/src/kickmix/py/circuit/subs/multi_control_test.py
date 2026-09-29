from __future__ import annotations

import random

import pytest
from src.kickmix.py.circuit.subs.fuzz_test_util import assert_fuzz_testing_acts_like

import kickmix as km


def rand_controls(n: int) -> int:
    w = random.randrange(n + 1)
    vals = [1] * w + [0] * (n - w)
    random.shuffle(vals)
    return sum(vals[k] << k for k in range(len(vals)))


@pytest.mark.parametrize("n_controls", [0, 1, 2, 3, 10, 50])
@pytest.mark.parametrize("n_targets", range(4))
def test_fuzz_multi_controlled_x(n_controls: int, n_targets: int):
    builder = km.CircuitBuilder()
    controls = builder.create_quantum_register(n_controls, name="controls")
    targets = builder.create_quantum_register(n_targets, name="targets")
    builder.multi_controlled_x(controls, targets)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            if controls == ~(-1 << c):
                targets ^= ~(-1 << t)
        """,
        shots=64,
        context={'t': n_targets, 'c': n_controls},
        input_sampler=lambda: {
            'controls': rand_controls(n_controls),
            'targets': random.randrange(1 << n_targets),
        },
    )


@pytest.mark.parametrize('b', [False, True])
def test_fuzz_multi_controlled_x_mixed_controls(b: bool):
    builder = km.CircuitBuilder()
    q_controls = builder.create_quantum_register(4, name="q_controls")
    c_controls = builder.create_quantum_register(3, name="c_controls")
    targets = builder.create_quantum_register(1, name="targets")
    builder.multi_controlled_x(q_controls + c_controls + km.array([b]), targets)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            if b and q_controls == 15 and c_controls == 7:
                targets ^= 1
        """,
        shots=64,
        context={'b': b},
        input_sampler=lambda: {
            'q_controls': rand_controls(4),
            'c_controls': rand_controls(3),
            'targets': random.randrange(2),
        },
    )
