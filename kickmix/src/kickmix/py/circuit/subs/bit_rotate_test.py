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
from src.kickmix.py.circuit.subs.fuzz_test_util import assert_fuzz_testing_acts_like

import kickmix as km


@pytest.mark.parametrize("n", [0, 1, 2, 8, 64])
def test_fuzz_left_rotate_no_control(n: int):
    builder = km.CircuitBuilder()
    target = builder.create_quantum_register(n, name="target")
    builder.left_rotate(target)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            if n:
                mask = ~(-1 << n)
                target = ((target << 1) | target >> (n - 1)) & mask
        """,
        context={'n': n},
        shots=64,
    )


@pytest.mark.parametrize("n", [0, 1, 2, 8, 64])
def test_fuzz_left_rotate_classical_control(n: int):
    builder = km.CircuitBuilder()
    target = builder.create_quantum_register(n, name="target")
    control = builder.create_classical_register(1, name="control")[0]
    builder.left_rotate(target, control=control)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            if control and n:
                mask = ~(-1 << n)
                target = ((target << 1) | target >> (n - 1)) & mask
        """,
        context={'n': n},
        shots=64,
    )


@pytest.mark.parametrize("n", [0, 1, 2, 8, 64])
def test_fuzz_left_rotate(n: int):
    builder = km.CircuitBuilder()
    target = builder.create_quantum_register(n, name="target")
    control = builder.create_quantum_register(1, name="control")[0]
    builder.left_rotate(target, control=control)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            if control and n:
                mask = ~(-1 << n)
                target = ((target << 1) | target >> (n - 1)) & mask
        """,
        context={'n': n},
        shots=64,
    )


@pytest.mark.parametrize("n", [0, 1, 2, 8, 64])
def test_fuzz_right_rotate(n: int):
    builder = km.CircuitBuilder()
    target = builder.create_quantum_register(n, name="target")
    control = builder.create_quantum_register(1, name="control")[0]
    builder.right_rotate(target, control=control)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            if control and n:
                mask = ~(-1 << n)
                target = ((target >> 1) | target << (n - 1)) & mask
        """,
        context={'n': n},
        shots=64,
    )
