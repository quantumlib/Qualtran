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
from fuzz_test_util import assert_fuzz_testing_acts_like

import kickmix as km


@pytest.mark.parametrize('n', [0, 1, 2, 10, 20, 256])
def test_builder_iadd(n: int):
    builder = km.CircuitBuilder()
    a = builder.create_quantum_register(n, name="a")
    b = builder.create_quantum_register(n, name="b")
    builder.iadd(b, target=a)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            a += b
            a %= 2**n
        """,
        shots=64,
        context={'n': n},
        input_sampler=lambda: {'a': random.randrange(1 << n), 'b': random.randrange(1 << n)},
    )


@pytest.mark.parametrize('n', [0, 1, 2, 10, 20, 256])
def test_builder_isub(n: int):
    builder = km.CircuitBuilder()
    a = builder.create_quantum_register(n, name="a")
    b = builder.create_quantum_register(n, name="b")
    builder.isub(b, target=a)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            a -= b
            a %= 2**n
        """,
        shots=64,
        context={'n': n},
        input_sampler=lambda: {'a': random.randrange(1 << n), 'b': random.randrange(1 << n)},
    )


@pytest.mark.parametrize('n', [0, 1, 2, 10, 20, 70, 256])
def test_builder_iadd_classical(n: int):
    offset = random.randrange(-(1 << n), 1 << n) if n else 0
    builder = km.CircuitBuilder()
    a = builder.create_quantum_register(n, name="a")
    c = builder.create_quantum_register(1, name="c")
    builder.free(builder.alloc_qubits(n))
    builder.iadd(offset, target=a, control=c[0], btol=float('inf'))
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            if c:
                a += offset
                a %= 2**n
        """,
        shots=64,
        context={'n': n, 'offset': offset},
        input_sampler=lambda: {'a': random.randrange(1 << n), 'c': random.randrange(2)},
    )


@pytest.mark.parametrize('n', [0, 1, 2, 10, 20, 70, 256])
def test_builder_isub_classical(n: int):
    offset = random.randrange(-(1 << n), 1 << n) if n else 0
    builder = km.CircuitBuilder()
    a = builder.create_quantum_register(n, name="a")
    c = builder.create_quantum_register(1, name="c")
    builder.free(builder.alloc_qubits(n))
    builder.isub(offset, target=a, control=c[0], btol=float('inf'))
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        """
            if c:
                a -= offset
                a %= 2**n
        """,
        shots=64,
        context={'n': n, 'offset': offset},
        input_sampler=lambda: {'a': random.randrange(1 << n), 'c': random.randrange(2)},
    )


def test_builder_iadd_auto_and_errors():
    builder = km.CircuitBuilder()
    a = builder.create_quantum_register(4, name="a")
    b = builder.create_quantum_register(4, name="b")
    c = builder.create_classical_register(4, name="c")

    # Quantum offset
    builder.iadd(b, target=a)
    # Integer offset
    builder.iadd(5, target=a)
    # Classical register offset
    builder.iadd(c, target=a)

    # btol with quantum offset is an error
    with pytest.raises(ValueError, match="btol is only supported for classical"):
        builder.iadd(b, target=a, btol=5.0)


def test_iadd_positional_args():
    builder = km.CircuitBuilder()
    a = builder.create_quantum_register(4, name="a")
    b = builder.create_quantum_register(4, name="b")
    # Positional offset with keyword target should work:
    builder.iadd(b, target=a)
    builder.isub(b, target=a)
    # Target cannot be passed positionally:
    with pytest.raises(TypeError):
        builder.iadd(b, a)
    with pytest.raises(TypeError):
        builder.isub(b, a)
