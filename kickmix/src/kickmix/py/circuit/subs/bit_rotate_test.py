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

import pytest
from src.kickmix.py.circuit.subs.fuzz_test_util import assert_fuzz_testing_acts_like

import kickmix as km


def create_arg(builder: km.CircuitBuilder, arg: Any, name: str, *, length: int | None = None):
    if arg is km.q:
        v = builder.create_quantum_register(1 if length is None else length, name=name)
        return (v[0] if length is None else v), name
    elif arg is km.b:
        v = builder.create_classical_register(1 if length is None else length, name=name)
        return (v[0] if length is None else v), name
    elif isinstance(arg, (bool, int)):
        return arg, repr(arg)
    elif isinstance(arg, (list, tuple)) and all(isinstance(e, bool) for e in arg):
        return arg, repr(sum(e << k for k, e in enumerate(arg)))
    else:
        raise NotImplementedError(f'{arg=}')


def test_controlled_left_rotate_cost():
    builder = km.CircuitBuilder()
    target = builder.create_quantum_register(32, "target")
    control = builder.create_quantum_register(1, "control")[0]
    builder.left_rotate(target, control=control)
    circuit = builder.finish_circuit()
    assert circuit.max_magic() == (len(target) - 1)


def test_indexed_left_rotate_cost():
    builder = km.CircuitBuilder()
    target = builder.create_quantum_register(32, "target")
    shift = builder.create_quantum_register(5, "shift")
    builder.left_rotate(target, shift=shift)
    circuit = builder.finish_circuit()
    expected = (len(target) // 2) * (len(shift) + 1) - 1
    assert circuit.max_magic() == expected


def test_controlled_indexed_left_rotate_cost():
    builder = km.CircuitBuilder()
    target = builder.create_quantum_register(32, "target")
    shift = builder.create_quantum_register(5, "shift")
    control = builder.create_quantum_register(1, "control")[0]
    builder.left_rotate(target, shift=shift, control=control)
    circuit = builder.finish_circuit()
    expected = (len(target) // 2 + 1) * (len(shift) + 1) - 1
    assert circuit.max_magic() == expected


@pytest.mark.parametrize("n", [*range(8), 13, 64])
@pytest.mark.parametrize("shift", [-2, -1, 0, 1, 2, 13, km.q, km.b, [True], [True, False, True, True]])
@pytest.mark.parametrize("control", [km.q, km.b, False, True])
def test_fuzz_left_rotate(n: int, shift: Any, control: Any):
    builder = km.CircuitBuilder()
    target = builder.create_quantum_register(n, name="target")
    control, control_expr = create_arg(builder, control, "control")
    shift, shift_expr = create_arg(builder, shift, "shift", length=4)
    builder.left_rotate(target, shift=shift, control=control)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        f"""
            if n and {control_expr}:
                mask = ~(-1 << n)
                target = ((target << ({shift_expr} % n)) | target >> (-{shift_expr} % n)) & mask
        """,
        context={'n': n},
        shots=64,
    )


@pytest.mark.parametrize("n", [*range(8), 13, 64])
def test_fuzz_left_rotate_simple(n: int):
    builder = km.CircuitBuilder()
    target = builder.create_quantum_register(n, name="target")
    builder.left_rotate(target)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        f"""
            if n:
                mask = ~(-1 << n)
                target = ((target << 1) | target >> (n-1)) & mask
        """,
        context={'n': n},
        shots=64,
    )


@pytest.mark.parametrize("n", [*range(8), 13, 64])
@pytest.mark.parametrize("shift", [-2, -1, 0, 1, 2, 13, km.q, km.b, [True], [True, False, True, True]])
@pytest.mark.parametrize("control", [km.q, km.b, False, True])
def test_fuzz_right_rotate(n: int, shift: Any, control: Any):
    builder = km.CircuitBuilder()
    target = builder.create_quantum_register(n, name="target")
    control, control_expr = create_arg(builder, control, "control")
    shift, shift_expr = create_arg(builder, shift, "shift", length=4)
    builder.right_rotate(target, shift=shift, control=control)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        f"""
            if n and {control_expr}:
                mask = ~(-1 << n)
                target = ((target << (-{shift_expr} % n)) | target >> ({shift_expr} % n)) & mask
        """,
        context={'n': n},
        shots=64,
    )


@pytest.mark.parametrize("n", [*range(8), 13, 64])
def test_fuzz_right_rotate_simple(n: int):
    builder = km.CircuitBuilder()
    target = builder.create_quantum_register(n, name="target")
    builder.right_rotate(target)
    circuit = builder.finish_circuit()

    assert_fuzz_testing_acts_like(
        circuit,
        f"""
            if n:
                mask = ~(-1 << n)
                target = ((target << (n-1)) | (target >> 1)) & mask
        """,
        context={'n': n},
        shots=64,
    )
