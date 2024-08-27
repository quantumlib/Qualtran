#  Copyright 2024 Google LLC
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

import random
from typing import cast

import numpy as np
import pytest
from attrs import evolve

from qualtran import BloqBuilder, QAny, Register, Signature, Soquet
from qualtran.bloqs.basic_gates import (
    CNOT,
    Hadamard,
    IntEffect,
    IntState,
    Ry,
    Swap,
    TGate,
    XGate,
    ZGate,
)
from qualtran.bloqs.block_encoding.linear_combination import (
    _linear_combination_block_encoding,
    LinearCombination,
)
from qualtran.bloqs.block_encoding.product_test import TestBlockEncoding
from qualtran.bloqs.block_encoding.unitary import Unitary
from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
from qualtran.testing import execute_notebook


def test_linear_combination(bloq_autotester):
    bloq_autotester(_linear_combination_block_encoding)


def test_linear_combination_signature():
    assert _linear_combination_block_encoding().signature == Signature(
        [Register("system", QAny(1)), Register("ancilla", QAny(2)), Register("resource", QAny(5))]
    )


def test_linear_combination_checks():
    with pytest.raises(ValueError):
        _ = LinearCombination((), (), lambd_bits=1)
    with pytest.raises(ValueError):
        _ = LinearCombination((Unitary(TGate()),), (), lambd_bits=1)
    with pytest.raises(ValueError):
        _ = LinearCombination((Unitary(TGate()),), (1.0,), lambd_bits=1)
    with pytest.raises(ValueError):
        _ = LinearCombination((Unitary(TGate()), Unitary(CNOT())), (1.0,), lambd_bits=1)
    with pytest.raises(ValueError):
        _ = LinearCombination((Unitary(TGate()), Unitary(Hadamard())), (0.0, 0.0), lambd_bits=1)
    with pytest.raises(ValueError):
        _ = LinearCombination((Unitary(TGate()), TestBlockEncoding()), (1.0, 1.0), lambd_bits=1)


def test_linear_combination_params():
    u1 = evolve(Unitary(TGate()), alpha=0.5, ancilla_bitsize=2, resource_bitsize=1, epsilon=0.01)
    u2 = evolve(Unitary(Hadamard()), alpha=0.5, ancilla_bitsize=1, resource_bitsize=1, epsilon=0.1)
    bloq = LinearCombination((u1, u2), (1.0, 1.0), lambd_bits=1)
    assert bloq.system_bitsize == 1
    assert bloq.alpha == (0.5 * 0.5 + 0.5 * 0.5) * 2
    assert bloq.epsilon == (0.5 + 0.5) * max(0.01, 0.1)
    assert bloq.ancilla_bitsize == 1 + max(1, 2)
    assert bloq.resource_bitsize == max(1, 1) + 4  # dependent on state preparation


def get_tensors(bloq):
    alpha = bloq.alpha
    bb = BloqBuilder()
    system = bb.add_register("system", cast(int, bloq.system_bitsize))
    ancilla = cast(Soquet, bb.add(IntState(0, bloq.ancilla_bitsize)))
    resource = cast(Soquet, bb.add(IntState(0, bloq.resource_bitsize)))
    system, ancilla, resource = bb.add_t(bloq, system=system, ancilla=ancilla, resource=resource)
    bb.add(IntEffect(0, cast(int, bloq.ancilla_bitsize)), val=ancilla)
    bb.add(IntEffect(0, cast(int, bloq.resource_bitsize)), val=resource)
    bloq = bb.finalize(system=system)
    return bloq.tensor_contract() * alpha


def test_linear_combination_tensors():
    bloq = _linear_combination_block_encoding()
    from_gate = (
        0.25 * TGate().tensor_contract()
        + -0.25 * Hadamard().tensor_contract()
        + 0.25 * XGate().tensor_contract()
        + -0.25 * ZGate().tensor_contract()
    )
    from_tensors = get_tensors(bloq)
    np.testing.assert_allclose(from_gate, from_tensors)


def run_gate_test(gates, lambd, lambd_bits=1, atol=1e-07):
    bloq = LinearCombination(tuple(Unitary(g) for g in gates), lambd, lambd_bits)
    from_gate = sum(l * g.tensor_contract() for l, g in zip(lambd, gates))
    from_tensors = get_tensors(bloq)
    np.testing.assert_allclose(from_gate, from_tensors, atol=atol)


def test_linear_combination_alpha():
    lambd = (2.0, 3.0)
    gates = (evolve(Unitary(TGate()), alpha=2.0), evolve(Unitary(Hadamard()), alpha=4.0))
    bloq = LinearCombination(gates, lambd, lambd_bits=1)
    from_gate = sum(l * g.U.tensor_contract() * g.alpha for l, g in zip(lambd, gates))
    from_tensors = get_tensors(bloq)
    np.testing.assert_allclose(from_gate, from_tensors)


# all coefficients are multiples of small negative powers of 2 after normalization
exact2 = [[0.0, 1.0], [1 / 3, 1 / 3], [0.5, 0.5], [0.25, 0.25], [2.0, 6.0], [1.0, 0.0]]
exact3 = [
    [0.0, 0.0, 1.0],
    [1.0, -0.5, 0.5],
    [1 / 4, 1 / 4, -1 / 2],
    [1 / 2, 1 / 4, 1 / 4],
    [3 / 16, -1 / 4, 1 / 16],
    [-1.0, 0.0, 0.0],
]
exact5 = [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [9 / 16, -1 / 16, 1 / 8, -1 / 8, 1 / 8],
    [1 / 4, 1 / 8, -1 / 16, 1 / 32, -1 / 32],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 0.5, 0.5, -1.0, 1.0],
]


@pytest.mark.parametrize('lambd', exact2)
def test_linear_combination2(lambd):
    run_gate_test([TGate(), Hadamard()], lambd)


@pytest.mark.parametrize('lambd', exact2)
def test_linear_combination_twogate(lambd):
    run_gate_test([CNOT(), Hadamard().controlled()], lambd)


@pytest.mark.parametrize('lambd', exact3)
def test_linear_combination3(lambd):
    run_gate_test([TGate(), Hadamard(), XGate()], lambd)


@pytest.mark.parametrize('lambd', exact5)
def test_linear_combination5(lambd):
    run_gate_test([TGate(), Hadamard(), XGate(), ZGate(), Ry(angle=np.pi / 4.0)], lambd)


# coefficients are not multiples of small negative powers of 2 after normalization
approx2 = [
    [1 / 3, 2 / 3],
    [2 / 3, 1 / 3],
    [1 / 2, -1 / 4],
    [1 / 8, 1 / 2],
    [-1 / 3, 1 / 2],
    [1 / 3, 1 / 5],
    [1.0, -7.0],
    [3.7, 2.1],
    [1.0, 0.0],
]
approx5 = [
    [1.0, 0.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, -1.0, 1.0],
    [1.5, 2.4, -3.3, 4.2, 5.1],
    [1 / 7, -2 / 7, 3 / 7, 4 / 7, -5 / 7],
]


@pytest.mark.slow
@pytest.mark.parametrize('lambd', approx2)
def test_linear_combination_approx2(lambd):
    run_gate_test([TGate(), Hadamard()], lambd, lambd_bits=9, atol=0.003)


@pytest.mark.slow
@pytest.mark.parametrize('lambd', approx5)
def test_linear_combination_approx5(lambd):
    run_gate_test(
        [
            TGate().controlled(),
            Hadamard().controlled(),
            CNOT(),
            Swap(1),
            Ry(angle=np.pi / 4.0).controlled(),
        ],
        lambd,
        lambd_bits=9,
        atol=0.002,
    )


@pytest.mark.slow
def test_linear_combination_approx_random():
    random_state = np.random.RandomState(1234)

    for _ in range(10):
        n = random_state.randint(3, 6)
        bitsize = random_state.randint(1, 3)
        gates = [MatrixGate.random(bitsize, random_state=random_state) for _ in range(n)]
        lambd = [random.uniform(-10, 10) for _ in range(n)]
        run_gate_test(gates, lambd, lambd_bits=9, atol=0.02)


def test_linear_combination_signal_state():
    assert isinstance(_linear_combination_block_encoding().signal_state.prepare, PrepareIdentity)
    _ = _linear_combination_block_encoding().signal_state.decompose_bloq()


@pytest.mark.notebook
def test_notebook():
    execute_notebook('linear_combination')
