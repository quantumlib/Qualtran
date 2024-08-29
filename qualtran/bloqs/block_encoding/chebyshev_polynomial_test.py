#  Copyright 2023 Google LLC
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

from functools import cached_property
from typing import Dict, Tuple

import numpy as np
import pytest
from attr import field, frozen
from numpy.typing import NDArray

from qualtran import BloqBuilder, Signature, Soquet, SoquetT
from qualtran.bloqs.basic_gates import Hadamard, Identity, IntEffect, IntState, XGate
from qualtran.bloqs.block_encoding import BlockEncoding, Unitary
from qualtran.bloqs.block_encoding.chebyshev_polynomial import (
    _chebyshev_poly_even,
    _chebyshev_poly_odd,
    _scaled_chebyshev_poly_even,
    _scaled_chebyshev_poly_odd,
    ChebyshevPolynomial,
)
from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
from qualtran.linalg.matrix import random_hermitian_matrix
from qualtran.symbolics import is_symbolic, SymbolicFloat, SymbolicInt
from qualtran.testing import assert_equivalent_bloq_example_counts, execute_notebook


def test_chebyshev_poly_even(bloq_autotester):
    bloq_autotester(_chebyshev_poly_even)


def test_chebyshev_poly_even_counts():
    assert_equivalent_bloq_example_counts(_chebyshev_poly_even)


def test_chebyshev_poly_odd(bloq_autotester):
    bloq_autotester(_chebyshev_poly_odd)


def test_chebyshev_poly_odd_counts():
    assert_equivalent_bloq_example_counts(_chebyshev_poly_odd)


def test_chebyshev_checks():
    from qualtran.bloqs.block_encoding.product_test import TestBlockEncoding

    with pytest.raises(ValueError):
        _ = ChebyshevPolynomial(Unitary(XGate()), -1)
    with pytest.raises(ValueError):
        _ = ChebyshevPolynomial(TestBlockEncoding(), 2)


def test_chebyshev_alpha():
    assert _chebyshev_poly_even().alpha == 1
    assert _chebyshev_poly_odd().alpha == 1


def gate_test(bloq):
    r'''Given `bloq` implementing $B[A/\alpha]$, returns $A$ by tensor contraction.'''
    alpha = bloq.alpha
    assert (
        not is_symbolic(bloq.system_bitsize)
        and not is_symbolic(bloq.ancilla_bitsize)
        and not is_symbolic(bloq.resource_bitsize)
    )
    bb = BloqBuilder()
    system = bb.add_register("system", bloq.system_bitsize)
    ancilla = bb.add(IntState(0, bloq.ancilla_bitsize))
    resource = bb.add(IntState(0, bloq.resource_bitsize))
    system, ancilla, resource = bb.add(bloq, system=system, ancilla=ancilla, resource=resource)
    bb.add(IntEffect(0, bloq.ancilla_bitsize), val=ancilla)
    bb.add(IntEffect(0, bloq.resource_bitsize), val=resource)
    bloq = bb.finalize(system=system)
    return bloq.tensor_contract() * alpha


def t4(x):
    return 8 * np.linalg.matrix_power(x, 4) - 8 * np.linalg.matrix_power(x, 2) + np.eye(2)


def t5(x):
    return 16 * np.linalg.matrix_power(x, 5) - 20 * np.linalg.matrix_power(x, 3) + 5 * x


@pytest.mark.slow
def test_chebyshev_poly_even_tensors():
    from_gate = t4((XGate().tensor_contract() + Hadamard().tensor_contract()) / 2.0)
    bloq = _chebyshev_poly_even()
    from_tensors = gate_test(bloq)
    np.testing.assert_allclose(from_gate, from_tensors, atol=1e-14)


def test_chebyshev_poly_odd_tensors():
    from_gate = t5(Hadamard().tensor_contract())
    bloq = _chebyshev_poly_odd()
    assert not is_symbolic(bloq.system_bitsize)

    bb = BloqBuilder()
    system = bb.add_register("system", bloq.system_bitsize)
    system = bb.add(bloq, system=system)
    bloq = bb.finalize(system=system)
    from_tensors = bloq.tensor_contract()
    np.testing.assert_allclose(from_gate, from_tensors, atol=1e-14)


def test_chebyshev_zero_order():
    bloq = ChebyshevPolynomial(Unitary(Hadamard()), order=0)
    bb = BloqBuilder()
    system = bb.add_register("system", 1)
    system = bb.add(bloq, system=system)
    bloq = bb.finalize(system=system)
    from_tensors = bloq.tensor_contract()

    from_gate = Identity().tensor_contract()
    np.testing.assert_allclose(from_gate, from_tensors, atol=1e-14)


def test_chebyshev_first_order():
    bloq = ChebyshevPolynomial(Unitary(Hadamard()), order=1)
    bb = BloqBuilder()
    system = bb.add_register("system", 1)
    system = bb.add(bloq, system=system)
    bloq = bb.finalize(system=system)
    from_tensors = bloq.tensor_contract()

    from_gate = Hadamard().tensor_contract()
    np.testing.assert_allclose(from_gate, from_tensors, atol=1e-14)


def test_scaled_chebyshev_poly_even(bloq_autotester):
    bloq_autotester(_scaled_chebyshev_poly_even)


def test_scaled_chebyshev_poly_odd(bloq_autotester):
    bloq_autotester(_scaled_chebyshev_poly_odd)


@pytest.mark.slow
def test_scaled_chebyshev_even_tensors():
    from_gate = t4((XGate().tensor_contract() + Hadamard().tensor_contract()))
    bloq = _scaled_chebyshev_poly_even()
    from_tensors = gate_test(bloq)
    np.testing.assert_allclose(from_gate, from_tensors, atol=0.06)


@pytest.mark.slow
def test_scaled_chebyshev_odd_tensors():
    from_gate = t5(Hadamard().tensor_contract() * 3.14)
    bloq = _scaled_chebyshev_poly_odd()
    from_tensors = gate_test(bloq)
    np.testing.assert_allclose(from_gate, from_tensors, atol=1e-14)


@pytest.mark.notebook
def test_notebook():
    execute_notebook('chebyshev_polynomial')


@frozen
class TestBlockEncoding(BlockEncoding):
    """Instance of `BlockEncoding` to block encode a matrix with one system qubit by adding one
    ancilla qubit and one resource qubit."""

    matrix: Tuple[Tuple[complex, ...], ...] = field(
        converter=lambda mat: tuple(tuple(row) for row in mat)
    )
    alpha: SymbolicFloat = 1
    epsilon: SymbolicFloat = 0

    system_bitsize: SymbolicInt = 1
    ancilla_bitsize: SymbolicInt = 1
    resource_bitsize: SymbolicInt = 1

    @classmethod
    def from_matrix(cls, a: NDArray):
        """Given A, constructs a block encoding of A."""
        ua = np.zeros((4, 4))
        a = a.astype(complex)
        ua[:2, :2] = a.real
        ua[2:, 2:] = -a.real
        ua[:2, 2:] = ua[2:, :2] = np.sqrt(np.eye(2) - a.T @ a).real
        return cls(np.kron(np.eye(2), ua))

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(system=1, ancilla=1, resource=1)

    @property
    def signal_state(self) -> BlackBoxPrepare:
        return BlackBoxPrepare(PrepareIdentity.from_bitsizes([1]))

    def build_composite_bloq(
        self, bb: BloqBuilder, system: Soquet, ancilla: Soquet, resource: Soquet
    ) -> Dict[str, SoquetT]:
        bits = bb.join(np.array([system, ancilla, resource]))
        bits = bb.add(MatrixGate(3, self.matrix, atol=3e-8), q=bits)
        system, ancilla, resource = bb.split(bits)
        return {"system": system, "ancilla": ancilla, "resource": resource}


def test_chebyshev_matrix():
    a = (XGate().tensor_contract() + 3.0 * Hadamard().tensor_contract()) / 4.0
    from_gate = t4(a)
    bloq = ChebyshevPolynomial(TestBlockEncoding.from_matrix(a), order=4)
    from_tensors = gate_test(bloq)
    np.testing.assert_allclose(from_gate, from_tensors, atol=2e-15)


def test_chebyshev_poly_signal_state():
    assert isinstance(_chebyshev_poly_even().signal_state.prepare, PrepareIdentity)
    _ = _chebyshev_poly_even().signal_state.decompose_bloq()


@pytest.mark.slow
def test_chebyshev_matrix_random():
    random_state = np.random.RandomState(1234)
    for _ in range(20):
        a = random_hermitian_matrix(2, random_state)
        from_gate = t4(a)
        bloq = ChebyshevPolynomial(TestBlockEncoding.from_matrix(a), order=4)
        from_tensors = gate_test(bloq)
        np.testing.assert_allclose(from_gate, from_tensors, atol=3e-8)
