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

import numpy as np
import pytest

from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import Hadamard, Identity, IntEffect, IntState, XGate
from qualtran.bloqs.block_encoding import Unitary
from qualtran.bloqs.block_encoding.chebyshev_polynomial import (
    _chebyshev_poly_even,
    _chebyshev_poly_odd,
    _scaled_chebyshev_poly_even,
    _scaled_chebyshev_poly_odd,
    ChebyshevPolynomial,
)
from qualtran.symbolics import is_symbolic
from qualtran.testing import assert_equivalent_bloq_example_counts


def test_chebyshev_poly_even(bloq_autotester):
    bloq_autotester(_chebyshev_poly_even)


def test_chebyshev_poly_even_counts():
    assert_equivalent_bloq_example_counts(_chebyshev_poly_even)


def test_chebyshev_poly_odd(bloq_autotester):
    bloq_autotester(_chebyshev_poly_odd)


def test_chebyshev_poly_odd_counts():
    assert_equivalent_bloq_example_counts(_chebyshev_poly_odd)


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
