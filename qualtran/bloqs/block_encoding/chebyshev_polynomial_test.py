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
    soqs = {}
    soqs["system"] = bb.add_register("system", bloq.system_bitsize)
    if bloq.ancilla_bitsize > 0:
        soqs["ancilla"] = bb.add(IntState(0, bloq.ancilla_bitsize))
    if bloq.resource_bitsize > 0:
        soqs["resource"] = bb.add(IntState(0, bloq.resource_bitsize))
    soqs = bb.add_d(bloq, **soqs)
    if bloq.ancilla_bitsize > 0:
        bb.add(IntEffect(0, bloq.ancilla_bitsize), val=soqs["ancilla"])
    if bloq.resource_bitsize > 0:
        bb.add(IntEffect(0, bloq.resource_bitsize), val=soqs["resource"])
    bloq = bb.finalize(system=soqs["system"])
    return bloq.tensor_contract() * alpha


def t2(x):
    return 2 * np.linalg.matrix_power(x, 2) - np.eye(2)


def t4(x):
    return 8 * np.linalg.matrix_power(x, 4) - 8 * np.linalg.matrix_power(x, 2) + np.eye(2)


def t5(x):
    return 16 * np.linalg.matrix_power(x, 5) - 20 * np.linalg.matrix_power(x, 3) + 5 * x


@pytest.mark.slow
def test_chebyshev_poly_even_tensors():
    from_gate = t4(XGate().tensor_contract())
    bloq = _chebyshev_poly_even()
    from_tensors = gate_test(bloq)
    np.testing.assert_allclose(from_gate, from_tensors, atol=1e-14)


def test_chebyshev_poly_odd_tensors():
    from_gate = t5(Hadamard().tensor_contract())
    bloq = _chebyshev_poly_odd()
    assert not is_symbolic(bloq.system_bitsize)

    from_tensors = gate_test(bloq)
    np.testing.assert_allclose(from_gate, from_tensors, atol=1e-14)


def test_chebyshev_zero_order():
    from_gate = Identity().tensor_contract()
    bloq = ChebyshevPolynomial(Unitary(XGate()), order=0)
    from_tensors = gate_test(bloq)
    np.testing.assert_allclose(from_gate, from_tensors, atol=1e-14)


def test_chebyshev_first_order():
    from_gate = XGate().tensor_contract()
    bloq = ChebyshevPolynomial(Unitary(XGate()), order=1)
    from_tensors = gate_test(bloq)
    np.testing.assert_allclose(from_gate, from_tensors, atol=1e-14)


def test_scaled_chebyshev_poly_even(bloq_autotester):
    bloq_autotester(_scaled_chebyshev_poly_even)


def test_scaled_chebyshev_poly_odd(bloq_autotester):
    bloq_autotester(_scaled_chebyshev_poly_odd)


def test_scaled_chebyshev_poly_even_tensors():
    from_gate = t2(XGate().tensor_contract() * 3.14)
    bloq = _scaled_chebyshev_poly_even()
    from_tensors = gate_test(bloq)
    np.testing.assert_allclose(from_gate, from_tensors, atol=0.06)


@pytest.mark.slow
def test_scaled_chebyshev_poly_odd_tensors():
    from_gate = t5(Hadamard().tensor_contract() * 3.14)
    bloq = _scaled_chebyshev_poly_odd()
    from_tensors = gate_test(bloq)
    np.testing.assert_allclose(from_gate, from_tensors, atol=1e-14)
