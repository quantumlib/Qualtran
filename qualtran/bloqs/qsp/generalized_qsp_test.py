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
from typing import Dict, Optional, Sequence

import cirq
import numpy as np
import pytest
import sympy
from attrs import define

from qualtran import Bloq, bloq_example, Controlled, CtrlSpec, GateWithRegisters
from qualtran.bloqs.basic_gates.su2_rotation import SU2RotationGate
from qualtran.bloqs.chemistry.ising.walk_operator import get_walk_operator_for_1d_ising_model
from qualtran.bloqs.for_testing.atom import TestGWRAtom
from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
from qualtran.bloqs.qsp.generalized_qsp import (
    _gqsp,
    _gqsp_with_large_negative_power,
    _gqsp_with_negative_power,
    GeneralizedQSP,
    qsp_complementary_polynomial,
    qsp_phase_factors,
)
from qualtran.linalg.polynomial.basic import evaluate_polynomial_of_unitary_matrix
from qualtran.linalg.polynomial.qsp_testing import (
    assert_is_qsp_polynomial,
    check_gqsp_polynomial_pair_on_random_points_on_unit_circle,
    random_qsp_polynomial,
    scale_down_to_qsp_polynomial,
)
from qualtran.linalg.testing import assert_matrices_almost_equal
from qualtran.resource_counting import SympySymbolAllocator
from qualtran.symbolics import Shaped
from qualtran.testing import execute_notebook


def test_gqsp_example(bloq_autotester):
    bloq_autotester(_gqsp)
    bloq_autotester(_gqsp_with_negative_power)
    bloq_autotester(_gqsp_with_large_negative_power)


@pytest.mark.parametrize("degree", [3, 4, 5, 10, 40, pytest.param(100, marks=pytest.mark.slow)])
def test_complementary_polynomial(degree: int):
    random_state = np.random.RandomState(42)

    for _ in range(5):
        P = random_qsp_polynomial(degree, random_state=random_state)
        Q = qsp_complementary_polynomial(P, verify=True)
        check_gqsp_polynomial_pair_on_random_points_on_unit_circle(P, Q, random_state=random_state)


@pytest.mark.parametrize("degree", [3, 4, 5, 10, 20, 30, pytest.param(100, marks=pytest.mark.slow)])
def test_real_polynomial_has_real_complementary_polynomial(degree: int):
    random_state = np.random.RandomState(42)

    for _ in range(10):
        P = random_qsp_polynomial(degree, random_state=random_state, only_real_coeffs=True)
        Q = qsp_complementary_polynomial(P, verify=True)
        Q = np.around(Q, decimals=8)
        assert np.isreal(Q).all()


def verify_generalized_qsp(
    U: GateWithRegisters,
    P: Sequence[complex],
    Q: Optional[Sequence[complex]] = None,
    *,
    negative_power: int = 0,
    tolerance: float = 1e-5,
):
    input_unitary = cirq.unitary(U)
    N = input_unitary.shape[0]
    if Q is None:
        gqsp_U = GeneralizedQSP.from_qsp_polynomial(
            U, P, negative_power=negative_power, verify=True
        )
    else:
        gqsp_U = GeneralizedQSP(U, P, Q, negative_power=negative_power)
    result_unitary = cirq.unitary(gqsp_U)

    expected_top_left = evaluate_polynomial_of_unitary_matrix(
        P, input_unitary, offset=-negative_power
    )
    actual_top_left = result_unitary[:N, :N]
    assert_matrices_almost_equal(expected_top_left, actual_top_left, atol=tolerance)

    assert not isinstance(gqsp_U.Q, Shaped)
    expected_bottom_left = evaluate_polynomial_of_unitary_matrix(
        gqsp_U.Q, input_unitary, offset=-negative_power
    )
    actual_bottom_left = result_unitary[N:, :N]
    assert_matrices_almost_equal(expected_bottom_left, actual_bottom_left, atol=tolerance)


@pytest.mark.parametrize("bitsize", [1, 2, 3])
@pytest.mark.parametrize(
    "degree",
    [
        pytest.param(degree, marks=pytest.mark.slow if degree > 40 else ())
        for degree in [2, 3, 5, 20, 40, 100, 150, 180]
    ],
)
def test_generalized_qsp_with_real_poly_on_random_unitaries(bitsize: int, degree: int):
    random_state = np.random.RandomState(42)

    for _ in range(10):
        U = MatrixGate.random(bitsize, random_state=random_state)
        P = random_qsp_polynomial(degree, random_state=random_state, only_real_coeffs=True)
        verify_generalized_qsp(U, P)


@pytest.mark.parametrize("bitsize", [1, 2, 3])
@pytest.mark.parametrize(
    "degree",
    [
        pytest.param(degree, marks=pytest.mark.slow if degree > 40 else ())
        for degree in [2, 3, 5, 40, 60, 100, 120]
    ],
)
@pytest.mark.parametrize("negative_power", [0, 1, 2])
def test_generalized_qsp_with_complex_poly_on_random_unitaries(
    bitsize: int, degree: int, negative_power: int
):
    random_state = np.random.RandomState(42)

    for _ in range(10):
        U = MatrixGate.random(bitsize, random_state=random_state)
        P = random_qsp_polynomial(degree, random_state=random_state)
        verify_generalized_qsp(U, P, negative_power=negative_power)


@pytest.mark.parametrize("negative_power", [0, 1, 2, 3, 4])
def test_call_graph(negative_power: int):
    random_state = np.random.RandomState(42)

    ssa = SympySymbolAllocator()
    arbitrary_rotation = SU2RotationGate.arbitrary(ssa)

    def catch_rotations(bloq: Bloq) -> Bloq:
        if isinstance(bloq, SU2RotationGate):
            return arbitrary_rotation
        return bloq

    U = MatrixGate.random(1, random_state=random_state)
    P = (0.5, 0, 0.5)
    gsqp_U = GeneralizedQSP.from_qsp_polynomial(U, P, negative_power=negative_power)

    g, sigma = gsqp_U.call_graph(max_depth=1, generalizer=catch_rotations)

    expected_counts: Dict[Bloq, int] = {arbitrary_rotation: 3}
    if negative_power < 2:
        expected_counts[U.controlled(control_values=[0])] = 2 - negative_power
    if negative_power > 0:
        expected_counts[U.adjoint().controlled()] = min(2, negative_power)
    if negative_power > 2:
        expected_counts[U.adjoint()] = negative_power - 2

    assert sigma == expected_counts


@bloq_example
def _gqsp_1d_ising() -> GeneralizedQSP:
    W, _ = get_walk_operator_for_1d_ising_model(2, 1e-4)
    gqsp_1d_ising = GeneralizedQSP.from_qsp_polynomial(W, (0.5, 0, 0.5), negative_power=1)
    return gqsp_1d_ising


def test_gqsp_1d_ising_example(bloq_autotester):
    bloq_autotester(_gqsp_1d_ising)


@pytest.mark.parametrize("degree", [2, 4, 6])
@pytest.mark.parametrize("negative_power", [0, 2, 3, 5])
def test_symbolic_call_graph(degree: int, negative_power: int):
    U = TestGWRAtom()
    P = Shaped(shape=(degree + 1,))
    gqsp = GeneralizedQSP.from_qsp_polynomial(U, P, negative_power=negative_power)

    ssa = SympySymbolAllocator()
    arbitrary_rotation = SU2RotationGate.arbitrary(ssa)

    def catch_rotations(bloq: Bloq) -> Bloq:
        if isinstance(bloq, SU2RotationGate):
            return arbitrary_rotation
        return bloq

    _, sigma = gqsp.call_graph(max_depth=1, generalizer=catch_rotations)

    expected_sigma: dict[Bloq, int] = {arbitrary_rotation: degree + 1}
    if degree > negative_power:
        expected_sigma[Controlled(U, CtrlSpec(cvs=0))] = degree - negative_power
    if negative_power > 0:
        expected_sigma[Controlled(U.adjoint(), CtrlSpec())] = min(degree, negative_power)
    if negative_power > degree:
        expected_sigma[U.adjoint()] = negative_power - degree

    assert sigma == expected_sigma


@define(slots=False)
class SymbolicGQSP:
    r"""Run the Generalized QSP algorithm on a symbolic input unitary

    Assuming the input unitary $U$ is symbolic, we construct the signal matrix $A$ as:

        $$
        A = \begin{bmatrix} U & 0 \\ 0 & 1 \end{bmatrix}
        $$

    This matrix $A$ is used to symbolically compute the GQSP transformed unitary, and the
    norms of the output blocks are upper-bounded. We then check if this upper-bound is negligible.
    """

    P: Sequence[complex]

    @cached_property
    def Q(self):
        return qsp_complementary_polynomial(self.P)

    def get_symbolic_qsp_matrix(self, U: sympy.Symbol):
        theta, phi, lambd = qsp_phase_factors(self.P, self.Q)

        # 0-controlled U gate
        A = np.array([[U, 0], [0, 1]])

        # compute the final GQSP unitary
        # mirrors the implementation of GeneralizedQSP::decompose_from_registers,
        # as cirq.unitary does not support symbolic matrices
        W = SU2RotationGate(theta[0], phi[0], lambd).rotation_matrix
        for t, p in zip(theta[1:], phi[1:]):
            W = A @ W
            W = SU2RotationGate(t, p, 0).rotation_matrix @ W

        return W

    @staticmethod
    def upperbound_matrix_norm(M, U) -> float:
        return float(sum(abs(c) for c in sympy.Poly(M, U).coeffs()))

    def verify(self):
        check_gqsp_polynomial_pair_on_random_points_on_unit_circle(
            self.P, self.Q, random_state=np.random.RandomState(42)
        )
        U = sympy.symbols('U')
        W = self.get_symbolic_qsp_matrix(U)
        actual_PU, actual_QU = W[:, 0]

        expected_PU = sympy.Poly(reversed(self.P), U)
        expected_QU = sympy.Poly(reversed(self.Q), U)

        error_PU = self.upperbound_matrix_norm(expected_PU - actual_PU, U)
        error_QU = self.upperbound_matrix_norm(expected_QU - actual_QU, U)

        assert abs(error_PU) <= 1e-5
        assert abs(error_QU) <= 1e-5


@pytest.mark.parametrize(
    "degree",
    [
        pytest.param(degree, marks=pytest.mark.slow if degree >= 5 else ())
        for degree in [2, 3, 4, 5, 10]
    ],
)
def test_generalized_real_qsp_with_symbolic_signal_matrix(degree: int):
    random_state = np.random.RandomState(102)

    for _ in range(10):
        P = random_qsp_polynomial(degree, random_state=random_state)
        SymbolicGQSP(P).verify()


@pytest.mark.parametrize("t", [2, 5, 7])
@pytest.mark.parametrize("precision", [1e-4, 1e-7, 1e-9])
def test_complementary_polynomials_for_jacobi_anger_approximations(t: float, precision: float):
    from qualtran.linalg.polynomial.jacobi_anger_approximations import (
        approx_exp_cos_by_jacobi_anger,
        degree_jacobi_anger_approximation,
    )

    if precision == 1e-9:
        pytest.skip("high precision tests not enforced yet (Issue #860)")

    random_state = np.random.RandomState(42 + int(t))

    d = degree_jacobi_anger_approximation(t, precision=precision)
    assert isinstance(d, int)
    P = approx_exp_cos_by_jacobi_anger(t, degree=d)
    # TODO(#860) current scaling method does not compute true maximum, so we scale down a bit more by (1 - 2\eps)
    P = scale_down_to_qsp_polynomial(list(P)) * (1 - 2 * precision)
    assert_is_qsp_polynomial(list(P))

    Q = qsp_complementary_polynomial(list(P), verify=True, verify_precision=1e-5)
    check_gqsp_polynomial_pair_on_random_points_on_unit_circle(
        list(P), Q, random_state=random_state, rtol=precision
    )
    verify_generalized_qsp(MatrixGate.random(1, random_state=random_state), list(P), Q)


@pytest.mark.notebook
def test_notebook():
    execute_notebook('generalized_qsp')
