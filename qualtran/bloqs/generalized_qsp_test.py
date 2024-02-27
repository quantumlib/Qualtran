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
from typing import Optional, Sequence, Tuple, Union

import attrs
import cirq
import numpy as np
import pytest
import scipy
import sympy
from attrs import define, frozen
from cirq.testing import random_unitary
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

from qualtran import (
    BloqBuilder,
    BoundedQUInt,
    GateWithRegisters,
    QBit,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.generalized_qsp import (
    GeneralizedQSP,
    HamiltonianSimulationByGQSP,
    qsp_complementary_polynomial,
    qsp_phase_factors,
    SU2RotationGate,
)
from qualtran.bloqs.qubitization_walk_operator import QubitizationWalkOperator
from qualtran.bloqs.select_and_prepare import PrepareOracle, SelectOracle
from qualtran.cirq_interop import BloqAsCirqGate


def assert_angles_almost_equal(
    actual: Union[float, Sequence[float]], desired: Union[float, Sequence[float]]
):
    """Helper to check if two angle sequences are equal (up to multiples of 2*pi)"""
    np.testing.assert_almost_equal(np.exp(np.array(actual) * 1j), np.exp(np.array(desired) * 1j))


def test_cirq_decompose_SU2_to_single_qubit_pauli_gates():
    random_state = np.random.default_rng(42)

    for _ in range(20):
        theta = random_state.random() * 2 * np.pi
        phi = random_state.random() * 2 * np.pi
        lambd = random_state.random() * 2 * np.pi

        gate = SU2RotationGate(theta, phi, lambd)

        expected = gate.rotation_matrix
        actual = cirq.unitary(BloqAsCirqGate(gate))
        np.testing.assert_allclose(actual, expected)


def check_polynomial_pair_on_random_points_on_unit_circle(
    P: Union[Sequence[complex], Polynomial],
    Q: Union[Sequence[complex], Polynomial],
    *,
    random_state: np.random.RandomState,
    rtol: float = 1e-7,
    n_points: int = 1000,
):
    P = Polynomial(P)
    Q = Polynomial(Q)

    for _ in range(n_points):
        z = np.exp(random_state.random() * np.pi * 2j)
        np.testing.assert_allclose(np.abs(P(z)) ** 2 + np.abs(Q(z)) ** 2, 1, rtol=rtol)


def random_qsp_polynomial(
    degree: int, *, random_state: np.random.RandomState, only_real_coeffs=False
) -> Sequence[complex]:
    poly = random_state.random(size=degree) / degree
    if not only_real_coeffs:
        poly = poly * np.exp(random_state.random(size=degree) * np.pi * 2j)
    return poly


@pytest.mark.parametrize("degree", [4, 5])
def test_complementary_polynomial_quick(degree: int):
    random_state = np.random.RandomState(42)

    for _ in range(2):
        P = random_qsp_polynomial(degree, random_state=random_state)
        Q = qsp_complementary_polynomial(P, verify=True)
        check_polynomial_pair_on_random_points_on_unit_circle(P, Q, random_state=random_state)


@pytest.mark.slow
@pytest.mark.parametrize("degree", [3, 4, 5, 10, 20, 30, 100])
def test_complementary_polynomial(degree: int):
    random_state = np.random.RandomState(42)

    for _ in range(10):
        P = random_qsp_polynomial(degree, random_state=random_state)
        Q = qsp_complementary_polynomial(P, verify=True)
        check_polynomial_pair_on_random_points_on_unit_circle(P, Q, random_state=random_state)


@pytest.mark.parametrize("degree", [3, 4, 5, 10, 20, 30, 100])
def test_real_polynomial_has_real_complementary_polynomial(degree: int):
    random_state = np.random.RandomState(42)

    for _ in range(10):
        P = random_qsp_polynomial(degree, random_state=random_state, only_real_coeffs=True)
        Q = qsp_complementary_polynomial(P, verify=True)
        Q = np.around(Q, decimals=8)
        assert np.isreal(Q).all()


@frozen
class RandomGate(GateWithRegisters):
    bitsize: int
    matrix: NDArray

    @staticmethod
    def create(bitsize: int, *, random_state=None) -> 'RandomGate':
        matrix = random_unitary(2**bitsize, random_state=random_state)
        return RandomGate(bitsize, matrix)

    @property
    def signature(self) -> Signature:
        return Signature.build(q=self.bitsize)

    def _unitary_(self):
        return self.matrix

    def adjoint(self) -> GateWithRegisters:
        return RandomGate(self.bitsize, self.matrix.conj().T)

    def __pow__(self, power):
        if power == -1:
            return self.adjoint()
        return NotImplemented

    def __hash__(self):
        return hash(tuple(np.ravel(self.matrix)))


def evaluate_polynomial_of_matrix(
    P: Sequence[complex], U: NDArray, *, negative_power: int = 0
) -> NDArray:
    assert U.ndim == 2 and U.shape[0] == U.shape[1]

    pow_U = np.linalg.matrix_power(U.conj().T, negative_power)
    result = np.zeros(U.shape, dtype=U.dtype)

    for c in P:
        result += pow_U * c
        pow_U = pow_U @ U

    return result


def assert_matrices_almost_equal(A: NDArray, B: NDArray):
    assert A.shape == B.shape
    assert np.linalg.norm(A - B) <= 1e-5


def verify_generalized_qsp(
    U: GateWithRegisters,
    P: Sequence[complex],
    Q: Optional[Sequence[complex]] = None,
    *,
    negative_power: int = 0,
):
    input_unitary = cirq.unitary(U)
    N = input_unitary.shape[0]
    if Q is None:
        gqsp_U = GeneralizedQSP.from_qsp_polynomial(U, P, negative_power=negative_power)
    else:
        gqsp_U = GeneralizedQSP(U, P, Q, negative_power=negative_power)
    result_unitary = cirq.unitary(gqsp_U)

    expected_top_left = evaluate_polynomial_of_matrix(
        P, input_unitary, negative_power=negative_power
    )
    actual_top_left = result_unitary[:N, :N]
    assert_matrices_almost_equal(expected_top_left, actual_top_left)

    expected_bottom_left = evaluate_polynomial_of_matrix(
        gqsp_U.Q, input_unitary, negative_power=negative_power
    )
    actual_bottom_left = result_unitary[N:, :N]
    assert_matrices_almost_equal(expected_bottom_left, actual_bottom_left)


@pytest.mark.slow
@pytest.mark.parametrize("bitsize", [1, 2, 3])
@pytest.mark.parametrize("degree", [2, 3, 4, 5, 50, 100, 150, 180])
def test_generalized_qsp_with_real_poly_on_random_unitaries(bitsize: int, degree: int):
    random_state = np.random.RandomState(42)

    for _ in range(10):
        U = RandomGate.create(bitsize, random_state=random_state)
        P = random_qsp_polynomial(degree, random_state=random_state, only_real_coeffs=True)
        verify_generalized_qsp(U, P)


@pytest.mark.slow
@pytest.mark.parametrize("bitsize", [1, 2, 3])
@pytest.mark.parametrize("degree", [2, 3, 4, 5, 50, 100, 120])
@pytest.mark.parametrize("negative_power", [0, 1, 2])
def test_generalized_qsp_with_complex_poly_on_random_unitaries(
    bitsize: int, degree: int, negative_power: int
):
    random_state = np.random.RandomState(42)

    for _ in range(10):
        U = RandomGate.create(bitsize, random_state=random_state)
        P = random_qsp_polynomial(degree, random_state=random_state)
        verify_generalized_qsp(U, P, negative_power=negative_power)


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
        check_polynomial_pair_on_random_points_on_unit_circle(
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


@pytest.mark.slow
@pytest.mark.parametrize("degree", [2, 3, 4, 5, 10])
def test_generalized_real_qsp_with_symbolic_signal_matrix(degree: int):
    random_state = np.random.RandomState(102)

    for _ in range(10):
        P = random_qsp_polynomial(degree, random_state=random_state)
        SymbolicGQSP(P).verify()


@frozen
class RandomPrepareOracle(PrepareOracle):
    U: RandomGate

    @property
    def selection_registers(self) -> tuple[Register, ...]:
        return (Register('selection', BoundedQUInt(bitsize=self.U.bitsize)),)

    @staticmethod
    def create(bitsize: int, *, random_state: np.random.RandomState):
        matrix = random_unitary(2**bitsize, random_state=random_state)

        # make the first column (weights alpha_i) all reals
        alpha = matrix[:, 0]
        matrix = matrix * (alpha.conj() / np.abs(alpha))[:, None]

        # verify that it is still unitary
        np.testing.assert_allclose(matrix @ matrix.conj().T, np.eye(2**bitsize), atol=1e-10)
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(2**bitsize), atol=1e-10)

        return RandomPrepareOracle(RandomGate(bitsize, matrix))

    def build_composite_bloq(self, bb: BloqBuilder, selection: SoquetT) -> dict[str, SoquetT]:
        selection = bb.add(self.U, q=selection)
        return {'selection': selection}

    def __pow__(self, power):
        if power == -1:
            return self.U.adjoint()
        return NotImplemented

    @cached_property
    def alphas(self):
        np.testing.assert_almost_equal(np.imag(self.U.matrix[:, 0]), 0)
        return self.U.matrix[:, 0] ** 2


@frozen
class PauliSelectOracle(SelectOracle):
    select_bitsize: int
    target_bitsize: int
    select_unitaries: tuple[cirq.DensePauliString, ...]
    control_val: Optional[int] = None

    @property
    def control_registers(self) -> Tuple[Register, ...]:
        return () if self.control_val is None else (Register('control', QBit()),)

    @property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (Register('selection', BoundedQUInt(bitsize=self.select_bitsize)),)

    @property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register('target', BoundedQUInt(bitsize=self.target_bitsize)),)

    def adjoint(self):
        return self

    def __pow__(self, power):
        if abs(power) == 1:
            return self
        return NotImplemented

    def controlled(
        self,
        num_controls: Optional[int] = None,
        control_values=None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> 'cirq.Gate':
        if num_controls is None:
            num_controls = 1
        if control_values is None:
            control_values = [1] * num_controls
        if (
            isinstance(control_values, Sequence)
            and isinstance(control_values[0], int)
            and len(control_values) == 1
            and self.control_val is None
        ):
            return attrs.evolve(self, control_val=control_values[0])
        raise NotImplementedError()

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        selection: NDArray[cirq.Qid],
        target: NDArray[cirq.Qid],
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:
        if self.control_val is not None:
            selection = np.concatenate([selection, quregs['control']])

        for cv, U in enumerate(self.select_unitaries):
            bits = tuple(map(int, bin(cv)[2:].zfill(self.select_bitsize)))[::-1]
            if self.control_val is not None:
                bits = (*bits, self.control_val)
            yield U.on(*target).controlled_by(*selection, control_values=bits)


def random_qubitization_walk_operator(
    select_bitsize: int, target_bitsize: int, *, random_state: np.random.RandomState
) -> tuple[QubitizationWalkOperator, cirq.PauliSum]:
    prepare = RandomPrepareOracle.create(select_bitsize, random_state=random_state)

    dps = tuple(
        cirq.DensePauliString(random_state.random_integers(0, 3, size=target_bitsize))
        for _ in range(2**select_bitsize)
    )
    select = PauliSelectOracle(select_bitsize, target_bitsize, dps)

    np.testing.assert_allclose(np.linalg.norm(prepare.alphas, 1), 1)

    ham = cirq.PauliSum.from_pauli_strings(
        [
            dp.on(*cirq.LineQubit.range(target_bitsize)) * alpha
            for dp, alpha in zip(dps, prepare.alphas)
        ]
    )

    return QubitizationWalkOperator(prepare=prepare, select=select), ham


def verify_hamiltonian_simulation_by_gqsp(
    W: QubitizationWalkOperator, H: NDArray[np.complex_], *, t: float, alpha: float = 1
):
    N = H.shape[0]
    W_e_iHt = HamiltonianSimulationByGQSP(W, t=t, alpha=alpha, precision=1e-10)
    result_unitary = cirq.unitary(W_e_iHt)

    expected_top_left = scipy.linalg.expm(1j * H * t)
    actual_top_left = result_unitary[:N, :N]
    assert_matrices_almost_equal(expected_top_left, actual_top_left)


@pytest.mark.parametrize("precision", [1e-5, 1e-10])
def test_cos_approximation_fast(precision: float):
    random_state = np.random.RandomState(42)

    for t in [1, 2, 3]:
        for alpha in [0.5, 1]:
            bloq = HamiltonianSimulationByGQSP(None, t=t, alpha=alpha, precision=precision)
            check_polynomial_pair_on_random_points_on_unit_circle(
                bloq._approx_cos, [0], random_state=random_state, rtol=precision
            )


@pytest.mark.slow
@pytest.mark.parametrize("precision", [1e-5, 1e-7, 1e-10])
def test_cos_approximation(precision):
    random_state = np.random.RandomState(42)

    for t in random_state.random(10):
        for alpha in random_state.random(5):
            bloq = HamiltonianSimulationByGQSP(None, t=t * 10, alpha=alpha, precision=precision)
            check_polynomial_pair_on_random_points_on_unit_circle(
                bloq._approx_cos, [0], random_state=random_state, rtol=precision
            )


# @pytest.mark.slow
@pytest.mark.parametrize("select_bitsize", [1])
@pytest.mark.parametrize("target_bitsize", [1])
def test_hamiltonian_simulation_by_gqsp(select_bitsize: int, target_bitsize: int):
    random_state = np.random.RandomState(42)

    for _ in range(5):
        W, H = random_qubitization_walk_operator(
            select_bitsize, target_bitsize, random_state=random_state
        )
        verify_hamiltonian_simulation_by_gqsp(W, H.matrix(), t=5, alpha=1)
