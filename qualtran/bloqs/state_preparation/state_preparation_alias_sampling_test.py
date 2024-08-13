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

import cirq
import numpy as np
import pytest
import sympy

from qualtran import Bloq
from qualtran.bloqs.chemistry.ising import get_1d_ising_lcu_coeffs
from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
    _sparse_state_prep_alias,
    _sparse_state_prep_alias_from_list,
    _sparse_state_prep_alias_symb,
    _state_prep_alias,
    _state_prep_alias_symb,
    SparseStatePreparationAliasSampling,
    StatePreparationAliasSampling,
)
from qualtran.cirq_interop.testing import GateHelper
from qualtran.symbolics import ceil, log2
from qualtran.testing import assert_valid_bloq_decomposition, execute_notebook


def test_state_prep_alias_sampling_autotest(bloq_autotester):
    bloq_autotester(_state_prep_alias)
    bloq_autotester(_state_prep_alias_symb)


def test_sparse_state_prep_alias_sampling_autotest(bloq_autotester):
    bloq_autotester(_sparse_state_prep_alias)
    bloq_autotester(_sparse_state_prep_alias_from_list)
    bloq_autotester(_sparse_state_prep_alias_symb)


def test_mu_from_precision():
    coeffs = [1.0, 1, 3, 2]
    mu = 3
    bloq = StatePreparationAliasSampling.from_probabilities(
        coeffs, precision=2**-mu / len(coeffs) * sum(coeffs)
    )
    assert bloq.mu == mu


def test_mu_from_symbolic_precision():
    L, qlambda, mu = sympy.symbols(r"L \lambda \mu", integer=True)
    bloq = StatePreparationAliasSampling.from_n_coeff(L, qlambda, precision=2**-mu / L * qlambda)
    assert sympy.simplify(bloq.mu) == mu


def test_state_prep_alias_sampling_symb():
    bloq = _state_prep_alias_symb.make()
    L, logL, log_eps_inv = bloq.n_coeff, bloq.selection_bitsize, bloq.mu
    # Scales as 4L + O(logL) + O(log(1 / eps))
    expected_t_count_expr = 4 * L + 8 * log_eps_inv + 19 * logL - 8
    assert isinstance(expected_t_count_expr, sympy.Expr)
    assert bloq.t_complexity().t == expected_t_count_expr

    # Compare bloq counts via expression to actual bloq counts and make sure they
    # are "close enough".
    # The discrepency here comes from the fact that symbolic counts of
    # `PrepareUniformSuperposition` assume worst case and cannot compute the remainder
    # of dividing `n` by the highest power of `2` at resolution time.
    N, epsilon = 2**16 - 1, 1e-4
    random_state = cirq.value.parse_random_state(1234)
    lcu_coefficients = np.abs(random_state.randn(N).astype(float))
    bloq_concrete = StatePreparationAliasSampling.from_probabilities(
        unnormalized_probabilities=lcu_coefficients.tolist(), precision=epsilon
    )
    concrete_t_counts = bloq_concrete.t_complexity().t
    # Symbolic T-counts
    symb_t_counts = int(
        expected_t_count_expr.subs(
            {
                L: N,
                sympy.Symbol(r"\lambda"): sum(abs(lcu_coefficients)),
                sympy.Symbol(r"\epsilon"): epsilon,
            }
        )
    )
    np.testing.assert_allclose(concrete_t_counts, symb_t_counts, rtol=5e-4)
    # Ensure the symbolic bloq can decomposes into a composite bloq
    assert_valid_bloq_decomposition(bloq)


def assert_state_preparation_valid_for_coefficient(
    lcu_coefficients: np.ndarray, epsilon: float, *, sparse: bool = False, atol: float = 1e-6
):
    gate: Bloq
    coeff_precision = epsilon * np.sum(np.abs(lcu_coefficients))
    if sparse:
        gate = SparseStatePreparationAliasSampling.from_dense_probabilities(
            unnormalized_probabilities=lcu_coefficients.tolist(),
            precision=coeff_precision,
            nonzero_threshold=atol,
        )
    else:
        gate = StatePreparationAliasSampling.from_probabilities(
            unnormalized_probabilities=lcu_coefficients.tolist(), precision=coeff_precision
        )

    assert_valid_bloq_decomposition(gate)
    _ = gate.call_graph()

    g = GateHelper(gate)
    qubit_order = g.operation.qubits

    # Assertion to ensure that simulating the `decomposed_circuit` doesn't run out of memory.
    assert len(g.circuit.all_qubits()) < 20
    result = cirq.Simulator(dtype=np.complex128).simulate(g.circuit, qubit_order=qubit_order)
    state_vector = result.final_state_vector
    # State vector is of the form |l>|temp_{l}>. We trace out the |temp_{l}> part to
    # get the coefficients corresponding to |l>.
    L, logL = len(lcu_coefficients), len(g.quregs['selection'])
    qlambda = sum(abs(lcu_coefficients))
    state_vector = state_vector.reshape(2**logL, len(state_vector) // 2**logL)
    prepared_state = np.linalg.norm(state_vector, axis=1)

    # Assert that the absolute square of prepared state (probabilities instead of amplitudes) is
    # same as `lcu_coefficients` upto `epsilon`.
    np.testing.assert_allclose(
        abs(prepared_state[:L]) ** 2, lcu_coefficients / qlambda, atol=epsilon
    )
    np.testing.assert_allclose(np.abs(prepared_state[L:]), 0, atol=epsilon)


def test_state_preparation_via_coherent_alias_sampling_quick():
    num_sites, epsilon = 2, 1e-2
    lcu_coefficients = get_1d_ising_lcu_coeffs(num_sites)
    assert_state_preparation_valid_for_coefficient(lcu_coefficients, epsilon)


@pytest.mark.slow
@pytest.mark.parametrize("num_sites, epsilon", [[2, 3e-3], [3, 3.0e-3], [4, 5.0e-3], [7, 8.0e-3]])
def test_state_preparation_via_coherent_alias_sampling(num_sites, epsilon):
    lcu_coefficients = get_1d_ising_lcu_coeffs(num_sites)
    assert_state_preparation_valid_for_coefficient(lcu_coefficients, epsilon)


def test_state_preparation_via_coherent_alias_for_0_mu():
    lcu_coefficients = np.array([1 / 8] * 8)
    assert_state_preparation_valid_for_coefficient(lcu_coefficients, 2e-1)


def test_state_preparation_via_coherent_alias_sampling_diagram():
    data = np.asarray(range(1, 5)) / np.sum(range(1, 5))
    gate = StatePreparationAliasSampling.from_probabilities(
        unnormalized_probabilities=data.tolist(), precision=0.05
    )
    g = GateHelper(gate)
    qubit_order = g.operation.qubits

    circuit = cirq.Circuit(cirq.decompose_once(g.operation))
    cirq.testing.assert_has_diagram(
        circuit,
        '''
selection0: ────────UNIFORM(4)───In───────────────────×(y)───
                    │            │                    │
selection1: ────────target───────In───────────────────×(y)───
                                 │                    │
sigma_mu0: ─────────H⨂3──────────┼────────In(y)───────┼──────
                    │            │        │           │
sigma_mu1: ─────────H⨂3──────────┼────────In(y)───────┼──────
                    │            │        │           │
sigma_mu2: ─────────H⨂3──────────┼────────In(y)───────┼──────
                                 │        │           │
alt0: ───────────────────────────QROM_a───┼───────────×(x)───
                                 │        │           │
alt1: ───────────────────────────QROM_a───┼───────────×(x)───
                                 │        │           │
keep0: ──────────────────────────QROM_b───In(x)───────┼──────
                                 │        │           │
keep1: ──────────────────────────QROM_b───In(x)───────┼──────
                                 │        │           │
keep2: ──────────────────────────QROM_b───In(x)───────┼──────
                                          │           │
less_than_equal: ─────────────────────────⨁(x <= y)───@──────
''',
        qubit_order=qubit_order,
    )


def test_sparse_state_preparation_via_coherent_alias():
    lcu_coefficients = np.array([1 / 8 if j < 8 else 0.0 for j in range(16)])
    assert_state_preparation_valid_for_coefficient(lcu_coefficients, 2e-1, sparse=True)

    lcu_coefficients = np.array([1 if j < 6 else 0.0 for j in range(10)])
    assert_state_preparation_valid_for_coefficient(lcu_coefficients, 2e-1, sparse=True)


def test_symbolic_sparse_state_prep_t_complexity():
    from qualtran.cirq_interop.t_complexity_protocol import TComplexity

    N, d = sympy.symbols("N d", positive=True, integer=True)
    qlambda, eps = sympy.symbols(r"\lambda \epsilon")
    logN = ceil(log2(N))
    logd = ceil(log2(d))
    mu = ceil(log2(1 / (N * eps)))
    bloq = SparseStatePreparationAliasSampling.from_n_coeff(N, d, qlambda, precision=eps)
    assert bloq.t_complexity() == TComplexity(
        t=4 * d + 8 * mu + 7 * logN + 12 * logd - 8,
        clifford=d * mu * logN**2 + 13 * d + 47 * mu + 10 * logN + 52 * logd - 12,
        rotations=2,
    )


@pytest.mark.notebook
def test_notebook():
    execute_notebook('state_preparation_alias_sampling')
