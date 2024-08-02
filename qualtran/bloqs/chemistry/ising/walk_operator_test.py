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
import sympy

from qualtran.bloqs.chemistry.ising.hamiltonian import get_1d_ising_hamiltonian_norm_upper_bound
from qualtran.bloqs.chemistry.ising.walk_operator import (
    get_prepare_precision_from_eigenphase_precision,
    get_walk_operator_for_1d_ising_model,
)
from qualtran.bloqs.state_preparation import StatePreparationAliasSampling
from qualtran.linalg.lcu_util import sub_bit_prec_from_epsilon
from qualtran.symbolics import ceil, log2


def test_symbolic_precision_for_ising():
    eps, L, J, Gamma = sympy.symbols(r"\epsilon L J Gamma")
    qlambda = L * (J + Gamma)
    norm_H = get_1d_ising_hamiltonian_norm_upper_bound(L, J, Gamma)
    delta = get_prepare_precision_from_eigenphase_precision(eps, L, qlambda, norm_H)
    _ = sub_bit_prec_from_epsilon(L, delta / qlambda)

    assert sympy.simplify(delta - eps / (eps**2 + 1) * (2 * J * Gamma / (J + Gamma))) == 0


def test_symbolic_1d_ising_walk_op():
    n = 4
    J, Gamma = 1, -1
    eps = sympy.symbols(r"\epsilon", real=True, positive=True)
    walk, ham = get_walk_operator_for_1d_ising_model(n, eps)
    assert walk.sum_of_lcu_coefficients == n * (abs(J) + abs(Gamma))

    # check expression for probability bitsize `mu`
    assert isinstance(walk.prepare, StatePreparationAliasSampling)
    mu = walk.prepare.mu
    assert isinstance(mu, sympy.Expr)
    # sympy limitation: unable to match exact expressions
    assert str(mu.simplify()) == str(ceil(log2(2.0 * eps + 2.0 / eps)))
