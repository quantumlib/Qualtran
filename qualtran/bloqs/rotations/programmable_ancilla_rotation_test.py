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
import numpy as np
import pytest
import sympy

from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.bloqs.basic_gates._shims import Measure
from qualtran.bloqs.rotations.programmable_ancilla_rotation import (
    _zpow_programmed_ancilla,
    _zpow_programmed_ancilla_symb,
    _zpow_using_programmed_ancilla,
    _zpow_using_programmed_ancilla_symb,
    _zpow_using_programmed_ancilla_symb_rounds,
    ZPowProgrammedAncilla,
    ZPowUsingProgrammedAncilla,
)
from qualtran.resource_counting.t_counts_from_sigma import t_counts_from_sigma
from qualtran.symbolics import ceil, log2


def test_rz_resource_state_examples(bloq_autotester):
    bloq_autotester(_zpow_programmed_ancilla)
    bloq_autotester(_zpow_programmed_ancilla_symb)


def test_rz_resource_state_tensor_on_random_angles():
    rng = np.random.default_rng(42)

    for exponent in rng.random(size=5):
        bloq = ZPowProgrammedAncilla(exponent=exponent)
        np.testing.assert_allclose(
            bloq.tensor_contract(), np.array([1, np.exp(1j * np.pi * exponent)]) / np.sqrt(2)
        )


def test_rz_via_par_examples(bloq_autotester):
    if bloq_autotester.check_name == 'serialize':
        pytest.skip('Cannot serialize Measure')

    bloq_autotester(_zpow_using_programmed_ancilla)
    bloq_autotester(_zpow_using_programmed_ancilla_symb)
    bloq_autotester(_zpow_using_programmed_ancilla_symb_rounds)


def test_t_cost_including_rotations():
    theta, eps, p = sympy.symbols(r'theta, \epsilon, p')
    bloq = ZPowUsingProgrammedAncilla.from_failure_probability(
        exponent=theta, max_fail_probability=p, eps=eps
    )
    _, sigma = bloq.call_graph()
    n_rot = ceil(log2(1 / p))
    eps_per_rot = eps / n_rot
    # TODO(#1250): Ideally, we should be able to get this symbolic cost via more standard
    # ways like `bloq.t_complexity()` or `get_cost_value(bloq, QECGatesCost())` but both
    # of them ignore epsilon right now.
    assert t_counts_from_sigma(sigma) == ceil(1.149 * log2(1.0 / eps_per_rot) + 9.2) * n_rot


def test_rz_via_par_call_graphs():
    _, sigma_rz = ZPowUsingProgrammedAncilla(np.pi / 4).call_graph(max_depth=1)
    assert sigma_rz == {
        CNOT(): 2,
        XGate(): 2,
        ZPowProgrammedAncilla(np.pi / 4, eps=1e-11 / 2): 1,
        ZPowProgrammedAncilla(np.pi / 2, eps=1e-11 / 2): 1,
        Measure(): 2,
    }

    phi, eps = sympy.symbols(r"\phi \epsilon")

    _, sigma_rz_symb = ZPowUsingProgrammedAncilla(phi, eps=eps, n_rounds=3).call_graph(max_depth=1)
    assert sigma_rz_symb == {
        CNOT(): 3,
        XGate(): 3,
        ZPowProgrammedAncilla(phi, eps / 3): 1,
        ZPowProgrammedAncilla(2 * phi, eps / 3): 1,
        ZPowProgrammedAncilla(4 * phi, eps / 3): 1,
        Measure(): 3,
    }

    phi0, n = sympy.symbols(r"_\phi0 n")
    _, sigma_rz_symb_rounds = ZPowUsingProgrammedAncilla(phi, eps=eps, n_rounds=n).call_graph(
        max_depth=1
    )
    assert sigma_rz_symb_rounds == {
        CNOT(): n,
        XGate(): n,
        ZPowProgrammedAncilla(phi0, eps / n): n,
        Measure(): n,
    }
