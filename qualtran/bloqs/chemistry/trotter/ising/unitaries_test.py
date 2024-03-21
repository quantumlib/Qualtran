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
import scipy.linalg

from qualtran.bloqs.chemistry.ising import get_1d_ising_pauli_terms
from qualtran.bloqs.chemistry.trotter.ising.unitaries import (
    _ising_x,
    _ising_zz,
    IsingXUnitary,
    IsingZZUnitary,
)


def test_ising_x(bloq_autotester):
    bloq_autotester(_ising_x)


def test_ising_zz(bloq_autotester):
    bloq_autotester(_ising_zz)


@pytest.mark.parametrize('nsites', (2, 3, 4))
def test_ising_test_unitaries(nsites):
    qubits = cirq.LineQubit.range(nsites)
    j_zz = 2
    gamma_x = 0.1
    dt = 0.01
    zz_terms, x_terms = get_1d_ising_pauli_terms(qubits, j_zz, gamma_x)
    exp_zz = scipy.linalg.expm(-1j * dt * cirq.PauliSum.from_pauli_strings(zz_terms).matrix())
    exp_zz_test = IsingZZUnitary(nsites=nsites, angle=2 * dt * j_zz).tensor_contract()
    np.testing.assert_allclose(exp_zz_test, exp_zz)
    exp_x = scipy.linalg.expm(-1j * dt * cirq.PauliSum.from_pauli_strings(x_terms).matrix())
    exp_x_test = IsingXUnitary(nsites=nsites, angle=2 * dt * gamma_x).tensor_contract()
    np.testing.assert_allclose(exp_x_test, exp_x)
