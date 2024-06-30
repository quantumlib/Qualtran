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
import attrs
import numpy as np
import pytest

from qualtran import Bloq, Signature
from qualtran.bloqs.chemistry.trotter.ising import IsingXUnitary, IsingZZUnitary
from qualtran.bloqs.chemistry.trotter.trotterized_unitary import _trott_unitary, TrotterizedUnitary


def test_trotterized_unitary(bloq_autotester):
    bloq_autotester(_trott_unitary)


def test_construction_checks(bloq_autotester):
    class NotAnAttrsClass(Bloq):
        def __init__(self, angle: float, b: int):
            self.angle = angle
            self.b = b

        @property
        def signature(self) -> 'Signature':
            return Signature.build(a=1, b=1)

    with pytest.raises(ValueError, match=r'Bloq must be an attrs.*'):
        TrotterizedUnitary(
            bloqs=(NotAnAttrsClass(0.1, 2), NotAnAttrsClass(0.2, 2)),
            indices=(0, 1),
            coeffs=(0.55, 0.2),
            timestep=0.1,
        )

    @attrs.frozen
    class CustomSignature(Bloq):
        angle: float
        bitsize_a: int
        bitsize_b: int

        @property
        def signature(self) -> 'Signature':
            return Signature.build(a=self.bitsize_a, b=self.bitsize_b)

    with pytest.raises(ValueError, match=r'Bloqs must have the same.*'):
        TrotterizedUnitary(
            bloqs=(
                CustomSignature(angle=0.1, bitsize_a=2, bitsize_b=2),
                CustomSignature(angle=0.2, bitsize_a=2, bitsize_b=3),
            ),
            indices=(0, 1),
            coeffs=(0.55, 0.2),
            timestep=0.1,
        )

    @attrs.frozen
    class NoAngle(Bloq):
        bitsize_a: int
        bitsize_b: int

        @property
        def signature(self) -> 'Signature':
            return Signature.build(a=self.bitsize_a, b=self.bitsize_b)

    with pytest.raises(ValueError, match=r'Bloq must have a parameter named.*'):
        TrotterizedUnitary(
            bloqs=(NoAngle(2, 2), NoAngle(2, 2)), indices=(0, 1), coeffs=(0.55, 0.2), timestep=0.1
        )


@pytest.mark.parametrize('nsites', (2, 3, 4))
def test_trotterized_unitary_tensor_contract_suzuki_2(nsites):
    # e^{0.5 dt * H_x} e^{dt * H_zz} e^{dt * H_x}
    j_zz = 2
    gamma_x = 0.1
    dt = 0.01
    indices = (0, 1, 0)
    # factor of 2 for Rotation angle factors
    coeffs = (0.5 * gamma_x, j_zz, 0.5 * gamma_x)
    # zz_terms, x_terms = _build_1d_ising_pauli_terms(qubits, j_zz, gamma_x)
    zz_bloq = IsingZZUnitary(nsites=nsites, angle=2 * dt * j_zz)
    x_bloq = IsingXUnitary(nsites=nsites, angle=0.5 * 2 * dt * gamma_x)
    suzuki = TrotterizedUnitary(
        bloqs=(x_bloq, zz_bloq), indices=indices, coeffs=coeffs, timestep=dt
    )
    zz_mat = zz_bloq.tensor_contract()
    x_mat = x_bloq.tensor_contract()
    ref_step = x_mat @ zz_mat @ x_mat
    bloq_step = suzuki.tensor_contract()
    np.testing.assert_allclose(bloq_step, ref_step)


@pytest.mark.parametrize('nsites', (2, 3, 4))
def test_trotterized_unitary_tensor_contract_suzuki_4(nsites):
    j_zz = 2
    gamma_x = 0.1
    dt = 0.01
    indices = (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0)
    coeffs = np.array(
        [
            0.20724538589718786,
            0.4144907717943757,
            0.4144907717943757,
            0.4144907717943757,
            -0.12173615769156357,
            -0.6579630871775028,
            -0.12173615769156357,
            0.4144907717943757,
            0.4144907717943757,
            0.4144907717943757,
            0.20724538589718786,
        ]
    )
    coeffs = tuple([c * gamma_x if i == 0 else c * j_zz for (i, c) in zip(indices, coeffs)])
    bloqs = (IsingXUnitary(nsites=nsites, angle=0), IsingZZUnitary(nsites=nsites, angle=0))
    ref_step = np.eye(2**nsites)
    for i, c in zip(indices, coeffs):
        ref_step = attrs.evolve(bloqs[i], angle=2 * dt * c).tensor_contract() @ ref_step
    suzuki = TrotterizedUnitary(bloqs=bloqs, indices=indices, coeffs=coeffs, timestep=dt)
    bloq_step = suzuki.tensor_contract()
    np.testing.assert_allclose(bloq_step, ref_step)
