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
from cirq_ft.algos.arithmetic_gates import LessThanEqualGate, LessThanGate
from cirq_ft.linalg.lcu_util import preprocess_lcu_coefficients_for_reversible_sampling

import qualtran.testing as qlt_testing
from qualtran import BloqBuilder, Register
from qualtran.bloqs.chemistry.thc_select import SelectTHC
from qualtran.cirq_interop import CirqGateAsBloq
from qualtran.testing import execute_notebook


def _make_select():
    from qualtran.bloqs.chemistry.thc import SelectTHC

    num_spat = 4
    num_mu = 8
    return SelectTHC(num_mu=num_mu, num_spat=num_spat)


def test_select_thc():
    num_mu = 10
    num_spin_orb = 2 * 4
    select = SelectTHC(num_mu=num_mu, num_spin_orb=num_spin_orb)
    qlt_testing.assert_valid_bloq_decomposition(select)


@pytest.mark.parametrize("theta", 2 * np.pi * np.random.random(10))
def test_interleaved_cliffords(theta):
    a, b = cirq.LineQubit.range(2)
    XY = cirq.Circuit([cirq.X(a), cirq.Y(b)])
    UXY = cirq.unitary(XY)
    RXY_ref = scipy.linalg.expm(-1j * theta * UXY / 2)
    C0 = [cirq.H(a), cirq.S(b) ** -1, cirq.H(b), cirq.CNOT(a, b)]
    C1 = [cirq.S(a), cirq.H(a), cirq.H(b), cirq.CNOT(a, b)]
    RXY = cirq.unitary(cirq.Circuit([C0, cirq.Rz(rads=theta)(b), cirq.inverse(C0)]))
    assert np.allclose(RXY, RXY_ref)
    YX = cirq.Circuit([cirq.Y(a), cirq.X(b)])
    UYX = cirq.unitary(YX)
    RYX_ref = scipy.linalg.expm(1j * theta * UYX / 2)
    RYX = cirq.unitary(cirq.Circuit([C1, cirq.Rz(rads=-theta)(b), cirq.inverse(C1)]))
    assert np.allclose(RYX.T, RYX_ref)
